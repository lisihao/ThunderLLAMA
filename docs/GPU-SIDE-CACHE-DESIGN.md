# GPU-Side Cache 技术设计文档

> 目标：90-100x prefill skip speedup
> 时间：2-3 周
> 状态：设计阶段

---

## 1. 背景

### 当前性能

| 配置 | Speedup | LMCache 加载 | 瓶颈 |
|------|---------|--------------|------|
| CPU-side (磁盘) | 65x | 0.016s | 磁盘 I/O + CPU→GPU 传输 |

### 性能瓶颈分解

**LMCache 加载时间 0.016s 组成**（~800 tokens, 3-4 chunks）：
- 磁盘读取（mmap）：~6ms
- CPU → GPU 传输：~8ms
- Metal kernel 调用：~2ms

**更长 prompt 的瓶颈更明显**（~1500 tokens, 6 chunks）：
- LMCache 加载：0.061s
- 其中 CPU → GPU 传输：~35ms（57%）

### 目标

- **消除 CPU → GPU 传输**（最大瓶颈）
- **减少磁盘 I/O**（GPU 是热 cache）
- **目标 speedup：90-100x**

---

## 2. 架构设计

### 2.1 三层存储架构

```
┌─────────────────────────────────────────────────────────────┐
│  L1: GPU Metal Buffer Pool (热 cache)                      │
│  - 大小: 10GB                                                │
│  - 位置: GPU 统一内存（M4 Pro Unified Memory Architecture）│
│  - 特点: 零拷贝访问，Metal kernel 直接读取                  │
│  - 策略: LRU eviction                                       │
├─────────────────────────────────────────────────────────────┤
│  L2: CPU 内存 (温 cache, 当前已有)                         │
│  - 大小: 2GB                                                 │
│  - 特点: 快速访问，但需要 CPU → GPU 传输                    │
│  - 策略: LRU eviction to L3                                 │
├─────────────────────────────────────────────────────────────┤
│  L3: 磁盘 mmap (冷 cache, 当前已有)                        │
│  - 大小: 100GB                                               │
│  - 特点: 持久化，跨进程共享                                  │
│  - 策略: LRU eviction                                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

#### Prefill Skip 流程（L1 命中）

```
┌──────────┐
│ llama    │
│ context  │
└────┬─────┘
     │
     │ 1. Check L1 GPU pool
     ▼
┌─────────────────┐
│ L1: GPU Pool    │  ←─── Metal kernel 直接读取（零拷贝）
│ (Metal Buffer)  │
└─────────────────┘
     │
     │ 2. Blit to KV cache tensors (GPU 内部拷贝，~1ms)
     ▼
┌─────────────────┐
│ KV Cache        │
│ (ggml tensors)  │
└─────────────────┘
     │
     │ 3. Skip forward pass
     ▼
   Done
```

**性能分析**：
- L1 查找：~0.001ms（hash lookup）
- GPU blit：~1-2ms（Metal 内部拷贝，带宽 >400 GB/s）
- **总计：~0.002-0.003s**（vs 当前 0.016s）

#### Prefill Skip 流程（L1 未命中，L2/L3 命中）

```
L2/L3 hit → Read to CPU → Upload to L1 GPU pool → Blit to KV cache
                  ↓              ↓                      ↓
               ~6ms          ~8ms                    ~2ms
            (当前瓶颈)     (最大瓶颈)               (可接受)
```

**优化后**：
- L2/L3 hit → Upload to L1 → Next request hits L1 (warm up)
- 第一次慢（~16ms），第二次快（~2ms）

---

## 3. 关键技术

### 3.1 Metal Buffer Pool

```cpp
class MetalBufferPool {
private:
    id<MTLBuffer> pool_buffer_;          // 10GB pre-allocated
    std::map<uint64_t, size_t> offset_map_;  // key_hash -> offset
    std::list<uint64_t> lru_list_;       // LRU order
    std::map<uint64_t, std::list<uint64_t>::iterator> lru_index_;
    size_t next_offset_ = 0;
    const size_t pool_size_ = 10ULL * 1024 * 1024 * 1024;  // 10GB

public:
    // Initialize pool with MTLResourceStorageModeShared
    // (M4 Pro unified memory - CPU/GPU can both access)
    void init(id<MTLDevice> device);

    // Store chunk in GPU pool
    // Returns offset, or -1 if eviction needed
    size_t store(uint64_t key_hash, const void* k_data, size_t k_size,
                                     const void* v_data, size_t v_size);

    // Get chunk offset (for Metal kernel)
    size_t get_offset(uint64_t key_hash);

    // Evict LRU chunk
    void evict_lru();
};
```

### 3.2 Metal Blit Encoder

```cpp
// Blit from pool to KV cache tensors (GPU-side copy)
void blit_from_pool_to_kv_cache(
    id<MTLCommandBuffer> cmd_buf,
    id<MTLBuffer> pool_buffer,
    size_t pool_offset,
    id<MTLBuffer> k_tensor,
    id<MTLBuffer> v_tensor,
    size_t chunk_size
) {
    id<MTLBlitCommandEncoder> blit = [cmd_buf blitCommandEncoder];

    [blit copyFromBuffer:pool_buffer
           sourceOffset:pool_offset
                toBuffer:k_tensor
       destinationOffset:0
                    size:chunk_size];

    [blit endEncoding];
}
```

**性能优势**：
- Metal blit 使用 GPU DMA（Direct Memory Access）
- 带宽：>400 GB/s（vs PCIe 4.0 ~32 GB/s）
- **10x 带宽提升**

### 3.3 Unified Memory 优势

**M4 Pro Unified Memory Architecture**：
- CPU 和 GPU 共享同一块物理内存
- 使用 `MTLResourceStorageModeShared`
- **CPU 可以直接写入 GPU buffer**（无需显式拷贝）

```cpp
// Allocate shared buffer (CPU/GPU both accessible)
id<MTLBuffer> buffer = [device newBufferWithLength:pool_size_
                                           options:MTLResourceStorageModeShared];

// CPU 写入（零拷贝）
void* cpu_ptr = [buffer contents];
memcpy(cpu_ptr + offset, k_data, k_size);

// GPU 读取（同一块内存）
// Metal kernel can directly access this buffer
```

---

## 4. 实现计划

### Week 1: GPU Buffer Pool 基础框架

**文件修改**：
- `src/metal-buffer-pool.h` (新建)
- `src/metal-buffer-pool.cpp` (新建)
- `src/thunder-lmcache-storage.h` (添加 L1 support)
- `src/thunder-lmcache-storage.cpp` (添加 L1 logic)

**里程碑**：
- [x] 创建 `MetalBufferPool` class
- [x] 预分配 10GB shared buffer
- [x] 实现 `store()`/`get_offset()` 接口
- [x] LRU eviction 逻辑
- [ ] 单元测试

**代码示例**：
```cpp
// metal-buffer-pool.h
class MetalBufferPool {
    id<MTLDevice> device_;
    id<MTLBuffer> pool_buffer_;
    // ...
};

// thunder-lmcache-storage.cpp
thunder_kv_chunk * ThunderChunkStorage::get(const thunder_kv_chunk_key & key) {
    // 1. Check L1 GPU pool first
    if (metal_pool_ && metal_pool_->has(key_hash)) {
        size_t offset = metal_pool_->get_offset(key_hash);
        // Return GPU offset (caller will blit)
        // ...
    }

    // 2. Check L2/L3 (fallback)
    // ...
}
```

### Week 2: Metal Kernel 集成

**文件修改**：
- `src/llama-context.cpp` (修改 prefill skip 逻辑)
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` (Metal blit)

**里程碑**：
- [ ] Prefill skip 使用 GPU pool
- [ ] Metal blit encoder 集成
- [ ] Benchmark 对比 CPU-side cache
- [ ] 性能分析（Metal GPU profiler）

**代码示例**：
```cpp
// llama-context.cpp
if (lmcache_can_skip_compute) {
    // Get GPU pool offset
    size_t gpu_offset = storage->get_gpu_offset(chunk_key);

    if (gpu_offset != (size_t)-1) {
        // Blit from GPU pool to KV cache
        blit_from_pool_to_kv_cache(metal_cmd_buf, gpu_offset, k_tensor, v_tensor);
        // Skip forward pass
        continue;
    } else {
        // Fallback to L2/L3
        // ...
    }
}
```

### Week 3: LRU 策略和最终优化

**文件修改**：
- `src/metal-buffer-pool.cpp` (优化 LRU)
- `src/thunder-lmcache-storage.cpp` (L1 <-> L2/L3 promotion)

**里程碑**：
- [ ] GPU pool 满时，evict to L2
- [ ] L2/L3 热 chunk promote to L1
- [ ] 多 slot 并发测试（4 slots）
- [ ] 端到端 benchmark（90-100x 目标）

**代码示例**：
```cpp
// metal-buffer-pool.cpp
void MetalBufferPool::evict_lru() {
    if (lru_list_.empty()) return;

    uint64_t victim_hash = lru_list_.back();
    lru_list_.pop_back();

    size_t offset = offset_map_[victim_hash];
    offset_map_.erase(victim_hash);

    // Copy back to L2/L3 if needed
    // ...

    // Mark this offset as free
    free_offsets_.push_back(offset);
}
```

---

## 5. 性能预测

### 5.1 理论分析

**当前瓶颈**（~800 tokens, 3-4 chunks）：
- 磁盘读取：~6ms
- CPU → GPU：~8ms（**最大瓶颈**）
- Metal kernel：~2ms
- **总计：~16ms**

**GPU-side cache**（L1 命中）：
- GPU pool 查找：~0.001ms
- Metal blit：~2ms
- **总计：~2ms**

**加速比**：16ms / 2ms = **8x 加速**

### 5.2 端到端预测

| Prompt 长度 | 当前 Speedup | GPU-side Speedup | 提升 |
|-------------|-------------|------------------|------|
| ~800 tokens | 65x | **~520x** | 8x |
| ~1500 tokens | 62x | **~500x** | 8x |

**但是**，考虑到生成阶段的时间（~1ms/token）：
- Generate 1 token: ~0.014s
- LMCache 加载优化：16ms → 2ms（节省 14ms）
- **实际 speedup**：受 generate 时间限制

### 5.3 实际预期

**保守估计**（考虑 generate 时间）：
- Cold start: 3.761s
- GPU-side skip: ~0.030s（LMCache 2ms + generate 1 token 14ms + overhead 14ms）
- **Speedup: ~125x**

**理想场景**（纯 prefill，generate 0 tokens）：
- Cold start: 0.700s
- GPU-side skip: ~0.007s
- **Speedup: ~100x** ✅

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GPU 内存不足（10GB pool + 模型 17GB） | 中 | 高 | 实现 LRU + disk fallback |
| Metal API 复杂度 | 中 | 中 | 参考 ggml-metal.cpp 现有代码 |
| 多 slot 冲突 | 低 | 中 | Buffer pool 设计时考虑并发 |
| 统一内存限制（M4 Pro 48GB 总共） | 低 | 高 | 监控内存使用，动态调整 pool size |

---

## 7. 验证计划

### 7.1 单元测试

- [ ] `MetalBufferPool` 分配/释放测试
- [ ] LRU eviction 正确性测试
- [ ] 多线程并发测试

### 7.2 集成测试

- [ ] Prefill skip 使用 GPU pool
- [ ] L1/L2/L3 三层 promotion 测试
- [ ] 多 slot 并发测试

### 7.3 性能测试

- [ ] Benchmark: L1 vs L2/L3
- [ ] 端到端 speedup（目标 90-100x）
- [ ] Metal GPU profiler 分析

---

## 8. 后续优化

### 8.1 Async Prefetch

- L2/L3 → L1 异步预取
- 预测下一个 chunk（基于 access pattern）

### 8.2 Compression

- GPU 端压缩（Metal kernel）
- 减少 pool 大小需求

### 8.3 Multi-GPU

- 分布式 LMCache（多 GPU 场景）

---

*GPU-Side Cache Design v1.0*
*创建于: 2026-03-14*
*目标: 90-100x prefill skip speedup*
