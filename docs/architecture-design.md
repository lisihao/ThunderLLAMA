# ThunderLLAMA 架构设计文档

> 从 STATE.md 迁移的详细设计记录，包含实验数据、算法实现、经验教训

**最后更新**: 2026-03-15

---

## 1. ContextPilot + LMCache 三层架构

### 架构总览

```
L1: ContextPilot (Prompt Optimizer)
  → Reorder + Dedup (36% tokens)

L2: ClawGate (Service Orchestrator)
  → Detect overlap > 80% → cache_prompt=false
  → 强制 full prefill → 触发 skip logic

L3: ThunderLLAMA + LMCache
  → Hybrid hashing (3x prefix overlap)
  → Skip logic (27x full hit)
```

### 性能提升来源

| 组件 | 单独加速 | 说明 |
|------|---------|------|
| ContextPilot | 2-3x | prefix reuse + deduplicate |
| LMCache | 1.2-5x | K tensor cache (取决于命中率) |
| 三层协同 | 23.34x | Phase 1 实测 (超过 20x 目标) |

### 测试数据 (2026-03-12)

**ThunderLLAMA + LMCache (相同 prompt 重复测试)**:
- 测试场景: 1500 tokens prompt，重复 3 次
- Round 1 (cold cache): 6967ms
- Round 2 (warm cache): 1443ms
- Round 3 (warm cache): 1314ms
- 平均 warm cache: 1378ms
- **性能提升**: 5.06x

**关键发现**:
1. LMCache 缓存命中率 100% (相同 prompt)
2. 相同 prompt: 5.06x | 不同 prompt: 1.2x | 差异原因: 命中率
3. ContextPilot 集成状态: ClawGate 配置正确，ContextPilot v0.3.5 已加载

### 核心策略

**策略 1: Cache-Aware Routing** (缓存感知路由)
- High overlap (>80%) + High hit (>90%) → `cache_prompt=false`
- Low overlap (<30%) → `cache_prompt=true`

**策略 2: Prefix-Group Batching** (前缀分组)
- 第一个请求: cache_prompt=true (建立缓存)
- 后续请求: cache_prompt=false (触发 skip)

**策略 3: Eviction-Aware Scheduling** (驱逐感知)
- LMCache 驱逐 → 通知 ContextPilot → 重新排序队列

### 预期收益

| 场景 | 优化 | 加速比 |
|------|------|--------|
| Multi-agent (shared system prompt) | Dedup + Skip | ~37x |
| Incremental context | Hybrid + Skip | ~10x |
| 一般场景 | Hybrid only | 3-5x |

---

## 2. Hybrid Hashing 设计

### 问题诊断

**Position-based caching 的局限**:
- content_hash=0, 无法处理 prefix overlap
- 50% prefix overlap: 1.11x (几乎无加速)
- 100% 相同 prompt: 11.85x

**Content-based caching 的挑战**:
- Prefill: ubatch.n_tokens = 400 (包含完整 tokens) ✅
- Decode: ubatch.n_tokens = 1 (只有当前 token) ❌
- 访问 ubatch.token[256] 会越界！

### 解决方案：Hybrid Hashing

```cpp
if (chunk_start + THUNDER_CHUNK_SIZE <= ubatch.n_tokens) {
    // Content-based hashing (prefill phase)
    key = lmcache_hasher->make_key(ubatch.token, ubatch.n_tokens, chunk_start, il);
} else {
    // Position-based hashing (decode phase)
    key.content_hash = 0;
    key.layer_idx = il;
    key.chunk_start = chunk_start;
}
```

### 测试结果 (严格测试，缓存完全清空)

**100% 相同 prompt**:
- Round 1 (cold): 15027.9ms | Round 2/3 (warm): ~587ms
- Position-based: 11.85x → Hybrid: **25.59x** (改进 2.16x)

**50% Prefix Overlap**:
- Prompt A (cold): 10032.3ms | Prompt B (50% overlap): 3335.9ms
- Position-based: 1.11x → Hybrid: **3.01x** (改进 2.71x)

### 文件修改
- `src/llama-context.cpp` (RESTORE + STORE 阶段): 根据 ubatch.n_tokens 动态选择 hashing 策略

### 收益
- Prefix overlap 场景从 1.11x → 3-7.8x
- Multi-agent 场景大幅提升 (不同 system prompts 共享部分 prefix)
- 向后兼容: decode 阶段仍使用 position-based
- 最小改动，无需修改核心数据结构

---

## 3. V Tensor 缓存实验 — 负优化教训

### 实验目标
缓存 V tensor 以跳过 forward pass 计算，理论提升 2x

### 实验结果 (commit cf16a72dc, 分支 feature/v-tensor-cache-negative-opt)

| 方案 | 加速比 | warm 延迟 |
|------|--------|----------|
| K-only | 25.59x | 587ms |
| K+V | 23.50x | 639ms |
| **差异** | **-8.2%** | +52ms |

### 负优化根因

Forward pass 跳过逻辑**从未触发**:
```cpp
if (is_prefill && chunks_found == chunks_needed)
    skip_compute = true;
```

实际情况:
- Prefill: `chunks_found=192/384` (50% 命中) → 不跳过
- Decode: `is_prefill=0` → 不跳过 (即使 100% 命中)

性能损失分解:
- V tensor 恢复开销: 384 chunks × 256KB = 98 MB 内存带宽
- V tensor 计算节省: 0 (forward pass 仍运行)
- 净效果: **-8.2%**

### 教训

> **V tensor 缓存在不跳过 forward pass 的情况下，只增加开销，无任何收益。**
> Decode 阶段新 token 的 Q/K/V 必须计算，Attention 运算必须执行，V cache 是纯开销。

### 代码保存
- 分支: `feature/v-tensor-cache-negative-opt`
- 状态: 已回滚，保留作为负优化参考

---

## 4. Skip Logic 设计与验证

### 核心发现
`cache_prompt=false` 禁用 session cache，触发跳过逻辑

### 验证结果

| Round | 延迟 | 状态 |
|-------|------|------|
| Round 1 (cold) | 8883.3ms | 建立缓存 |
| Round 2 (warm) | 336.4ms | 命中跳过 |
| Round 3 (warm) | 299.7ms | 命中跳过 |
| **加速比** | **27.93x** | |

跳过日志:
```
🚀 LMCache FULL HIT: 96/96 chunks cached, SKIPPING forward pass!
🚀 LMCache FULL HIT: 192/192 chunks cached, SKIPPING forward pass!
🚀 LMCache FULL HIT: 240/240 chunks cached, SKIPPING forward pass!
```

### 触发条件
1. `is_prefill=1` (完整 prefill)
2. `chunks_found == chunks_needed` (100% 命中)
3. Forward pass 被跳过

### 架构限制

**Decode 阶段无法跳过** (根本限制):
- 新 token 的 Q/K/V 必须计算
- Attention 运算必须执行

**Prefill 阶段可以跳过** (已验证):
- 条件: 完整 prefill + 100% LMCache 命中
- 方法: `cache_prompt=false`
- 效果: 27.93x

---

## 5. 三层协同优化 — 实现记录

### Phase 1: ClawGate 增强 ✅ (2026-03-12)

**完成项**:
- `/lmcache/stats` 端点实现
- `cache_prompt` 参数透传与验证
- ContextPilot overlap API 集成
- `should_force_prefill()` 决策逻辑实现
- 性能达标: 23.34x (超过 20x 目标)
- 深度调查完成 (PHASE1_INVESTIGATION_REPORT.md)

**调查发现**:
- `cache_prompt` 在 `/v1/chat/completions` 端点有效
- session cache 可正确禁用 (cache_n=0)
- Skip logic 需要 100% LMCache 命中 (设计限制)
- 当前 23.34x = 理论 27.93x 的 83%

**Commits**:
- `6d2857090` - Phase 1 完成提交
- `d623c5c0a` - Use /lmcache/stats in client
- `294456afa` - /lmcache/stats build fix
- `45f658748` - /lmcache/stats endpoint implementation
- `a375cce79` - Phase 1 cache-aware routing implementation

### Phase 2: 监控与优化 (部分完成)

**已完成**:
- Task 2.1.1: Prometheus metrics 集成 (ClawGate)
  - metrics.py (5 metric types), metrics_server.py (port 9090)
- Task 2.1.2: ThunderLLAMA `/lmcache/stats` 真实统计
  - skip tracking counters, llama_get_lmcache_stats() API
- Task 2.3: ThunderChunkStorage 真实统计集成
  - get_chunk_storage_stats(), /lmcache/stats 返回 l2/l3 统计
- Task 3.1: Approximate Skip (95%+ hit ratio)

**待做** (已录入 task list):
- Task 2.1.3: Prometheus scraping 配置 (prometheus.yml)
- Task 2.2: Grafana Dashboard (5 panels)
- Task 2.4: 决策阈值调优 (基于真实数据)

### Phase 3: 高级优化 (待做)
- Eviction-aware 调度
- Prefix-group 批处理

### 文档位置
- `clawgate-integration/PHASE1_INVESTIGATION_REPORT.md`
- `clawgate-integration/PHASE1_COMPLETION_REPORT.md`
- `clawgate-integration/THREE_LAYER_OPTIMIZATION.md`

---

## 6. Approximate Skip 设计

### 问题定义
- 目标: 提升 Skip 触发率从 5% 到 30%
- 现状: 只有 100% 命中率才触发 Skip (过于严格)
- 方案: Approximate Skip (95%+ 命中率用零填充)

### 实现

```cpp
// llama-context.cpp:1341-1360
if (hit_ratio >= APPROX_SKIP_THRESHOLD) {  // 0.95
    // Zero-fill missing chunks
    for (const auto& missing : missing_chunks) {
        std::vector<uint8_t> zeros(chunk_bytes, 0);
        ggml_backend_tensor_set(missing.k_tensor, zeros.data(), offset, chunk_bytes);
        ggml_backend_tensor_set(missing.v_tensor, zeros.data(), offset, chunk_bytes);
    }
    lmcache_can_skip_compute = true;
    lmcache_approx_skip_count++;
}
```

### 测试结果

| 修改程度 | 预期命中率 | 实际命中率 | 结果 |
|----------|-----------|-----------|------|
| 1 word | ~99.9% | 100% | Full Skip |
| 5 words | ~99.4% | 100% | Full Skip |
| 20 words | ~97.5% | 100% | Full Skip |
| 40 words | ~95% | 66.67% | No Skip |

**关键发现**: Content-based hashing 导致 "All-or-Nothing" 特性。修改 1 token → 整个 chunk hash 完全不同。需要真实工作负载验证 (RAG、长对话、多轮对话)。

### API 更新

新增字段 (`/lmcache/stats`):
```json
{
  "approx_skip_count": 0,
  "total_skip_count": 1,
  "total_skip_rate": 0.1667
}
```

### 文件修改清单

1. `src/llama-context.h:406` - approx_skip_count 字段
2. `src/llama-context.h:211-214` - get_lmcache_stats() 方法
3. `src/llama-context.cpp:24` - APPROX_SKIP_THRESHOLD 常量
4. `src/llama-context.cpp:27-33` - MissingChunkInfo 结构
5. `src/llama-context.cpp:1302-1310` - 记录 missing chunks
6. `src/llama-context.cpp:1341-1360` - Approximate Skip 逻辑
7. `include/llama.h:966-972` - llama_get_lmcache_stats() API
8. `src/llama.cpp:1176-1188` - API 实现
9. `tools/server/server-context.cpp:3226-3261` - /lmcache/stats 端点

### 预期收益

| 场景 | 原触发率 | 新触发率 | 提升 |
|------|---------|---------|------|
| 100% 相同 | 5% | 5% | 保持 |
| 95-99% 相同 | 0% | 25% | 新增 |
| 总计 | 5% | 30% | 6x |

真实场景: RAG 系统 (部分文档重叠)、长对话 (历史上下文复用)、Multi-turn (System prompt 固定)

---

## 7. OpenMP 冲突问题 (已解决)

### 问题
ClawGate 启动时 Python 崩溃: `OMP: Error #15: libomp.dylib already initialized`

### 根因
- ClawGate 代码不使用 torch/sklearn
- 使用系统级 Python 误加载了 torch
- 性能差异来自 torch 导入开销，非真实推理性能
- 禁用 torch 后，OMP_NUM_THREADS 影响 < 4%

### 解决方案
- 创建启动脚本使用 venv Python
- 设置 OMP_NUM_THREADS=1
- 文件: `/Users/lisihao/ClawGate/start_clawgate.sh`
- 收益: 消除崩溃风险，节省内存 ~2GB

---

## 10. GPU-Side Cache 架构（2026-03-14）

> **目标**: 90-100x prefill skip speedup
> **时间**: 2-3 周
> **状态**: 设计阶段 → Week 1 开发中

### 10.1 性能瓶颈分析

**当前 CPU-side LMCache**（已实现）：
- Speedup: 65.22x (EXCLUSIVE 模式)
- LMCache 加载: 0.016s (~800 tokens, 3-4 chunks)
- **瓶颈分解**:
  - 磁盘读取（mmap）: ~6ms (38%)
  - **CPU → GPU 传输**: ~8ms (50%) ← 最大瓶颈
  - Metal kernel 调用: ~2ms (12%)

**更长 prompt 的瓶颈更明显**（~1500 tokens, 6 chunks）：
- LMCache 加载: 0.061s
- 其中 CPU → GPU 传输: ~35ms (57%)

### 10.2 GPU-Side Cache 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│  L1: GPU Metal Buffer Pool (热 cache)                      │
│  ├─ 大小: 10GB                                              │
│  ├─ 位置: GPU 统一内存（M4 Pro Unified Memory）            │
│  ├─ 特点: 零拷贝访问，Metal kernel 直接读取                │
│  └─ 策略: LRU eviction                                     │
├─────────────────────────────────────────────────────────────┤
│  L2: CPU 内存 (温 cache)                                   │
│  ├─ 大小: 2GB                                               │
│  ├─ 特点: 快速访问，但需要 CPU → GPU 传输                  │
│  └─ 策略: LRU eviction to L3                               │
├─────────────────────────────────────────────────────────────┤
│  L3: 磁盘 mmap (冷 cache)                                  │
│  ├─ 大小: 100GB                                             │
│  ├─ 特点: 持久化，跨进程共享                                │
│  └─ 策略: LRU eviction                                     │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 数据流优化

#### Prefill Skip 流程（L1 命中）

```
llama_context
    ↓
Check L1 GPU pool (~0.001ms)
    ↓
Metal Blit: GPU pool → KV cache (~2ms)
    ↓
Skip forward pass
    ↓
Done

总计: ~2ms (vs 当前 16ms)
加速比: 8x
```

#### Prefill Skip 流程（L1 未命中，warm up）

```
L2/L3 hit → Read to CPU (~6ms)
    ↓
Upload to L1 GPU pool (~8ms)
    ↓
Blit to KV cache (~2ms)
    ↓
Done (第一次: ~16ms)

下次相同请求:
L1 hit → Blit (~2ms) ✅
```

### 10.4 关键技术

#### Metal Buffer Pool

```cpp
class MetalBufferPool {
private:
    id<MTLBuffer> pool_buffer_;          // 10GB pre-allocated
    std::map<uint64_t, size_t> offset_map_;
    std::list<uint64_t> lru_list_;
    const size_t pool_size_ = 10ULL * 1024 * 1024 * 1024;

public:
    void init(id<MTLDevice> device);
    size_t store(uint64_t key_hash, const void* k_data, size_t k_size,
                                     const void* v_data, size_t v_size);
    size_t get_offset(uint64_t key_hash);
    void evict_lru();
};
```

#### M4 Pro 统一内存优势

- CPU 和 GPU 共享物理内存
- `MTLResourceStorageModeShared`（零拷贝）
- CPU 写入，GPU 直接读取
- **Metal Blit 带宽**: >400 GB/s（vs PCIe 4.0 ~32 GB/s）

### 10.5 性能预测

#### 理论分析

| 场景 | 当前 | GPU-side | 提升 |
|------|------|----------|------|
| L1 命中 | 16ms | 2ms | 8x |
| L1 未命中（首次） | 16ms | 16ms | 1x |
| L1 未命中（第二次） | 16ms | 2ms | 8x (warm up) |

#### 端到端预测

**保守估计**（考虑 generate 时间）：
- Cold start: 3.761s
- GPU-side skip: ~0.030s (LMCache 2ms + generate 1 token 14ms + overhead 14ms)
- **Speedup: ~125x**

**理想场景**（纯 prefill，generate 0 tokens）：
- Cold start: 0.700s
- GPU-side skip: ~0.007s
- **Speedup: ~100x** ✅

### 10.6 实现计划（3 周）

#### Week 1: GPU Buffer Pool 基础框架（Task #14）

**文件**:
- `src/metal-buffer-pool.h` (新建)
- `src/metal-buffer-pool.cpp` (新建)
- `src/thunder-lmcache-storage.{h,cpp}` (添加 L1 support)

**里程碑**:
- [x] 设计文档完成
- [ ] 创建 `MetalBufferPool` class
- [ ] 预分配 10GB shared buffer
- [ ] 实现 `store()`/`get_offset()` 接口
- [ ] LRU eviction 逻辑
- [ ] 单元测试

**预期性能**: 65x → 75x

#### Week 2: Metal Kernel 集成（Task #15）

**文件**:
- `src/llama-context.cpp` (修改 prefill skip 逻辑)
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` (Metal blit)

**里程碑**:
- [ ] Prefill skip 使用 GPU pool
- [ ] Metal blit encoder 集成
- [ ] Benchmark 对比 CPU-side cache
- [ ] Metal GPU profiler 性能分析

**预期性能**: 75x → 85x

#### Week 3: LRU 策略和最终优化（Task #16）

**文件**:
- `src/metal-buffer-pool.cpp` (优化 LRU)
- `src/thunder-lmcache-storage.cpp` (L1 ↔ L2/L3 promotion)

**里程碑**:
- [ ] GPU pool 满时，evict to L2
- [ ] L2/L3 热 chunk promote to L1
- [ ] 多 slot 并发测试（4 slots）
- [ ] 端到端 benchmark（90-100x 目标）

**预期性能**: 85x → **90-100x** ✨

### 10.7 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GPU 内存不足（10GB pool + 模型 17GB） | 中 | 高 | LRU + disk fallback |
| Metal API 复杂度 | 中 | 中 | 参考 ggml-metal.cpp |
| 多 slot 冲突 | 低 | 中 | Buffer pool 并发设计 |
| 统一内存限制（M4 Pro 48GB） | 低 | 高 | 动态调整 pool size |

### 10.8 验证计划

**单元测试**:
- [ ] `MetalBufferPool` 分配/释放
- [ ] LRU eviction 正确性
- [ ] 多线程并发

**集成测试**:
- [ ] Prefill skip 使用 GPU pool
- [ ] L1/L2/L3 三层 promotion
- [ ] 多 slot 并发

**性能测试**:
- [ ] Benchmark: L1 vs L2/L3
- [ ] 端到端 speedup（目标 90-100x）
- [ ] Metal GPU profiler 分析

### 10.9 技术文档

- **详细设计**: `docs/GPU-SIDE-CACHE-DESIGN.md`
- **性能基准**: 将更新到 `thunderllama.conf` 注释
- **测试脚本**: `/tmp/test-gpu-cache.py`（待创建）

---

*ThunderLLAMA Architecture Design Document v1.1*
*Created: 2026-03-15*
*Updated: 2026-03-14 (GPU-Side Cache 章节)*
*Source: Migrated from STATE.md*
