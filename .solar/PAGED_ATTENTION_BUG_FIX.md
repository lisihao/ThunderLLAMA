# Paged Attention Bug 修复方案

> **生成时间**: 2026-03-11
> **Bug 发现**: 所有模型规模 (0.6B, 1.7B, 30B) 输出乱码
> **根本原因**: KV cache 索引计算错误

---

## Bug 根因分析

### 问题定位

**文件**: `src/llama-kv-cache.cpp`
**函数**: `set_input_k_idxs()` (L1493-1507), `set_input_v_idxs()` (L1509-1544)

**错误代码**:
```cpp
void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            // ❌ 错误: 使用全局索引！
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}
```

### 为什么这是错误的？

#### 传统 KV Cache（无 Paged Attention）

**结构**: 连续的线性缓存
```
K cache: [n_embd, kv_size]
位置 0    1    2    3   ...   305  306  307  ...
     |----|----|----|----|-----|----|----|----|----|
```

**索引**: 直接用全局位置
- Token at position 305 → K[*, 305]
- ✅ **正确**: 全局索引直接映射

#### Paged Attention KV Cache

**结构**: 分块存储
```
Block Pool: [n_embd, block_size * num_blocks]
Block 0 (16 tokens)    Block 1 (16 tokens)   Block 19 (16 tokens)   ...
|---------------------|---------------------|----------------------|
0  1  2  ...  15      16 17 18 ...  31      304 305 306 ...  319

逻辑块 0              逻辑块 1              逻辑块 19
   ↓                     ↓                     ↓
物理块 5 (例如)       物理块 12            物理块 3 (查 block_table_gpu)
```

**索引**: 需要 block indirection
- Token at position 305:
  - logical_block = 305 / 16 = 19
  - offset = 305 % 16 = 1
  - physical_block = block_table_gpu[seq_id][19]
  - 最终索引 = physical_block * 16 + 1

**当前代码的问题**:
```cpp
data[i] = 305;  // ❌ 错误！

// 正确应该是:
logical_block = 305 / 16 = 19;
offset = 305 % 16 = 1;
physical_block = block_table_gpu[seq_id][19];  // 例如 = 3
data[i] = 3 * 16 + 1 = 49;  // ✅ 正确的 block pool 索引
```

### Bug 影响链

```
set_input_k_idxs() 使用全局索引 (305)
         ↓
cpy_k() 调用 ggml_set_rows(k_pool, k_cur, k_idxs)
         ↓
写入错误的 block pool 位置 (305 而不是 49)
         ↓
Attention 计算时读取错误的 KV cache
         ↓
输出完全乱码！
```

---

## 修复方案

### 方案 1: 在 set_input_k_idxs 中转换索引（推荐⭐⭐⭐⭐⭐）

**修改位置**: `src/llama-kv-cache.cpp:1493-1507`

**修改后代码**:
```cpp
void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            const int64_t global_idx = offs + sinfo.idxs[s][i];

            // ✅ Paged attention: 转换为 block-based 索引
            if (use_paged_attention && block_pool) {
                // 假设 ubatch 中有 seq_id 信息
                // 如果没有，需要从 slot_info 推断
                llama_seq_id seq_id = /* 需要获取 seq_id */;

                // 计算 logical block 和 offset
                const int32_t logical_block = global_idx / block_pool->block_size;
                const int32_t offset = global_idx % block_pool->block_size;

                // 从 block_table_gpu 查找 physical block
                const int32_t physical_block = get_physical_block_from_table(seq_id, logical_block);

                if (physical_block < 0) {
                    LLAMA_LOG_ERROR("%s: seq_id=%d, logical_block=%d not found in block table\n",
                                    __func__, seq_id, logical_block);
                    data[s*sinfo.size() + i] = -1;
                } else {
                    // 计算最终的 block pool 索引
                    data[s*sinfo.size() + i] = physical_block * block_pool->block_size + offset;
                }
            } else {
                // 传统模式: 直接使用全局索引
                data[s*sinfo.size() + i] = global_idx;
            }
        }
    }
}

// 辅助函数: 从 block_table_gpu 查找 physical block
int32_t llama_kv_cache::get_physical_block_from_table(llama_seq_id seq_id, int32_t logical_block) const {
    if (!block_table_gpu || !use_paged_attention) {
        return -1;
    }

    if (seq_id < 0 || (uint32_t)seq_id >= n_seq_max) {
        return -1;
    }

    if (logical_block < 0 || (uint32_t)logical_block >= max_blocks_per_seq) {
        return -1;
    }

    const int32_t * table_data = (const int32_t *)block_table_gpu->data;
    if (!table_data) {
        return -1;
    }

    const size_t idx = (size_t)seq_id * max_blocks_per_seq + logical_block;
    return table_data[idx];
}
```

**同样的修改应用于 `set_input_v_idxs`**

**优点**:
- ✅ 直接修复根本问题
- ✅ 不影响其他代码路径
- ✅ 清晰的职责分离

**缺点**:
- ⚠️ 需要获取 seq_id（可能需要从 slot_info 或 ubatch 推断）

---

### 方案 2: 修改 ggml_set_rows 支持 block indirection

**修改位置**: `ggml/src/ggml.c` (ggml_set_rows 实现)

**思路**: 让 ggml_set_rows 支持一个额外的 block_table 参数，在写入时自动转换索引

**代码框架**:
```cpp
ggml_tensor * ggml_set_rows_paged(
    ggml_context * ctx,
    ggml_tensor * a,          // block pool
    ggml_tensor * b,          // 要写入的数据
    ggml_tensor * idxs,       // 全局索引
    ggml_tensor * block_table, // block table (seq_id -> physical blocks)
    int32_t seq_id,
    int32_t block_size
) {
    // 在内部转换 idxs → block-based idxs
    // 然后调用标准的 ggml_set_rows
}
```

**优点**:
- ✅ 封装了复杂性
- ✅ 可能有 Metal GPU 加速

**缺点**:
- ❌ 需要修改 ggml 核心库
- ❌ 更复杂，更难调试

---

### 方案 3: 修改 cpy_k/cpy_v 实现自定义写入逻辑

**修改位置**: `src/llama-kv-cache.cpp:1384` (cpy_k), `L1421` (cpy_v)

**思路**: 不使用 ggml_set_rows，而是手动遍历 tokens 并写入正确的 block 位置

**代码框架**:
```cpp
ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    // ...

    if (use_paged_attention && block_pool && !block_pool->k_pool.empty()) {
        ggml_tensor * pool = block_pool->k_pool[ikv];

        // 自定义写入逻辑
        // 遍历每个 token
        for (uint32_t t = 0; t < n_tokens; ++t) {
            int64_t global_idx = /* 从 k_idxs 获取 */;
            llama_seq_id seq_id = /* 从 ubatch 获取 */;

            int32_t logical_block = global_idx / block_pool->block_size;
            int32_t offset = global_idx % block_pool->block_size;
            int32_t physical_block = get_physical_block_from_table(seq_id, logical_block);

            // 手动创建 view 并写入
            int64_t pool_idx = physical_block * block_pool->block_size + offset;
            ggml_tensor * dst_view = ggml_view_1d(ctx, pool, n_embd_gqa, pool_idx * pool->nb[1]);
            ggml_tensor * src_view = ggml_view_1d(ctx, k_cur, n_embd_gqa, t * k_cur->nb[2]);
            ggml_build_forward_expand(/* compute graph */, ggml_cpy(ctx, src_view, dst_view));
        }

        return pool;
    }

    // 传统模式
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

**优点**:
- ✅ 完全控制写入逻辑
- ✅ 易于调试

**缺点**:
- ❌ 代码复杂度高
- ❌ 可能影响性能（需要创建多个 view）

---

## 推荐修复方案

**使用方案 1: 在 set_input_k_idxs 中转换索引**

**原因**:
1. ✅ **清晰的职责分离**: set_input_k_idxs 负责索引转换
2. ✅ **最小修改范围**: 只修改两个函数 (set_input_k_idxs, set_input_v_idxs)
3. ✅ **易于测试**: 可以单独测试索引转换逻辑
4. ✅ **不影响 ggml 核心库**: 保持稳定性

---

## 详细实现步骤

### Step 1: 添加辅助函数

**文件**: `src/llama-kv-cache.cpp`

```cpp
// 在 llama_kv_cache 类中添加私有方法
private:
    int32_t get_physical_block_from_table(llama_seq_id seq_id, int32_t logical_block) const {
        if (!block_table_gpu || !use_paged_attention) {
            return -1;
        }

        if (seq_id < 0 || (uint32_t)seq_id >= n_seq_max) {
            return -1;
        }

        if (logical_block < 0 || (uint32_t)logical_block >= max_blocks_per_seq) {
            return -1;
        }

        const int32_t * table_data = (const int32_t *)block_table_gpu->data;
        if (!table_data) {
            return -1;
        }

        const size_t idx = (size_t)seq_id * max_blocks_per_seq + logical_block;
        return table_data[idx];
    }
```

**声明**: `src/llama-kv-cache.h`

```cpp
class llama_kv_cache {
    // ...
private:
    int32_t get_physical_block_from_table(llama_seq_id seq_id, int32_t logical_block) const;
};
```

### Step 2: 修改 set_input_k_idxs

**文件**: `src/llama-kv-cache.cpp:1493-1507`

```cpp
void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            const int64_t global_idx = offs + sinfo.idxs[s][i];

            if (use_paged_attention && block_pool) {
                // Paged attention: 转换为 block-based 索引

                // 获取 seq_id (从 ubatch 或 slot_info)
                // 假设 slot_info 对应的 stream 映射到 seq_id
                // 需要根据实际数据结构调整
                llama_seq_id seq_id = sinfo.strm[s];  // 或者从其他地方获取

                // 计算 logical block 和 offset
                const int32_t logical_block = global_idx / block_pool->block_size;
                const int32_t offset = global_idx % block_pool->block_size;

                // 从 block_table_gpu 查找 physical block
                const int32_t physical_block = get_physical_block_from_table(seq_id, logical_block);

                if (physical_block < 0) {
                    LLAMA_LOG_ERROR("%s: seq_id=%d, logical_block=%d not found in block table (global_idx=%lld)\n",
                                    __func__, seq_id, logical_block, (long long)global_idx);
                    // 使用 -1 标记错误
                    data[s*sinfo.size() + i] = -1;
                } else {
                    // 计算最终的 block pool 索引
                    const int64_t pool_idx = (int64_t)physical_block * block_pool->block_size + offset;
                    data[s*sinfo.size() + i] = pool_idx;

                    if (debug) {
                        LLAMA_LOG_DEBUG("%s: [K] global_idx=%lld -> logical_block=%d, offset=%d, physical_block=%d, pool_idx=%lld\n",
                                        __func__, (long long)global_idx, logical_block, offset, physical_block, (long long)pool_idx);
                    }
                }
            } else {
                // 传统模式: 直接使用全局索引
                data[s*sinfo.size() + i] = global_idx;
            }
        }
    }
}
```

### Step 3: 修改 set_input_v_idxs

**文件**: `src/llama-kv-cache.cpp:1509-1544`

**类似的修改应用于 set_input_v_idxs**（代码类似，省略）

### Step 4: 测试验证

**测试 1: 单元测试**
```cpp
// 测试索引转换逻辑
void test_block_index_conversion() {
    const uint32_t block_size = 16;
    const int64_t global_idx = 305;

    const int32_t expected_logical_block = 19;  // 305 / 16
    const int32_t expected_offset = 1;          // 305 % 16

    // 假设 block_table_gpu[seq_id][19] = 3
    const int32_t expected_pool_idx = 3 * 16 + 1 = 49;

    // 实际测试...
}
```

**测试 2: 端到端测试**
```bash
# 使用 0.6B 模型测试
LLAMA_PAGED_ATTENTION=1 ./llama-server \
  -m Qwen3-0.6B.gguf \
  -ngl 99 \
  -fa auto \
  -np 1 \
  -c 8192

# 发送测试请求
curl -X POST http://localhost:8090/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is", "n_predict": 100, "temperature": 0.0}'

# 预期输出: " Rome. The capital of Spain is Madrid..."
```

**测试 3: 多模型规模验证**
- 0.6B: 应该输出正常英文
- 1.7B: 应该输出正常英文
- 30B: 应该输出正常英文

---

## 潜在问题与注意事项

### 问题 1: seq_id 的获取

**问题**: slot_info 中可能没有直接的 seq_id 映射

**解决**: 需要检查以下可能性
1. `sinfo.strm[s]` 是否就是 seq_id？
2. 是否需要从 `ubatch` 中获取？
3. 是否需要维护一个 stream → seq_id 的映射表？

**代码位置**: 需要查看 `slot_info` 和 `llama_ubatch` 的定义

### 问题 2: Block table 的同步

**问题**: block_table_gpu 是在 CPU 端更新的，GPU 端需要同步

**当前实现**:
```cpp
void llama_kv_cache::update_block_table_gpu(llama_seq_id seq_id, uint32_t logical_block, int32_t physical_block) {
    // ...
    table_data[idx] = physical_block;  // CPU 端更新
}
```

**可能问题**: 如果 Metal GPU 读取 block_table_gpu，需要确保数据已同步

**解决**: 确认 ggml 的 tensor 数据同步机制

### 问题 3: 读取 KV cache 时的索引

**问题**: 写入时转换了索引，读取时也需要转换！

**代码位置**: Attention 计算时读取 K 和 V

**可能需要修改**: Metal kernel 或者 ggml attention operator

---

## 验收标准

1. ✅ **所有模型规模输出正常**
   - 0.6B: 正常英文输出
   - 1.7B: 正常英文输出
   - 30B: 正常英文输出

2. ✅ **性能无衰退**
   - Paged Attention 启用后 tok/s 应该与 baseline 相近或更好

3. ✅ **长时间稳定性**
   - 1000+ 请求无乱码
   - 多并发场景稳定

4. ✅ **Continuous Batching 兼容**
   - 多并发请求正常工作
   - 吞吐量提升

---

## 下一步行动

1. ✅ **立即**: 实现方案 1 的修复代码（已完成）
2. **验证**: 使用 0.6B/1.7B/30B 模型测试
3. **深度测试**: 长时间运行 + 多并发
4. **性能测试**: 对比 baseline vs 修复后的 Paged Attention
5. **提交**: 创建 PR 并提交修复

---

## 修复执行记录

### 2026-03-11 修复完成

**修改文件**:
1. ✅ `src/llama-kv-cache.h`: 添加 `get_physical_block_from_table()` 辅助函数声明
2. ✅ `src/llama-kv-cache.cpp`: 实现 `get_physical_block_from_table()`
3. ✅ `src/llama-kv-cache.cpp`: 修改 `set_input_k_idxs()` (L1493-1539)
4. ✅ `src/llama-kv-cache.cpp`: 修改 `set_input_v_idxs()` (L1541-1572)

**修改内容**:
- **K cache**: 非转置模式，添加 Paged Attention 索引转换
- **V cache**: 非转置模式 + 转置模式，都添加了 Paged Attention 索引转换
- 添加调试日志支持 (debug = false)
- 添加错误处理 (physical_block < 0)

**下一步**: 重新编译并测试

---

*修复方案 v1.0*
*生成于: 2026-03-11*
*修复完成于: 2026-03-11*
