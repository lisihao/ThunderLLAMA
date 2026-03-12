# Paged Attention Bug 修复完成报告

> **修复时间**: 2026-03-11
> **修复状态**: ✅ 编译成功，待功能测试
> **修复范围**: Metal Kernel + KV Cache 索引转换

---

## 🎯 修复总结

### 问题回顾

**原始问题**: Paged Attention 实现导致输出垃圾内容（0% 可用性）

**根本原因**: 双重映射冲突
- **Bug 1**: CPU 端写入 KV cache 时使用全局索引而非物理块索引
- **Bug 2**: GPU 端读取 KV cache 时用错误输入（`ic` 缓存迭代索引）进行二次映射

---

## 🔧 修复方案

### 方案选择

**采用方案 A**: GPU 直接使用 CPU 预计算的 k_idxs/v_idxs（专家一致推荐）

**理由**:
- ✅ 避免 GPU 端复杂逻辑和错误的 ic 映射
- ✅ CPU 端已有完整的 logical→physical 映射信息
- ✅ 修改范围清晰，易于验证
- ✅ 不引入额外的 block_table 查表开销

---

## 📝 修改清单

### 1. 接口扩展（ggml.h）

**文件**: `ggml/include/ggml.h`
**修改**: Lines 2343-2353, 2359-2367

```c
// 扩展 flash_attn_ext_set_paged 和 _get_paged 接口
// 新增参数: struct ggml_tensor * k_idxs, struct ggml_tensor * v_idxs
```

**目的**: 传递 CPU 预计算的物理偏移量到 Metal kernel

---

### 2. 数据传递（ggml.c）

**文件**: `ggml/src/ggml.c`
**修改**: Lines 5377-5442

```c
// 将 k_idxs 和 v_idxs 存储到 src[6] 和 src[7]
a->src[5] = block_table;
a->src[6] = k_idxs;  // CPU 预计算的 K 物理偏移
a->src[7] = v_idxs;  // CPU 预计算的 V 物理偏移
```

**目的**: 利用 GGML tensor 的 src slot 机制传递索引数据

---

### 3. 调用链更新（llama-graph.h/cpp）

**文件**:
- `src/llama-graph.h`: Line 863-872
- `src/llama-graph.cpp`: 多处调用点

**修改**:
```cpp
// 修改 build_attn_mha 签名，添加 k_idxs 和 v_idxs 参数
ggml_tensor * build_attn_mha(
    ggml_tensor * q,
    ggml_tensor * k,
    ggml_tensor * v,
    ggml_tensor * kq_b,
    ggml_tensor * kq_mask,
    ggml_tensor * sinks,
    ggml_tensor * v_mla,
          float   kq_scale,
            int   il,
    ggml_tensor * k_idxs = nullptr,  // 新增
    ggml_tensor * v_idxs = nullptr); // 新增

// 所有调用点声明并传递变量
const auto & k_idxs = inp->get_k_idxs();
const auto & v_idxs = inp->get_v_idxs();
ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il, k_idxs, v_idxs);
```

**调用点**: 5 处（Lines 2058, 2141 等）

---

### 4. Metal 后端适配（ggml-metal-ops.cpp）

**文件**: `ggml/src/ggml-metal/ggml-metal-ops.cpp`
**修改**: Lines 2585-2600, 2728-2735, 2800-2803

```cpp
// 1. 获取 k_idxs 和 v_idxs 的 buffer ID
ggml_metal_buffer_id bid_k_idxs = (op->src[6] && op->src[6]->buffer)
    ? ggml_metal_get_buffer_id(op->src[6]) : bid_src0;
ggml_metal_buffer_id bid_v_idxs = (op->src[7] && op->src[7]->buffer)
    ? ggml_metal_get_buffer_id(op->src[7]) : bid_src0;

// 2. 调用 ggml_flash_attn_ext_get_paged 获取参数和索引张量
struct ggml_tensor * k_idxs_tensor = nullptr;
struct ggml_tensor * v_idxs_tensor = nullptr;
ggml_flash_attn_ext_get_paged(..., &k_idxs_tensor, &v_idxs_tensor);

// 3. 设置 buffer (slot 9 和 10)
ggml_metal_encoder_set_buffer(enc, bid_k_idxs, 9);
ggml_metal_encoder_set_buffer(enc, bid_v_idxs, 10);
```

**目的**: 将 CPU 端的索引数据传递到 GPU

---

### 5. Metal Kernel 修复（ggml-metal.metal）

**文件**: `ggml/src/ggml-metal/ggml-metal.metal`
**修改**: 6 处关键位置

#### 5.1 Kernel 签名扩展（Line 5804-5819）

```metal
kernel void kernel_flash_attn_ext(
    constant ggml_metal_kargs_flash_attn_ext & args,
    device const char * q,
    device const char * k,
    device const char * v,
    device const char * mask,
    device const char * sinks,
    device const char * pad,
    device const char * blk,
    device       char * dst,
    device const int64_t * k_idxs,  // 新增: CPU 预计算的 K 物理偏移
    device const int64_t * v_idxs,  // 新增: CPU 预计算的 V 物理偏移
    threadgroup  half * shmem_f16 [[threadgroup(0)]],
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort  tiisg[[thread_index_in_simdgroup]],
    ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
```

#### 5.2 核心修复逻辑（应用于 6 处）

**Before (错误)**:
```metal
// 使用 ic (缓存迭代索引) 作为逻辑位置 → 错误！
const uint64_t k_offset = calc_paged_k_offset(ic, iq3, args, blk);
```

**After (正确)**:
```metal
// 直接使用 CPU 预计算的物理偏移
const int64_t physical_token_offset = k_idxs ? k_idxs[ic] : -1;
if (physical_token_offset < 0) {
    continue;  // 跳过无效 token
}
const uint64_t k_offset = (uint64_t)physical_token_offset * args.token_stride_k;
```

**修改位置**:
1. Line 5390 (K cache 读取 - float16)
2. Line 5500 (K cache 读取 - bfloat16)
3. Line 5610 (V cache 读取 - float16)
4. Line 5720 (V cache 读取 - bfloat16)
5. 另外 2 处（非转置模式）

**关键改进**:
- ❌ 移除错误的 `calc_paged_k_offset(ic, ...)` GPU 端二次映射
- ✅ 直接使用 CPU 端的物理偏移量 `k_idxs[ic]`
- ✅ 添加负值检查（`< 0` 表示无效 token）
- ✅ 简化逻辑，避免双重映射冲突

---

### 6. 编译警告修复（ggml.c）

**文件**: `ggml/src/ggml.c`
**修改**: Line 5290

```c
// Before (警告: address of array 'k->name' will always evaluate to 'true')
bool is_paged = (k->name && strstr(k->name, "block_pool") != NULL);

// After (修复)
bool is_paged = (k->name[0] != '\0' && strstr(k->name, "block_pool") != NULL);
```

---

## 📊 修改统计

| 类别 | 文件数 | 代码行数 | 影响范围 |
|------|--------|----------|----------|
| **接口定义** | 1 (ggml.h) | ~20 | setter/getter 函数签名 |
| **数据传递** | 1 (ggml.c) | ~60 | src[6]/src[7] 数据存储 |
| **调用链** | 2 (llama-graph.*) | ~30 | build_attn_mha 签名+调用 |
| **Metal 后端** | 1 (ggml-metal-ops.cpp) | ~50 | buffer 设置+参数传递 |
| **Metal Kernel** | 1 (ggml-metal.metal) | ~100 | 核心修复逻辑×6 处 |
| **Bug 修复** | 1 (ggml.c) | 1 | 编译警告 |
| **总计** | **7 文件** | **~260 行** | **跨层修改** |

---

## ✅ 验收标准

### 编译验证
- ✅ **警告清零**: 无编译警告
- ✅ **所有目标构建成功**: llama-server, libllama.dylib, libggml*.dylib

### 功能测试（待执行）

#### 测试 1: 简单补全
```bash
./test_paged_attention.sh ~/models/Qwen3-0.6B.gguf
```

**预期**:
- ✅ 输出正常英文（无乱码）
- ✅ 语义连贯（不是垃圾内容）

#### 测试 2: KV Cache 复用
**场景**: 相同 prompt 两次请求（temperature=0.0）

**预期**:
- ✅ 两次输出完全一致
- ✅ 第二次请求复用 KV cache（latency 更低）

#### 测试 3: 多并发请求
**场景**: 3 个并发请求

**预期**:
- ✅ 所有请求正常输出
- ✅ Paged Attention 正确分配和释放块
- ✅ 无内存泄漏，无崩溃

#### 测试 4: 长上下文
**场景**: 8192 tokens context

**预期**:
- ✅ 正确处理长上下文
- ✅ 块映射正确（block_table 查表无误）

---

## 🔍 技术要点

### 为什么这个修复能解决问题？

**问题根源**: Flash Attention 的 `ic` 变量是 cache iteration index（0, C, 2*C, ...），**不是** logical position！

**错误路径**（修复前）:
```
CPU: logical_pos (305) → physical_offset (49) → k_idxs[ic]
GPU: ic (0) → calc_paged_k_offset(0, ...) → 错误的物理地址 → 读到垃圾数据
```

**正确路径**（修复后）:
```
CPU: logical_pos (305) → physical_offset (49) → k_idxs[ic]
GPU: k_idxs[ic] → 49 → 直接使用正确的物理偏移 → 读到正确数据
```

**关键洞察**:
1. CPU 端已经完成了完整的 logical→physical 映射（在 llama-kv-cache.cpp 中）
2. GPU 端不应该再次映射（会用错误的输入 ic）
3. Flash Attention 的循环只是遍历 cache tokens，ic 是迭代索引，不是逻辑位置
4. 直接使用 CPU 预计算结果是最简单、最可靠的方案

---

## 🎓 技术收获

### 对 Flash Attention 的理解

**Flash Attention 循环结构**:
```metal
for (int ic0 = 0; ; ++ic0) {
    int ic = ic0 * C;  // C = cache items per threadgroup
    if (ic >= args.ne11) break;

    // ic 是缓存迭代索引，用于:
    // 1. 索引到 k_idxs[ic] 获取物理偏移
    // 2. 不应该用于 block_table 查表！
}
```

**args.ne11** 含义: cache 中的总 token 数（所有 KV cache tokens）

### 对 Paged Attention 的理解

**两层映射**:
1. **逻辑层**: Logical Position → Logical Block + Offset
2. **物理层**: Logical Block → Physical Block（通过 block_table）

**关键**: 这两层映射应该在 **CPU 端一次性完成**，GPU 端只需要使用结果！

---

## 📚 相关文档

- [PAGED_ATTENTION_BUG_REPORT.md](./.solar/PAGED_ATTENTION_BUG_REPORT.md) - 根因分析
- [PAGED_ATTENTION_BUG_FIX.md](./.solar/PAGED_ATTENTION_BUG_FIX.md) - 原始修复（Bug 1）
- [test_paged_attention.sh](../test_paged_attention.sh) - 功能测试脚本

---

## 🚀 下一步

### 立即执行
```bash
cd /Users/lisihao/ThunderLLAMA

# 1. 找到可用的 GGUF 模型
find ~ -name "*.gguf" -size +100M | head -5

# 2. 运行测试脚本
./test_paged_attention.sh <模型路径>

# 例如:
# ./test_paged_attention.sh ~/models/Qwen3-0.6B.gguf
# ./test_paged_attention.sh ~/models/Qwen3-1.7B-Q4_K_M.gguf
```

### 预期结果
- ✅ **测试通过**: 输出正常，无乱码，KV cache 复用正确
- ❌ **测试失败**: 继续分析，可能还有其他问题

---

## 🏆 成功标准

修复成功的标志:
1. ✅ 编译无警告
2. ✅ 所有测试通过
3. ✅ 性能无衰退（tok/s 与 baseline 相近或更好）
4. ✅ 长时间运行稳定（1000+ 请求无乱码）

---

*修复完成报告 v1.0*
*编译成功时间: 2026-03-11 21:40*
*等待功能测试*
