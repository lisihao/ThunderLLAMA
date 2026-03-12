# Baseline vs Optimized 性能对比报告 (30B 模型)

> **生成时间**: 2026-03-11
> **模型**: Qwen3-30B-A3B-128K-Q5_K_M
> **测试方法**: 端到端 /completion endpoint

---

## 执行摘要

⚠️ **关键发现**: Configuration A 优化版本存在严重的**输出质量问题**（乱码），尽管性能略有提升（~2%）。

**结论**: 需要进一步调查优化参数组合，特别是 **KV Cache 量化 (q8_0)** 可能是导致乱码的主要原因。

---

## 配置对比

### Baseline 配置

```bash
llama-server \
  -m Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -ngl 99 \
  -c 2048 \
  --host 0.0.0.0 \
  --port 8090
```

**特点**:
- ✅ 基础配置：仅 GPU offload
- ✅ 无优化：Flash Attention auto, 无 Paged Attention, 无 KV 量化
- ✅ 单请求：1 slot, 无 Continuous Batching

### Optimized 配置 (Configuration A)

```bash
export LLAMA_PAGED_ATTENTION=1

llama-server \
  -m Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -cram 4096 \
  -kvo \
  -np 8 \
  -cb \
  -b 2048 \
  -ub 512 \
  --prompt-reuse-mode auto \
  -sps 0.5 \
  --mmap \
  --mlock \
  --repack \
  -t 10 \
  -tb 10 \
  -c 8192 \
  --host 0.0.0.0 \
  --port 8090
```

**特点**:
- ✅ Paged Attention: 启用
- ✅ Flash Attention: on (强制启用)
- ✅ KV Cache 量化: q8_0 (内存 ↓50%)
- ✅ Continuous Batching: 启用
- ✅ Slots: 8 (vs baseline 1)
- ✅ Prompt Reuse: auto

---

## 性能对比

### Test 1: 短提示 (128 tokens generation)

| 配置 | Tok/s | Total Time | 变化 |
|------|-------|------------|------|
| **Baseline** | 54.15 | 2.364s | - |
| **Optimized** | 55.27 | 2.316s | **+2.1%** ✅ |

### Test 2: 长提示 (256 tokens generation)

| 配置 | Tok/s | Total Time | 变化 |
|------|-------|------------|------|
| **Baseline** | 52.18 | 4.906s | - |
| **Optimized** | 51.68 | 4.953s | **-1.0%** ⚠️ |

**平均性能变化**: **+0.5%** (几乎无变化)

---

## 输出质量对比

### ⚠️ 严重问题: Optimized 版本输出乱码

#### Baseline 输出 (正常)

**Test 1**:
```
 Rome. The capital of Spain is Madrid. The capital of the United Kingdom is London.
 The capital of the Netherlands is Amsterdam. The capital of Belgium is Brussels.
 The capital of Portugal is Lisbon.
```
✅ **正常英文输出**

**Test 2**:
```
1. What is artificial intelligence?
2. What are the types of artificial intelligence?
3. What are the applications of artificial intelligence?
4. What are the challenges of artificial intelligence?
5.
```
✅ **正常英文输出**

#### Optimized 输出 (乱码)

**Test 1**:
```
dn

幻 ..."

ge做oicriptor.Direction
~~~~联系 idi图 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣...
```
❌ **严重乱码，包含中文字符和无意义符号**

**Test 2**:
```
2 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣 通用)衣...
```
❌ **严重乱码，重复中文字符**

---

## 根本原因分析

### 怀疑对象 1: KV Cache 量化 (q8_0) ⭐⭐⭐⭐⭐

**理由**:
- Qwen3-30B 模型可能对 KV cache 精度敏感
- q8_0 量化虽然文档称 "质量损失 < 1%"，但可能在某些模型上表现不佳
- Baseline (f16) 输出正常，Optimized (q8_0) 输出乱码

**验证方法**: 移除 `-ctk q8_0 -ctv q8_0`，重新测试

### 怀疑对象 2: Flash Attention (强制 on) ⭐⭐⭐

**理由**:
- Baseline 使用 `auto` 模式
- Optimized 强制 `on`
- 可能存在实现 bug 或兼容性问题

**验证方法**: 改回 `-fa auto`，重新测试

### 怀疑对象 3: Paged Attention ⭐⭐

**理由**:
- ThunderLLAMA 的核心创新
- 可能存在与 KV cache 量化的组合问题

**验证方法**: 关闭 `LLAMA_PAGED_ATTENTION=1`，重新测试

### 怀疑对象 4: Context 限制 (slot n_ctx = 1024) ⭐

**理由**:
- 日志显示 `n_ctx_seq (1024) < n_ctx_train (131072)`
- Slot 的 context 只有 1024，而非设置的 8192
- 可能导致模型行为异常

**验证方法**: 调整 slot context 参数

---

## 下一步行动

### 1. 排查 KV Cache 量化问题 (优先级 ⭐⭐⭐⭐⭐)

**测试**: 移除 KV cache 量化，使用 f16
```bash
LLAMA_PAGED_ATTENTION=1 llama-server \
  -m model.gguf \
  -ngl 99 \
  -fa on \
  # 移除 -ctk q8_0 -ctv q8_0
  -cram 4096 \
  -kvo \
  -np 8 \
  -cb \
  ...
```

### 2. 测试 Flash Attention auto 模式 (优先级 ⭐⭐⭐)

**测试**: 改回 `-fa auto`
```bash
LLAMA_PAGED_ATTENTION=1 llama-server \
  -m model.gguf \
  -ngl 99 \
  -fa auto \  # 改为 auto
  -cram 4096 \
  ...
```

### 3. 逐步添加优化选项 (优先级 ⭐⭐)

**策略**: 从 Baseline 开始，逐步添加优化选项，每次测试输出质量

1. Baseline + Flash Attention on
2. Baseline + Flash Attention on + Paged Attention
3. Baseline + Flash Attention on + Paged Attention + Continuous Batching
4. Baseline + Flash Attention on + Paged Attention + Continuous Batching + KV Quantization (q8_0)

### 4. 调查 Slot Context 限制 (优先级 ⭐)

**检查**: 为什么 slot n_ctx = 1024 而非 8192

---

## 总结

| 维度 | Baseline | Optimized | 结论 |
|------|----------|-----------|------|
| **性能** | 52-54 tok/s | 51-55 tok/s | 几乎无变化 (+0.5%) |
| **输出质量** | ✅ 正常 | ❌ **严重乱码** | **不可接受** |
| **内存占用** | 较高 (f16) | 较低 (q8_0 理论 ↓50%) | 未测量 |
| **并发能力** | 1 slot | 8 slots | 未测试 |

**关键结论**:
1. ❌ **Configuration A 存在严重质量问题，不可用于生产环境**
2. ⚠️ **KV Cache 量化 (q8_0) 是最可能的罪魁祸首**
3. ✅ **性能提升微弱 (+0.5%)，不足以抵消质量损失**
4. 🔍 **需要逐步排查优化选项，找到质量和性能的平衡点**

**建议**:
- 立即停止使用 Configuration A
- 优先测试移除 KV cache 量化的版本
- 考虑更保守的优化配置（如仅启用 Flash Attention + Paged Attention）

---

*报告生成于: 2026-03-11*
*测试环境: Apple M4 Pro (48GB), ThunderLLAMA*
*模型: Qwen3-30B-A3B-128K-Q5_K_M (Q5_K_M 量化)*
