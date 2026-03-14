# ThunderLLAMA 深度优化分析

**日期**: 2026-03-13
**模型**: Qwen3-30B-A3B-128K-Q5_K_M
**硬件**: Apple M4 Pro

## 当前性能基准

| 指标 | 当前值 |
|------|--------|
| 吞吐量 (生成) | 54-57 tok/s |
| 内存占用 | 22.2 GB (Q4 KV cache) |
| 编译优化 | -O3 (Release) |
| GPU 加速 | Metal (99 层) |

## 深度优化方向

### 🚀 Level 1: 推理算法优化（高影响）

#### 1. **Speculative Decoding** ⭐⭐⭐⭐⭐

**原理**: 使用小模型（draft model）快速生成候选 tokens，大模型批量验证

**两种实现方式**:

##### 方式 A: Draft Model（需要额外模型）

```bash
./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  --model-draft ~/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q5_K_M.gguf \
  --draft 16 \
  --draft-min 8 \
  --draft-p-min 0.75 \
  -ngld 99 \
  -c 4096 -ngl 99 --port 30000
```

**预期效果**:
- ✅ 吞吐量提升 **1.5-3x** (理想情况)
- ✅ 适合高重复性文本（代码生成、翻译）
- ❌ 需要额外 ~2 GB 内存（draft model）
- ❌ 需要下载/准备 draft 模型

**是否可行**: ✅ 可行，您有 Qwen3-0.6B 模型可用

##### 方式 B: N-gram Based（无需额外模型）⭐ 推荐

```bash
./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  --spec-type ngram-map-k \
  --spec-ngram-size-n 12 \
  --spec-ngram-size-m 48 \
  --spec-ngram-min-hits 1 \
  -c 4096 -ngl 99 --port 30000
```

**原理**:
- 使用 n-gram 缓存预测下一个 token
- 类似于"自动补全"机制
- 无需额外模型

**预期效果**:
- ✅ 吞吐量提升 **10-30%**（取决于文本重复性）
- ✅ 无需额外内存
- ✅ 适合重复模式多的场景（代码、对话）
- ❌ 对全新内容提升有限

**是否可行**: ✅✅ 强烈推荐，零成本优化

---

### ⚡ Level 2: 内存和缓存优化（中影响）

#### 2. **KV Cache 进阶优化**

##### 2.1 Rolling KV Cache

**原理**: 当 context 满时，滚动删除最旧的 tokens

```bash
--keep 512  # 保留最近 512 个 tokens
```

**预期效果**:
- ✅ 支持无限长对话
- ✅ 内存占用固定
- ❌ 丢失早期上下文

##### 2.2 KV Cache 混合精度

**原理**: 重要层用高精度，不重要层用低精度

**当前限制**: llama.cpp 不支持每层独立配置

**未来可行性**: ⭐⭐ 需要代码修改

#### 3. **Unified Memory 优化**

**M4 Pro 特有优势**: CPU 和 GPU 共享内存

**当前状态**: ✅ 已启用 `has unified memory = true`

**进一步优化**:

```bash
# 增大 Metal 工作集
export GGML_METAL_MAX_BUFFER_SIZE=40000000000  # 40 GB
```

**预期效果**:
- ✅ 减少 CPU-GPU 数据传输
- ✅ 提升 5-10% 性能

---

### 🔧 Level 3: 系统和编译优化（低影响但值得尝试）

#### 4. **编译器优化标志**

**当前**: `-O3` (GCC/Clang 标准优化)

**可尝试**:

```bash
# 重新编译，启用 Link-Time Optimization (LTO)
cd /Users/lisihao/ThunderLLAMA
rm -rf build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_BLAS=ON \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -flto" \
  -DCMAKE_C_FLAGS="-O3 -march=native -flto" \
  -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build --config Release -j
```

**预期效果**:
- ✅ 提升 2-5% 性能
- ⏱️ 编译时间更长

#### 5. **Metal Shaders 优化**

**当前限制**: M4 Pro 不支持 Tensor API

**可用优化**:
- ✅ Simdgroup reduction (已启用)
- ✅ Simdgroup matrix mul (已启用)
- ❌ Tensor API (需要 M5/A19)

**未来**: 等待 M5 芯片

#### 6. **NUMA 和 CPU 亲和性**

```bash
# 绑定到性能核心
--cpu-mask 0xFF  # 绑定到前 8 个核心（假设是性能核）
```

**预期效果**:
- ✅ 减少核心迁移开销
- ✅ 提升 1-3% 性能

---

### 📊 Level 4: 模型级优化（需要重新训练/转换）

#### 7. **模型量化降级**

**当前**: Q5_K_M (5-bit)

**可尝试**: Q4_K_M (4-bit)

```bash
# 需要使用 llama.cpp 的量化工具
# 从原始 F16/F32 模型重新量化
```

**预期效果**:
- ✅ 模型大小从 21.7 GB → ~17 GB
- ✅ 加载速度提升
- ✅ 吞吐量提升 5-10%（内存带宽）
- ❌ 质量轻微下降

**是否推荐**: ⭐⭐⭐ 值得尝试

#### 8. **模型剪枝/蒸馏**

**原理**: 移除不重要的参数/层

**当前限制**: 需要重新训练

**可行性**: ❌ 成本太高，不推荐

---

## 优化优先级排序

| 优化 | 难度 | 成本 | 预期提升 | 优先级 |
|------|------|------|----------|--------|
| **N-gram Speculative** | 低 | 零 | 10-30% | 🏆 P0 |
| **Flash Attention** | 低 | 零 | 10-30% | 🏆 P0 |
| **Paged Attention** | 低 | 零 | 并发提升 | ⭐ P1 |
| **Draft Model Speculative** | 中 | +2GB | 50-200% | ⭐ P1 |
| **Unified Memory 调优** | 低 | 零 | 5-10% | ⭐ P1 |
| **模型降级到 Q4** | 中 | 时间 | 5-10% | ⭐⭐ P2 |
| **LTO 重新编译** | 中 | 时间 | 2-5% | ⭐⭐ P2 |
| **CPU 亲和性** | 低 | 零 | 1-3% | ⭐⭐⭐ P3 |
| **Rolling KV Cache** | 低 | 零 | 长对话 | ⭐⭐⭐ P3 |

---

## 立即可行的优化组合

### 🏆 推荐配置：极致性能

```bash
#!/bin/bash

export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1
export THUNDER_PREFIX_MATCHING=1
export GGML_METAL_MAX_BUFFER_SIZE=40000000000

cd /Users/lisihao/ThunderLLAMA

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  --model-draft ~/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q5_K_M.gguf \
  --draft 16 \
  --draft-min 8 \
  --draft-p-min 0.75 \
  -ngld 99 \
  --spec-type ngram-map-k \
  --spec-ngram-size-n 12 \
  --spec-ngram-size-m 48 \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 4 \
  -b 4096 \
  --cpu-mask 0xFF \
  --port 30000 > /tmp/llama-server-ultra.log 2>&1 &

sleep 10

# 配置 Q4 KV cache
curl -s http://localhost:30000/thunder/kv-strategy \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"fixed","params":{"level":"q4_0"},"version":1}'
```

**预期提升**:
- 基础吞吐量: 54 tok/s
- Flash Attention: +15% → 62 tok/s
- N-gram Speculative: +20% → 74 tok/s
- Draft Model Speculative: +50% → 111 tok/s (理想情况)
- **总计: 2-3x 性能提升（取决于任务类型）**

### ⚡ 保守配置：稳定优化

如果不想冒险（draft model 可能不稳定）：

```bash
export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1
export THUNDER_PREFIX_MATCHING=1

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  --spec-type ngram-map-k \
  --spec-ngram-size-n 12 \
  --spec-ngram-size-m 48 \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 4 \
  -b 4096 \
  --port 30000
```

**预期提升**: 1.4-1.6x (54 tok/s → 75-86 tok/s)

---

## 测试验证方案

### 测试 1: N-gram Speculative

```bash
# 启动带 ngram 的服务器
# 测试代码生成（高重复性）
curl -s http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return",
    "max_tokens": 200
  }' | jq '.timings.predicted_per_second'
```

### 测试 2: Draft Model Speculative

```bash
# 对比有无 draft model 的性能
# 测试长文本生成
```

### 测试 3: 综合性能

```bash
# 运行完整的吞吐量测试
# 对比优化前后
```

---

## 风险评估

| 优化 | 风险 | 缓解措施 |
|------|------|----------|
| Draft Model | 可能不稳定 | 先测试 ngram，再尝试 draft |
| Speculative Decoding | 输出质量可能变化 | 对比测试，temperature=0 验证 |
| Flash Attention | 罕见数值不稳定 | 监控输出质量 |
| LTO 重新编译 | 编译失败 | 保留原 build 备份 |

---

## 最终建议

**立即执行（零风险，高收益）**:
1. ✅ 启用 N-gram Speculative (`--spec-type ngram-map-k`)
2. ✅ 启用 Flash Attention (`-fa on`)
3. ✅ 启用 Paged Attention
4. ✅ 配置 Q4 KV cache

**短期尝试（中风险，高收益）**:
1. ⭐ 测试 Draft Model Speculative（使用 0.6B draft）
2. ⭐ 调优 Unified Memory

**长期考虑（高成本）**:
1. 模型降级到 Q4（重新量化）
2. LTO 重新编译
3. 等待 M5 芯片（Tensor API）

**预期总提升**: **1.5-3x 吞吐量**（从 54 tok/s → 80-160 tok/s）

---

*深度优化分析版本: 1.0*
*分析日期: 2026-03-13*
*分析工具: ThunderLLAMA + llama.cpp*
