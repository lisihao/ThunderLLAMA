# ThunderLLAMA 30B 模型现实优化方案

**日期**: 2026-03-13
**模型**: Qwen3-30B-A3B-128K-Q5_K_M (30B, Q5 量化)
**硬件**: Apple M4 Pro
**当前性能**: 54-57 tok/s

## ❌ 已验证无效的优化（不要再尝试）

| 优化 | 结果 | 原因 |
|------|------|------|
| **Speculative Decoding (Draft Model)** | ❌ 质量↓ 性能无提升 | 30B 太大，验证成本高 |
| **N-gram Speculative** | ❌ 质量↓ 性能无提升 | N-gram 预测对大模型帮助有限 |

**教训**: 大模型 (30B+) 的瓶颈是**计算**而非**内存带宽**，推测解码的验证成本抵消收益。

---

## ✅ 已验证有效的优化

| 优化 | 效果 | 状态 |
|------|------|------|
| **KV Cache Q4 量化** | -72% 内存, -4% 性能 | ✅ 已启用 |
| **GPU Offload (-ngl 99)** | 全层 GPU 加速 | ✅ 已启用 |
| **LMCache** | 重复内容跳过 | ✅ 已启用 |

---

## 🎯 可尝试的优化方向

### Level 1: 缓存和内存复用 ⭐⭐⭐⭐

#### 1. **Cache Reuse (KV Shifting)**

```bash
--cache-reuse 256  # 重用缓存，避免重新计算
```

**原理**:
- 当新 prompt 与之前有相同前缀时，"移位"复用 KV cache
- 类似于滑动窗口

**适用场景**:
- ✅ RAG (system prompt 相同)
- ✅ 多轮对话 (历史 context 相同)
- ✅ 批量任务 (template 相同)

**预期效果**:
- 相同前缀场景：首 token 延迟降低 50-80%
- 吞吐量提升：10-30%（取决于前缀重复率）

**风险**: 低

---

#### 2. **增大 Cache RAM**

```bash
--cache-ram 16384  # 从默认 8GB 增大到 16GB
```

**原理**: 增大 prompt cache 容量，缓存更多历史

**预期效果**:
- 多用户场景：缓存命中率提升
- 长对话：减少重新计算

**风险**: 低（只是内存占用增加）

---

#### 3. **Unified KV Buffer**

```bash
--kv-unified  # 所有 slots 共享一个 KV buffer
```

**原理**: 减少内存碎片，提升缓存利用率

**预期效果**:
- 内存利用率提升 5-10%
- 并发能力提升

**风险**: 低

---

### Level 2: 系统级调优 ⭐⭐⭐

#### 4. **Continuous Batching 优化**

```bash
--cont-batching      # 确保启用
--prio-batch 2       # 提高批处理优先级
-b 4096              # 增大逻辑 batch size
-ub 1024             # 增大物理 batch size
```

**原理**:
- 动态批处理，充分利用 GPU
- 提高批处理优先级，减少调度延迟

**预期效果**:
- 多用户场景：吞吐量提升 10-20%
- 单用户：提升有限

**风险**: 低

---

#### 5. **Metal Buffer 优化**

```bash
--no-host  # 绕过 host buffer，直接使用 GPU 缓存
```

**原理**:
- M4 Pro 的 unified memory 优势
- 减少 CPU-GPU 数据传输

**预期效果**: 5-10% 性能提升

**风险**: 中（可能不稳定）

---

#### 6. **CPU 亲和性和优先级**

```bash
--cpu-mask 0xFF      # 绑定到性能核心
--prio 2             # 提高进程优先级
--threads 8          # 明确指定线程数
--threads-batch 8    # 批处理线程数
```

**预期效果**: 1-3% 性能提升

**风险**: 低

---

### Level 3: 激进的内存优化 ⭐⭐

#### 7. **模型权重降级到 Q4**

**当前**: Q5_K_M (5-bit, 21.7 GB)
**目标**: Q4_K_M (4-bit, ~17 GB)

**如何做**:
1. 下载或转换 Q4 版本模型
2. 重新测试质量
3. 对比性能

**预期效果**:
- 模型加载速度 +20%
- 内存节省 ~4.7 GB
- 吞吐量提升 5-10%（内存带宽）
- ⚠️ 质量可能轻微下降

**风险**: 中（需要重新下载/转换模型）

---

#### 8. **更小的 Context Size**

```bash
-c 2048  # 从 4096 降到 2048
```

**预期效果**:
- KV cache 内存减半
- 吞吐量提升 5-10%
- ⚠️ 不支持长对话

**风险**: 高（功能受限）

---

### Level 4: 编译优化 ⭐

#### 9. **Link-Time Optimization (LTO)**

```bash
cd /Users/lisihao/ThunderLLAMA
rm -rf build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -flto=thin" \
  -DCMAKE_C_FLAGS="-O3 -march=native -flto=thin"
cmake --build build --config Release -j
```

**预期效果**: 2-5% 性能提升

**风险**: 中（编译时间长，可能失败）

---

## 📊 优化优先级（基于 30B 模型特点）

| 优化 | 难度 | 成本 | 预期提升 | 风险 | 优先级 |
|------|------|------|----------|------|--------|
| **Cache Reuse** | 低 | 零 | 10-30% (特定场景) | 低 | 🏆 P0 |
| **Unified KV Buffer** | 低 | 零 | 5-10% | 低 | ⭐ P1 |
| **Continuous Batching 调优** | 低 | 零 | 10-20% (并发) | 低 | ⭐ P1 |
| **Metal Buffer 优化** | 低 | 零 | 5-10% | 中 | ⭐ P1 |
| **增大 Cache RAM** | 低 | 零 | 5-15% (多用户) | 低 | ⭐⭐ P2 |
| **CPU 亲和性** | 低 | 零 | 1-3% | 低 | ⭐⭐ P2 |
| **模型降级 Q4** | 中 | 时间 | 5-10% | 中 | ⭐⭐ P2 |
| **LTO 重新编译** | 中 | 时间 | 2-5% | 中 | ⭐⭐⭐ P3 |
| **减小 Context** | 低 | 功能 | 5-10% | 高 | ⚠️ 不推荐 |

---

## 🚀 推荐立即尝试的配置

### 配置 1: 缓存优化（零风险）

```bash
export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1
export THUNDER_PREFIX_MATCHING=1

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 4 \
  -b 4096 \
  -ub 1024 \
  --cache-reuse 256 \
  --cache-ram 16384 \
  --kv-unified \
  --cont-batching \
  --prio-batch 2 \
  --port 30000
```

**预期**: 10-20% 性能提升（RAG/多轮对话场景）

---

### 配置 2: 激进优化（需要测试）

```bash
export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1
export THUNDER_PREFIX_MATCHING=1

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 4 \
  -b 4096 \
  -ub 1024 \
  --cache-reuse 256 \
  --cache-ram 16384 \
  --kv-unified \
  --no-host \
  --cpu-mask 0xFF \
  --prio 2 \
  --threads 8 \
  --threads-batch 8 \
  --port 30000
```

**预期**: 15-30% 性能提升（理想情况）

---

## 🎯 现实预期

**基于大模型特点**，30B Q5 模型的优化空间有限：

| 场景 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| **单用户，新内容** | 54 tok/s | 57-60 tok/s | +5-10% |
| **RAG (重复前缀)** | 54 tok/s | 65-75 tok/s | +20-40% |
| **多用户并发** | 54 tok/s | 60-70 tok/s | +10-30% |

**核心教训**:
- 30B 大模型瓶颈是**计算**，不是内存或带宽
- 优化重点应放在**缓存复用**和**并发**，而非推理加速
- 单用户新内容场景的提升空间极其有限（< 10%）

**如果需要更高性能**：
1. 使用更小的模型 (14B/7B)
2. 使用更强的硬件 (M5 with Tensor API)
3. 接受当前性能，优化应用层（缓存、预计算）

---

*现实优化方案版本: 1.0*
*基于失败经验修订: 2026-03-13*
*记录于 Cortex: thunderllama-30b-spec-failed-2026*
