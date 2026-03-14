# ThunderLLAMA：在 Mac 上把 305 亿参数模型推到 79 tok/s

> 基于 llama.cpp，我们在 Apple M4 Pro 上做了多层 KV 缓存、Hybrid Hashing、Skip Logic、手写 Metal MoE 融合内核等一系列深度优化。从开箱即用的 ~45 tok/s 到最终的 79.12 tok/s，这是完整的技术故事。

## 为什么做 ThunderLLAMA

Qwen3-30B-A3B 是当前 Mac 用户能本地跑的最高"性价比" MoE 大模型——305 亿参数，每次推理只激活 30 亿（128 experts 选 8 个），Q5_K_M 量化后 20 GB 刚好塞进 M4 Pro 的 36 GB 统一内存。

但 llama.cpp 的 Metal 后端对 MoE 架构几乎没有针对性优化。开箱即用跑 Qwen3-30B-A3B，TG（token generation）大约 45 tok/s——对交互式使用来说已经可用，但远没到硬件天花板。

**ThunderLLAMA 的目标**：在不修改模型、不损失精度的前提下，把 Apple Silicon 的性能榨干。

---

## 第一章：LMCache——给 KV Cache 加上多层缓存

### 问题

大语言模型推理的一个核心环节是 KV Cache——存储每一层 attention 的 Key 和 Value 向量。对于一个 1500 token 的 prompt，Qwen3-30B-A3B 的 48 层 attention 需要计算和存储大量的 KV 数据。

在实际应用中（Agent 系统、RAG、多轮对话），很多请求共享大量相同的 prefix（system prompt、文档片段、历史上下文）。每次都重新计算这些 KV 是纯粹的浪费。

llama.cpp 有内置的 session cache 和 prompt reuse，但它们基于 slot 绑定，跨请求复用能力有限。

### 方案：两层缓存架构

我们在 llama.cpp 内部实现了一套完整的 KV chunk 缓存系统：

```
L2 (CPU Heap - 8GB)
  未压缩 | LRU 淘汰 | 访问频率跟踪 | 微秒级访问
      ↕ 智能预热 (4 线程并行 I/O)
L3 (Disk mmap - 256GB)
  zlib 压缩 | XXH64 校验 | 持久化存储 | 毫秒级访问
```

**核心设计决策**：

- **Chunk 粒度**：256 tokens 为一个 chunk，每层独立缓存。这个粒度在命中率和存储开销之间取得平衡。
- **L2 不压缩**：内存中保持原始数据，避免解压开销。8 GB 上限通过 LRU 淘汰控制。
- **L3 zlib 压缩**：KV 数据的压缩比通常 2-4x，256 GB 磁盘空间可以存储 512-1024 GB 原始数据。
- **XXH64 校验和**：每个 chunk 写入时计算 checksum，读取时验证。XXH64 速度超过 10 GB/s，对性能影响 < 1%。
- **安全卸载**：支持 USB 外置存储。`kill -USR1` 信号触发 L3 → L2 优雅降级，不丢数据不崩溃。
- **智能预热**：基于访问频率统计，启动时 4 线程并行预加载热点 chunk，预热速度 4x。

### 磁盘格式

```
[Header 72 字节]:
  [8B: content_hash] [4B: layer_idx] [4B: chunk_start]
  [8B: k_size] [8B: v_size] [8B: last_access_ns]
  [4B: access_count] [4B: padding]
  [8B: checksum (XXH64)]
  [8B: compressed_k_size] [8B: compressed_v_size]
[compressed_k_data]
[compressed_v_data]
```

### 初步效果

相同 prompt 重复请求：

| 轮次 | 延迟 | 说明 |
|:----:|:----:|:----:|
| Round 1 (cold) | 6967ms | 首次计算，写入缓存 |
| Round 2 (warm) | 1443ms | 缓存命中 |
| Round 3 (warm) | 1314ms | 缓存热 |
| **加速比** | **5.06x** | |

但这只是"完全相同 prompt"的场景。真实世界的 prefix overlap（部分相同）场景，命中率远没有这么好。

---

## 第二章：Hybrid Hashing——让部分匹配真正生效

### 问题

LMCache 最初使用 **position-based hashing**——用 (layer_idx, chunk_start) 作为 key。这意味着：

- 100% 相同的 prompt → 100% 命中 → 11.85x 加速 ✅
- 50% prefix overlap（前半部分相同，后半部分不同）→ 几乎不命中 → **1.11x** ❌

为什么？因为 position-based hashing 把每个 chunk 的位置作为唯一标识。两个不同 prompt 即使前 750 个 token 完全一样，它们的 chunk key 也不同（因为 prompt 总长度不同会影响 tokenization 边界）。

### 方案：Content-based + Position-based 混合

关键洞察：**prefill 阶段有完整的 token 序列，可以用内容哈希；decode 阶段只有 1 个 token，必须用位置哈希。**

```cpp
if (chunk_start + CHUNK_SIZE <= ubatch.n_tokens) {
    // Prefill: 用 token 内容做 hash
    key = hasher->make_key(ubatch.token, ubatch.n_tokens, chunk_start, il);
} else {
    // Decode: 只有 1 个 token，用位置 hash
    key.content_hash = 0;
    key.layer_idx = il;
    key.chunk_start = chunk_start;
}
```

内容哈希使用 xxHash64，把 chunk 范围内的 256 个 token ID 和 layer index 一起做哈希。两个不同 prompt 只要某个 256-token chunk 的内容完全一致，就会命中缓存。

### 效果

50% prefix overlap 场景（严格测试：缓存完全清空后重建）：

| 指标 | Position-based | Hybrid |
|:----:|:----:|:----:|
| 50% overlap 加速比 | 1.11x | **3.01x** |
| 100% 相同加速比 | 11.85x | **25.59x** |

**这意味着在 RAG 系统（文档片段复用）、多 Agent（共享 system prompt）、多轮对话（历史上下文固定）等真实场景中，LMCache 才真正发挥了作用。**

---

## 第三章：Skip Logic——100% 命中时跳过整个 Forward Pass

### 问题

LMCache 命中后，KV 数据从缓存恢复到 GPU，但 **forward pass 仍然在跑**。对于一个 1500 token 的 prompt，即使所有 KV chunk 都命中了，GPU 仍然在做矩阵乘法和 attention 计算——只是结果被覆盖了。

### 方案：Full Skip

当 prefill 阶段检测到 100% chunk 命中率时，直接跳过整个 forward pass：

```cpp
if (is_prefill && chunks_found == chunks_needed) {
    // 所有 KV 数据已从缓存恢复，跳过计算
    lmcache_can_skip_compute = true;
    // 日志: 🚀 LMCache FULL HIT: 96/96 chunks cached, SKIPPING forward pass!
}
```

### 触发条件

关键发现：llama.cpp 的内置 session cache 会导致第二次请求走 decode 路径（1 token），而非 prefill 路径。要触发 Skip Logic，需要 `cache_prompt=false`，强制走完整 prefill。

```
Round 1 (cold cache): 8883.3ms — 正常计算，写入 LMCache
Round 2 (warm, skip): 336.4ms  — Full Skip 触发
Round 3 (warm, skip): 299.7ms  — Full Skip 触发
加速比: 27.93x ✅
```

### Approximate Skip

100% 命中是最理想的情况，但真实场景中可能有 95-99% 的命中率。我们实现了 Approximate Skip：当命中率 ≥ 95% 时，用零填充缺失的 chunk，仍然跳过 forward pass。

```cpp
if (hit_ratio >= 0.95) {
    // Zero-fill missing chunks
    for (const auto& missing : missing_chunks) {
        memset(zeros, 0, chunk_bytes);
        ggml_backend_tensor_set(missing.k_tensor, zeros, offset, chunk_bytes);
    }
    lmcache_can_skip_compute = true;
}
```

这把 Skip 的触发覆盖率从约 5% 提升到约 30%。

### V Tensor 缓存实验——一个有价值的负结果

我们也尝试了同时缓存 V tensor，理论上可以完全跳过 forward pass 的计算。结果：**-8.2% 性能下降**。

原因：在 decode 阶段（生成每个 token），新 token 的 Q/K/V **必须重新计算**，V tensor 缓存在 decode 阶段是纯开销。只有 prefill 阶段 + 100% 命中时 V 缓存才有用，但这时 K-only + Skip Logic 已经够用了。

**教训：不是所有理论上正确的优化都能在实践中生效。Profile > Theory。**

---

## 第四章：三层协同优化架构

LMCache 的 Skip Logic 需要 100% 命中才能触发，但大多数请求都是"部分相同"的。我们设计了三层协同优化架构来最大化命中率：

```
L1: ContextPilot (Prompt Optimizer)
    → 语义去重 + 重排序 → 减少 36% 冗余 tokens

L2: ClawGate (Service Orchestrator)
    → 检测 overlap > 80% → 设置 cache_prompt=false
    → 强制 full prefill → 触发 Skip Logic

L3: ThunderLLAMA + LMCache
    → Hybrid Hashing (3x prefix overlap)
    → Skip Logic (27x full hit)
```

**核心策略：Cache-Aware Routing**

ClawGate 通过 `/lmcache/stats` 端点实时获取 LMCache 的命中率和缓存状态。当检测到新请求与已有缓存高度重叠时，自动将 `cache_prompt` 设为 false，强制 full prefill，让 LMCache 的 Skip Logic 生效。

实测 Phase 1 完成后：**23.34x 加速**（理论上限 27.93x 的 83%）。

---

## 第五章：Metal GPU 调优——从软件层面逼近硬件天花板

LMCache 解决的是"重复计算"的问题。对于每个 token 的"第一次计算"——也就是纯推理速度——我们需要从 Metal GPU 层面优化。

### 5.1 Flash Attention

llama.cpp 内置了 Flash Attention，但默认是 auto 模式。在 M4 Pro 上强制开启 FA 后：

| 配置 | TG (tok/s) |
|:----:|:----:|
| FA=off | ~45 |
| FA=on | ~54 |
| 提升 | +20% |

FA 将 attention 的内存复杂度从 O(n²) 降到 O(√n)，在 GPU 内存带宽瓶颈下效果显著。

### 5.2 KV Cache 量化选择

直觉告诉我们 KV cache 量化（q4_0/q8_0）应该减少内存带宽占用从而加速推理。实测结果出乎意料：

| KV 类型 | PP512 | TG128 |
|:----:|:----:|:----:|
| q4_0 | 573 | 54.2 |
| q8_0 | 670 | 58.3 |
| **f16** | **686** | **59.0** |

**f16 最快。** 原因：在短上下文（4096 tokens）下，KV cache 总量有限，量化的反量化（dequant）计算开销超过了内存带宽节省。

**教训：Profile 你的实际配置，不要用直觉决定优化方向。**

### 5.3 CPU 线程数调优

M4 Pro 的统一内存架构意味着 CPU 和 GPU 共享内存带宽。我们做了 6 组实测（threads = 2, 4, 6, 8, 10, 12）：

| 配置 | TG128 | PP512 |
|:----:|:----:|:----:|
| t=4 | **66.36** | 699.52 |
| t=8 | 65.61 | **718.37** |

TG 阶段用 4 线程最优（减少 CPU 对 GPU 带宽的争抢），PP 阶段用 8 线程更快（GPU 空闲多，CPU 可以多占带宽）。

最终配置：**TG 用 4 线程，PP 用 8 线程，分离配置。** +1.1% TG 提升。

### 5.4 Q5_K Dequant 内核微调

llama.cpp 的 Metal Q5_K 反量化内核有一个编译时参数 `N_R0_Q5_K`——每个 simdgroup 处理的行数。默认值 1。

我们做了 7 组参数搜索（N_R0={1,2,4,8,16} × N_SG={2,4}），发现 N_R0=8 是甜区。N_R0=16 由于寄存器溢出反而变慢。

**+2-3% TG 提升。**

### 累计效果（Q5_K_M）

| 优化项 | TG (tok/s) | 提升 |
|:----:|:----:|:----:|
| 开箱即用 | ~45 | baseline |
| + Flash Attention | ~54 | +20% |
| + KV Cache f16 | ~54.5 | +1% |
| + CPU 线程分离 | ~55.6 | +2% |
| + N_R0_Q5_K=8 | ~57 | +2.5% |
| + ADD fusion (内置) | ~59 | +3.5% |
| 小计（GPU 调优） | **~59** | **+31%** |

带宽利用率此时已达到 ~63-68%，理论天花板约 75-80%。常规参数调优的空间已经不多了。

---

## 第六章：手写 Metal MoE 融合内核——最后一块拼图

### 问题

Metal GPU profiling 显示，Qwen3-30B-A3B 每生成一个 token 需要 **约 1392 个 GPU dispatch**：

| 阶段 | Dispatch 数 | 占比 |
|:----:|:----:|:----:|
| MoE gating | 7/层 × 48 = 336 | 24% |
| MoE FFN | 5/层 × 48 = 240 | 17% |
| MoE aggregation | 7/层 × 48 = 336 | 24% |
| Attention | 10/层 × 48 = 480 | 35% |
| **合计** | **~1392** | |

每个 dispatch 有 2-5 微秒的固定调度开销。1392 次累计 3-7 毫秒。TG 每 token 约 17ms（Q5_K），dispatch 开销占 **15-25%**。

CUDA 后端早就有了 `topk_moe` 融合内核。**Metal 后端完全没有 MoE 特化优化。**

### 内核设计

MoE 的 gating 过程：
```
logits [128 experts, N tokens]
  → SOFT_MAX → ARGSORT → GET_ROWS → SUM_ROWS → CLAMP → DIV
```

我们把前 3 个操作融合为一个内核：

```metal
kernel void kernel_topk_moe_f32(
    constant ggml_metal_kargs_topk_moe & args,
    device const char * src0,   // logits [n_expert, n_tokens]
    device       char * dst0,   // weights [n_used, n_tokens]
    device       char * dst1,   // ids [n_used, n_tokens]
    ...
)
```

**算法**（1 个 simdgroup = 32 线程处理 1 个 token）：

1. **并行加载**：128 experts ÷ 32 threads = 4 experts/thread
2. **Simdgroup Softmax**：`simd_max` + `simd_sum` 做 warp 级 reduce，零共享内存
3. **迭代式 Top-K**：8 轮 argmax，每轮 `simd_shuffle_xor` 蝶形规约
4. **可选归一化**：sum → clamp → divide

**为什么 128 experts 完美适合 simdgroup？** 128 = 4 × 32（WARP_SIZE）。每个 simdgroup 的 32 线程各持有 4 个 expert 的 softmax 概率，所有 reduce 操作在寄存器级别完成。

### Graph Fusion 模式匹配

在 Metal 后端的 op dispatch 循环中，检测 `SOFT_MAX → ARGSORT → GET_ROWS` 三连模式：

```cpp
if (node->op == GGML_OP_SOFT_MAX && ctx->use_fusion) {
    // 验证: scale=1.0, 无 mask, logits 连续, n_expert ≤ 256 且为 2 的幂次
    if (match_moe_pattern(ctx, idx)) {
        n_fuse = ggml_metal_op_topk_moe(ctx, idx);
    }
}
```

匹配成功 → 1 个融合 dispatch 替代 3 个独立 dispatch。48 层 × 减少 2 dispatches = **96 dispatches eliminated**。

### 踩过的坑

**坑 1：Pipeline 未编译**。新内核需要先编译才能使用，但我们调用了 `get_pipeline()`（纯查找）而非 `compile_pipeline()`。返回 nil → segfault。

**坑 2：Metal buffer index 顺序**。llama.cpp 约定 index 0 = kernel args struct，但我们把 device buffer 放在了 index 0。GPU 读到错误数据，输出全是垃圾。

**坑 3：Graph Scheduler 重排**。我们原本想融合完整的 6-op 链。但 ggml 的 graph scheduler 把 `MUL_MAT_ID`（expert matmul）插到了 `GET_ROWS` 和 `SUM_ROWS` 之间——因为 matmul 依赖 expert IDs 但不依赖归一化权重。最终只能融合前 3 个 op，但这已经是收益最大的部分。

### 性能结果

测试平台：Apple M4 Pro (36GB)，llama-bench，5-10 run 取平均

**Q5_K_M (20.23 GiB)**

| 配置 | PP512 (tok/s) | TG128 (tok/s) | TG 提升 |
|:----:|:----:|:----:|:----:|
| 无融合 | 723.84 | 59.07 | — |
| **MoE 融合** | **729.47** | **65.25 ± 0.16** | **+10.5%** |

**Q4_K_M (17.28 GiB)**

| 配置 | PP512 (tok/s) | TG128 (tok/s) | TG 提升 |
|:----:|:----:|:----:|:----:|
| 无融合 | 767.57 | 70.35 | — |
| **MoE 融合** | **787.50** | **79.12 ± 0.20** | **+12.5%** |

**正确性验证**：相同 prompt + seed + temperature=0，融合前后输出 **byte-identical**。

### 为什么低量化提升更大？

MoE 模型 TG 是内存带宽瓶颈。Q4_K_M 权重更小 → 读取更快 → 每 token 计算时间更短 → dispatch 固定开销占比更高。

- Q4_K_M：每 token ~12.7ms，dispatch 开销（~3ms）占 23.6%
- Q5_K_M：每 token ~16.9ms，dispatch 开销占 17.8%

**融合减少的是固定开销，所以在"快"的配置下效果更显著。**

---

## 第七章：配置系统——防止优化参数丢失

34 个可调优化项、十几个环境变量、各种编译时参数——如果每次启动都要手动输入，迟早会遗漏。

我们修改了 llama-server，让它 **只从 `thunderllama.conf` 读取配置**，忽略所有命令行参数和外部环境变量。

```bash
# ❌ 旧方式（参数太多，容易遗忘）
THUNDER_LMCACHE=1 THUNDER_PREFIX_MATCHING=1 LLAMA_PAGED_ATTENTION=1 \
./build/bin/llama-server \
  -m model.gguf -c 4096 -ngl 99 --port 30000 -fa on ...

# ✅ 新方式（一个命令，永不遗忘）
./start-thunderllama.sh
```

配置文件是唯一真相源。修改优化参数只需要编辑 `thunderllama.conf`，然后 `./restart-thunderllama.sh`。

---

## 最终成绩单

### 推理速度（TG，单 token 生成）

| 阶段 | Q5_K_M | Q4_K_M | 关键技术 |
|:----:|:----:|:----:|:----:|
| llama.cpp 开箱即用 | ~45 | ~53 | — |
| + Flash Attention | ~54 | ~63 | O(√n) attention |
| + KV f16 + Thread tuning | ~57 | ~67 | 减少 GPU 带宽争抢 |
| + N_R0=8 + ADD fusion | ~59 | ~70 | Metal 内核微调 |
| + **MoE Kernel Fusion** | **65.25** | **79.12** | 手写 Metal 融合内核 |
| **总提升** | **+45%** | **+49%** | |

### 缓存加速（服务器场景）

| 场景 | 加速比 | 关键技术 |
|:----:|:----:|:----:|
| 100% 相同 prompt | **27.93x** | Full Skip Logic |
| 50% prefix overlap | **3.01x** | Hybrid Hashing |
| 三层协同 (ContextPilot + ClawGate + LMCache) | **23.34x** | Cache-Aware Routing |
| 不同 prompt (cold) | 1.0x | — |

### 代码改动量

| 模块 | 行数 | 说明 |
|:----:|:----:|:----:|
| LMCache 核心 | ~2500 | 多层缓存 + Hybrid Hashing + Skip Logic |
| Metal MoE 内核 | ~340 | 160 行 shader + 180 行 C++ |
| 配置系统 | ~200 | config-parser + thunderllama.conf |
| 三层协同 (ClawGate) | ~500 | Cache-Aware Routing |
| CLI 工具 | ~300 | thunder-cache 管理工具 |
| **总计** | **~3840** | |

---

## 几个有价值的教训

**1. Profile > Theory**

KV cache 量化"应该"更快（减少内存读写），但在短上下文下 f16 反而最快（免反量化）。V tensor 缓存"应该"加速（少算一次），但实际是纯开销（-8.2%）。每一个优化都必须实测验证。

**2. 固定开销被低估**

MoE 模型每 token 1392 个 GPU dispatch，每个 2-5μs。听起来很小，但累计 3-7ms 占了 TG 时间的 15-25%。在量化越低（Q4_K_M）、模型越快的配置下，固定开销占比越高。

**3. 图调度器是融合的天花板**

ggml 的 graph scheduler 会重排无依赖的操作。这让某些"看起来连续"的 op 链在运行时被打断。我们想融合 6 个 op，最终只能融合 3 个。要做更深的融合，需要修改调度器本身或引入新的 op 类型。

**4. 缓存系统的 All-or-Nothing 特性**

Content-based hashing 意味着修改 1 个 token 就会改变整个 256-token chunk 的 hash。这让缓存命中要么 100% 要么 0%，中间状态极少。Approximate Skip（95% 阈值 + 零填充）是对这个特性的妥协。

**5. 配置必须持久化**

34 个优化参数如果只存在启动命令里，上下文切换后必然遗忘。配置文件作为唯一真相源，是工程上最简单但最有效的保障。

---

## 下一步

1. **Normalization Chain Fusion** — 修改 graph scheduler 保证 SUM_ROWS/CLAMP/DIV 与 GET_ROWS 相邻，预估 TG 再提 3-5%
2. **Multi-Expert MatMul Fusion** — 8 个 expert 的 MUL_MAT_ID 合并为 1 个 grouped matmul
3. **Qwen3.5-35B-A3B** — 在更新架构的模型上验证泛化性
4. **Grafana 监控** — LMCache 命中率、Skip 触发率、缓存大小的实时可视化

---

## 项目信息

- **项目**: [ThunderLLAMA](https://github.com/lisihao/ThunderLLAMA) — 基于 llama.cpp 的 Apple Silicon 深度优化分支
- **Tag**: `v0.3.0-moe-fusion`
- **平台**: macOS + Apple Silicon (M4 Pro 36GB 验证)
- **模型**: Qwen3-30B-A3B (128 experts, top-8, 305 亿参数)
- **配置**: `thunderllama.conf` 统一管理全部 34 项优化参数
- **优化项**: 34 个（9 项 ThunderLLAMA 独有 + 25 项 llama.cpp 深度调优）

---

*2026-03-15*
