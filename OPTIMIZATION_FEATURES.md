# ThunderLLAMA 完整优化特性清单

> 基于代码扫描验证，包含所有可用优化项

## 快速索引

- [ThunderLLAMA 独有优化 (9项)](#1-thunderllama-独有优化9项)
- [Attention 机制优化 (2项)](#2-attention-机制优化2项)
- [KV Cache 优化 (7项)](#3-kv-cache-优化7项)
- [并发与批处理 (4项)](#4-并发与批处理优化4项)
- [硬件加速 (4项)](#5-硬件加速优化4项)
- [内存管理 (4项)](#6-内存管理优化4项)
- [推测优化 (2项)](#7-推测与特殊优化2项)
- [推荐配置](#推荐配置组合)

---

## 1. ThunderLLAMA 独有优化（9项）

| 优化项 | 启用方式 | 默认值 | 作用 | 性能提升 | 代码位置 |
|--------|---------|--------|------|---------|---------|
| **LMCache L2 (内存)** | `THUNDER_LMCACHE=1` | 禁用 | 8GB 内存缓存池 | - | `src/llama-context.cpp:442` |
| **LMCache L3 (磁盘)** | `THUNDER_LMCACHE=1` | 禁用 | 256GB 磁盘持久化缓存 | 跨会话复用 | `src/llama-context.cpp:443` |
| **Hybrid Hashing** | 自动 (LMCACHE=1) | - | xxHash64 内容+位置双哈希 | 3-7x (prefix overlap) | `src/thunder-lmcache-hash.cpp` |
| **Full Skip Logic** | 自动 (LMCACHE=1) | - | 100% 命中跳过计算 | **27x 加速** | `src/llama-context.cpp:1341` |
| **Approximate Skip** | 自动 (LMCACHE=1) | - | 95%+ 命中零填充跳过 | Skip 覆盖率 5%→30% | `src/llama-context.cpp:1341-1360` |
| **Smart Prefetch** | 自动 (LMCACHE=1) | - | 访问频率跟踪 + 并行 I/O | 4x L3 预取加速 | `src/thunder-lmcache-storage.cpp` |
| **Data Compression** | 自动 (L3 层) | - | zlib 压缩 | 2-4x 存储节省 | `src/thunder-lmcache-storage.cpp:12` |
| **Checksum Validation** | 自动 (L3 层) | - | XXH64 完整性校验 | 数据完整性保证 | `src/thunder-lmcache-storage.cpp:18` |
| **Adaptive Chunk Prefill** | `THUNDERLLAMA_CHUNK_PREFILL=N` | `max(32, n_batch/slots)` | 自适应分块大小 | 减少延迟抖动 | `tools/server/server-context.cpp:2120` |
| **LRU Freq-Protected** | `LMCACHE_FREQ_PROTECT=N` | `5` | 高频 chunk 免于驱逐 (second-chance) | 热点数据不被挤出 | `src/thunder-lmcache-storage.cpp` |
| **Cache Warm API** | `POST /lmcache/warm` | - | L3→L2 热数据预加载 | 冷启动加速 | `tools/server/server-context.cpp` |
| **TTL Expiration** | `LMCACHE_TTL_HOURS=N` | `0` (永不过期) | 过期 chunk 自动清理 | 防止缓存膨胀 | `src/thunder-lmcache-storage.cpp` |
| **Configurable Capacity** | `LMCACHE_L2_SIZE_GB` / `L3_SIZE_GB` | `8` / `256` | L2/L3 容量可配置 | 灵活适配硬件 | `thunderllama.conf` |

## 2. Attention 机制优化（2项）

| 优化项 | 启用方式 | 默认值 | 作用 | 性能提升 | 来源 |
|--------|---------|--------|------|---------|------|
| **Flash Attention** | `-fa on` | `auto` | O(sqrt(n)) 注意力计算 | **20-30% 加速** | llama.cpp |
| **Paged Attention** | `LLAMA_PAGED_ATTENTION=1` | 禁用 | KV cache 分页管理 | 8x jitter 降低 (6.0%→0.7%) | ThunderLLAMA 修复版 |

## 3. KV Cache 优化（7项）

| 优化项 | 启用方式 | 默认值 | 作用 | 性能提升 | 来源 |
|--------|---------|--------|------|---------|------|
| **KV Cache 量化 (K)** | `-ctk q8_0/q4_0` | `f16` | 量化 Key 缓存 | q8_0: ↓50%, q4_0: ↓75% 内存 | llama.cpp |
| **KV Cache 量化 (V)** | `-ctv q8_0/q4_0` | `f16` | 量化 Value 缓存 | q8_0: ↓50%, q4_0: ↓75% 内存 | llama.cpp |
| **Cache RAM 限制** | `-cram N` (MiB) | `8192` | 限制 KV cache 最大内存 | 可控内存占用 | llama.cpp |
| **KV Offload** | `-kvo` | 启用 | KV cache 卸载到 GPU | Metal GPU 加速 | llama.cpp |
| **Prefix Caching** | `--cache-prompt` | 禁用 | 缓存 prompt KV | **70-80% TTFT 降低** | llama.cpp (ThunderLLAMA 增强) |
| **Prompt Reuse** | `--prompt-reuse-mode auto/on` | `auto` | 跨请求复用 prompt KV | **>100x 命中加速** | llama.cpp |
| **Slot Similarity** | `-sps 0.5` | `0.10` | Prompt 相似度阈值 | 提高 reuse 命中率 | llama.cpp |

## 4. 并发与批处理优化（4项）

| 优化项 | 启用方式 | 默认值 | 作用 | 性能提升 | 来源 |
|--------|---------|--------|------|---------|------|
| **Continuous Batching** | `-cb` | 启用 | 动态批处理 | **40-60% 吞吐提升** | llama.cpp |
| **Parallel Slots** | `-np N` | `-1` (auto) | 并发请求槽位数 | 并发能力 | llama.cpp |
| **Batch Size** | `-b N` | `2048` | 逻辑批大小 | 影响吞吐上限 | llama.cpp |
| **Micro Batch** | `-ub N` | `512` | 物理批大小 | GPU 利用率 | llama.cpp |

## 5. 硬件加速优化（7项）

| 优化项 | 启用方式 | 默认值 | 作用 | 性能提升 | 来源 |
|--------|---------|--------|------|---------|------|
| **GPU Layers** | `-ngl 99` | `auto` | 卸载层数到 GPU | M4: 3-5x vs CPU | llama.cpp |
| **Metal 优化** | 自动启用 (macOS) | - | Apple Silicon GPU 加速 | 自动优化 | llama.cpp |
| **Metal Kernel Fusion** | `METAL_FUSION=1` | 启用 | ADD 融合 + MoE Gating 融合 | **TG +10-12%** | ThunderLLAMA |
| **K/V Projection Fusion** | `FUSED_QKV=1` | 启用 | K/V 权重融合 (GQA 适配) | **TG +9.8%, PP +8.5%** | ThunderLLAMA |
| **N_R0_Q5_K 调优** | 编译时 | `8` | Q5_K 每 simdgroup 处理行数 | **TG +2-3%** | ThunderLLAMA |
| **CPU 线程** | `-t N` | `-1` (auto) | CPU 推理线程数 | 少量层用 CPU 时有效 | llama.cpp |
| **Batch 线程** | `-tb N` | 同 `-t` | Prompt 处理线程数 | Prompt 阶段加速 | llama.cpp |

## 6. 内存管理优化（4项）

| 优化项 | 启用方式 | 默认值 | 作用 | 性能提升 | 来源 |
|--------|---------|--------|------|---------|------|
| **mmap** | `--mmap` | 启用 | 内存映射模型文件 | 减少加载时间 | llama.cpp |
| **mlock** | `--mlock` | 禁用 | 锁定模型在 RAM | 防止 swap | llama.cpp |
| **SWA Full Mode** | `--swa-full` | 禁用 | 全尺寸滑动窗口缓存 | 高内存场景优化 | llama.cpp |
| **Safe USB Unmount** | `kill -USR1 <pid>` | - | 优雅卸载 L3 磁盘缓存 | USB 安全移除 | ThunderLLAMA 独有 |

## 7. 推测与特殊优化（2项）

| 优化项 | 启用方式 | 默认值 | 作用 | 性能提升 | 来源 |
|--------|---------|--------|------|---------|------|
| **Speculative Decoding** | `--draft-model <path>` | 禁用 | 推测解码 | **2-3x 加速** | llama.cpp |
| **Session Cache** | 默认启用 | 启用 | 会话缓存 | 会话复用 | llama.cpp |

---

## 环境变量速查表

| 环境变量 | 用途 | 默认值 | 代码位置 |
|---------|------|--------|---------|
| `THUNDER_LMCACHE` | 启用 LMCache 系统 | `0` | `src/llama-context.cpp:426` |
| `THUNDER_LMCACHE_DISK_PATH` | L3 磁盘缓存路径 | `~/.cache/thunderllama/kv_cache.bin` | `src/thunder-lmcache-storage.cpp:32` |
| `THUNDERLLAMA_CHUNK_PREFILL` | 自适应分块大小 | `max(32, n_batch/slots)` | `tools/server/server-context.cpp:2120` |
| `LLAMA_PAGED_ATTENTION` | 启用 Paged Attention | `0` | `src/llama-context.cpp:293` |
| `GGML_METAL_FUSION_DISABLE` | 禁用 Metal 内核融合 | 未设置(启用) | `ggml/src/ggml-metal/ggml-metal-context.m` |
| `FUSED_QKV` | 启用 K/V Projection Fusion | `1` | `src/llama-qkv-fusion.cpp` |
| `LMCACHE_L2_SIZE_GB` | L2 内存缓存容量 (GB) | `8` | `src/llama-context.cpp` |
| `LMCACHE_L3_SIZE_GB` | L3 磁盘缓存容量 (GB) | `256` | `src/llama-context.cpp` |
| `LMCACHE_FREQ_PROTECT` | 频率保护阈值 | `5` | `src/thunder-lmcache-storage.cpp` |
| `LMCACHE_TTL_HOURS` | chunk 最大存活时间 (小时) | `0` (永不过期) | `src/thunder-lmcache-storage.cpp` |

---

## 推荐配置组合

### 🚀 配置 1: 高性能 Agent 场景

```bash
THUNDER_LMCACHE=1 \
./build/bin/llama-server \
  -m model.gguf \
  -c 8192 \
  -ngl 99 \
  -fa on \
  -cb \
  -np 8 \
  -b 2048 \
  --cache-prompt \
  --prompt-reuse-mode auto \
  -sps 0.5
```

**启用优化**: 11 个
**预期效果**: 首次正常，重复请求 **10-27x 加速**

---

### 💾 配置 2: 内存优化模式

```bash
THUNDER_LMCACHE=1 \
./build/bin/llama-server \
  -m model.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  -cb \
  -ctk q8_0 \
  -ctv q8_0 \
  -cram 4096 \
  --prompt-reuse-mode on
```

**启用优化**: 10 个
**预期效果**: 内存 **↓50%**，性能轻微下降 (<3%)

---

### 🔥 配置 3: 极限内存优化

```bash
THUNDER_LMCACHE=1 \
./build/bin/llama-server \
  -m model.gguf \
  -c 2048 \
  -ngl 99 \
  -fa on \
  -cb \
  -ctk q4_0 \
  -ctv q4_0 \
  -cram 2048
```

**启用优化**: 8 个
**预期效果**: 内存 **↓75%**，质量轻微下降 (~3%)

---

## 监控与管理

### LMCache 统计 API

```bash
curl http://localhost:8080/lmcache/stats
```

返回示例：
```json
{
  "total_prefills": 100,
  "skip_count": 30,
  "approx_skip_count": 15,
  "total_skip_rate": 0.45,
  "l2_chunks": 432,
  "l2_usage_bytes": 15138816,
  "l2_limit_bytes": 8589934592,
  "l2_utilization": 0.0018,
  "l3_chunks": 48,
  "l3_usage_bytes": 293830656,
  "l3_limit_bytes": 274877906944,
  "l3_utilization": 0.0011,
  "total_chunks": 480,
  "l2_to_l3_evictions": 0,
  "l3_permanent_evictions": 0,
  "freq_protected_saves": 0,
  "l2_hit_rate": 0.9999
}
```

### Thunder Cache CLI

```bash
# 查看统计
./build/bin/thunder-cache stats

# 清理缓存
./build/bin/thunder-cache clear

# 压缩缓存
./build/bin/thunder-cache compact
```

---

## 总计

| 类别 | 优化项数量 |
|------|-----------|
| **ThunderLLAMA 独有** | 13 |
| **Attention 机制** | 2 |
| **KV Cache 优化** | 7 |
| **并发批处理** | 4 |
| **硬件加速** | 6 |
| **内存管理** | 4 |
| **推测优化** | 2 |
| **总计** | **38 个可用优化项** |

---

## 性能基准 (M4 Pro, 2026-03-15)

### Qwen3-30B-A3B (全套优化: FA=1, Fusion=1, t=4)

| 量化 | 大小 | PP512 (tok/s) | TG128 (tok/s) |
|------|------|:------------:|:------------:|
| **Q5_K_M** | 20.23 GiB | 729.47 ± 6.80 | **65.25 ± 0.16** |
| **Q4_K_M** | 17.28 GiB | 787.50 ± 6.14 | **79.12 ± 0.20** |

### Metal Kernel Fusion 提速效果

| 量化 | 无融合 | 有融合 | 提升 |
|------|:------:|:-----:|:----:|
| Q5_K_M | 59.07 | 65.25 | **+10.5%** |
| Q4_K_M | 70.35 | 79.12 | **+12.5%** |

### K/V Projection Fusion 提速效果 (2026-03-15, 5-run benchmark)

| 指标 | Baseline (FUSED_QKV=0) | Fusion (FUSED_QKV=1) | 提升 |
|------|:---------------------:|:--------------------:|:----:|
| **TG tok/s** | 65.90 ± 5.15 | **72.35 ± 0.82** | **+9.8%** |
| **PP tok/s** | 74.72 ± 8.47 | **81.04 ± 0.60** | **+8.5%** |

- **稳定性提升**: TG 标准差 5.15 → 0.82 (-84%), PP 标准差 8.47 → 0.60 (-93%)
- **正确性验证**: Baseline vs Fusion 输出 IDENTICAL (seed=42, temp=0, greedy)
- **覆盖率**: 48/48 layers (100%) on Qwen3-30B-A3B
- **实现**: `ggml_concat(wk, wv)` + `ggml_view_2d` 切片 (GQA 8:1 适配)

---

**文档版本**: v1.3
**更新日期**: 2026-03-15
**验证方式**: 代码扫描 + 文档审查 + 性能基准测试 + 正确性验证
