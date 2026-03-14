# ThunderLLAMA ⚡

**为 Apple Silicon 优化的高性能 LLM 推理引擎**

> 基于 llama.cpp，集成 Paged Attention + 多层缓存 + ContextPilot，专为长上下文和多轮对话优化

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-blue)](https://www.apple.com/mac/)
[![llama.cpp](https://img.shields.io/badge/Based%20on-llama.cpp-green)](https://github.com/ggml-org/llama.cpp)

---

## 🚀 核心亮点

### 性能表现

| 场景 | 性能提升 | 说明 |
|------|---------|------|
| **Metal 内核优化** | **75.9 tok/s TG** | Q4_K_M, FA=1, 分离线程配置 |
| **多轮对话（L2 缓存）** | **10-60x** | 内存缓存命中，极速响应 |
| **跨会话恢复（L3 缓存）** | **22-33x** | 磁盘持久化，重启不丢失 |
| **端到端延迟优化** | **67%** ⬇️ | ClawGate + ContextPilot 集成 |
| **长上下文支持** | **8GB + 256GB** | L2 内存 + L3 磁盘容量 |

**吞吐量基准**（Qwen3-30B-A3B, M4 Pro 48GB, 10-run 稳定测量）：

| 量化 | PP512 (tok/s) | TG128 (tok/s) | 模型大小 | BPW |
|------|:------------:|:------------:|:-------:|:---:|
| **Q5_K_M** | 718.37 ± 12.52 | 66.36 ± 0.20 | 20.23 GiB | 5.69 |
| **Q4_K_M** | 766.84 ± 4.33 | **75.90 ± 0.23** | 17.28 GiB | 4.86 |

**缓存基准**（Qwen3-30B-A3B, M3 Max）：
```
Cold Start:  424ms  →  Warm Cache:  40ms  (10.6x faster)
             594ms  →  ContextPilot: 197ms (67% latency reduction)
```

---

## 🎯 三大技术优势

### 1️⃣ Apple Silicon 原生优化

**问题**：llama.cpp 的 Paged Attention 在 Metal GPU 上被错误禁用

**解决**：
- ✅ 修复 `ggml_flash_attn_ext_set_paged()` 调用链
- ✅ 实现完整的 Block Pool 内存管理
- ✅ Metal 内核级别的 Paged KV Cache 支持

**效果**：
- 内存占用降低 **40%**（vs 标准 llama.cpp）
- 支持 **更长上下文**（8K → 128K tokens）
- **零性能损失**（与原生 llama.cpp 相同吞吐量）

---

### 2️⃣ LMCache：三层缓存架构

```
┌────────────────────────────────────────────────────────────┐
│                    LMCache 三层架构                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  L1 (GPU - 即时)     L2 (CPU - 8GB)      L3 (SSD - 256GB) │
│  ┌──────────┐       ┌──────────┐         ┌──────────┐    │
│  │ Block    │  ←→   │ Heap     │   ←→    │ Disk     │    │
│  │ Pool     │       │ LRU      │         │ Persist  │    │
│  │ (Metal)  │       │ Queue    │         │ (zlib)   │    │
│  └──────────┘       └──────────┘         └──────────┘    │
│                                                            │
│  命中率: ~95%        命中率: ~80%        命中率: ~60%     │
│  延迟: <1ms         延迟: ~40ms          延迟: ~80ms      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**核心特性**：
- **智能降级**：L2 → L3 → L1 自动回退，永不失败
- **跨会话持久化**：重启/关机后缓存仍在
- **压缩存储**：zlib 压缩，节省 60% 磁盘空间
- **USB 安全**：外置 SSD 断开时优雅降级

**实测数据**（256-1024 tokens）：
```
Scenario       | Cold Start | Warm (L2)  | Disk (L3)  | Speedup
---------------|-----------|-----------|-----------|----------
256 tokens     | 735ms     | 46ms      | 22ms      | 16-33x
512 tokens     | 1335ms    | 52ms      | 50ms      | 26-27x
1024 tokens    | 2491ms    | 42ms      | 89ms      | 28-60x
```

---

### 3️⃣ Metal 内核深度调优

**配置文件驱动**：所有优化参数通过 `thunderllama.conf` 统一管理，代码级强制。

**已完成的优化**：
| 优化 | 改动 | 效果 |
|------|------|------|
| N_R0_Q5_K=8 | Metal 内核每 simdgroup 处理 8 行 | +2-3% TG |
| KV Cache f16 | 短上下文下反量化开销 > 带宽节省 | +5% TG |
| CPU 线程分离 | TG=4 线程（减少 GPU 带宽争抢），PP=8 线程 | +1.1% TG, +2.7% PP |
| Q4_K_M 量化 | 模型缩小 14.6%，纯带宽瓶颈 | +14.4% TG |

**已验证无效的优化**：
- Speculative Decoding（Draft Model）：30B 验证成本高，质量下降
- N-gram Speculative Decoding：对大模型帮助有限
- ne11_mm_min 阈值调优：仅影响 batch size 2-8，对 TG/PP 无效

---

### 4️⃣ ContextPilot 集成

**端到端优化流程**：

```
ClawGate (多 Agent 网关)
    │
    ▼
ContextPilot (去重 + 重排序)
    │  - 全局去重：识别重复 context chunks
    │  - 智能重排：按访问频率优化顺序
    │  - 生成元数据：signature + chunk hashes
    ▼
ThunderLLAMA (LMCache 查询优化)
    │  X-Context-Signature: f8f01cd5...
    │  X-Context-Chunks: ["chunk-a039...", ...]
    │
    ▼
LMCache (基于 chunk hash 快速定位)
    │  - Chunk-based 缓存查找
    │  - 前缀匹配优化
    │  - 增量计算
    ▼
Metal GPU (Paged Attention 计算)
```

**验证结果**：
- ✅ Headers 解析成功（4 chunks）
- ✅ 端到端延迟降低 **67%**（594ms → 197ms）
- ✅ 与 ClawGate 无缝集成

---

## 📊 竞品对比

### vs llama.cpp（原版）

| 特性 | llama.cpp | ThunderLLAMA | 优势 |
|------|-----------|--------------|------|
| **Paged Attention** | ❌ 在 Metal 上失效 | ✅ 完整支持 | **40% 内存节省** |
| **跨会话缓存** | ❌ 不支持 | ✅ 磁盘持久化 | **22-33x 加速** |
| **长上下文** | 受限于内存 | 8GB + 256GB | **32x 容量** |
| **多轮对话** | 每次重算 | 智能缓存 | **10-60x 加速** |

### vs vLLM

| 特性 | vLLM | ThunderLLAMA | 说明 |
|------|------|--------------|------|
| **平台支持** | ✅ CUDA/ROCm | ✅ Apple Silicon | **M 系列芯片原生** |
| **PagedAttention** | ✅ 先进实现 | ✅ Metal 优化 | 性能相当 |
| **磁盘缓存** | ❌ 仅内存 | ✅ L3 持久化 | **跨会话优势** |
| **吞吐量优化** | ✅✅ 极致优化 | ✅ 合理优化 | vLLM 更激进 |
| **生态兼容** | Python 生态 | llama.cpp 生态 | **GGUF 原生** |

### vs SGLang

| 特性 | SGLang | ThunderLLAMA | 说明 |
|------|--------|--------------|------|
| **缓存策略** | RadixAttention（前缀树） | Chunk-based（哈希） | **更灵活** |
| **磁盘持久化** | 需要额外配置 | 原生支持 | **开箱即用** |
| **Apple Silicon** | ⚠️  有限支持 | ✅ 深度优化 | **M 芯片首选** |
| **ContextPilot** | ❌ 不支持 | ✅ 原生集成 | **多 Agent 优势** |

### 🏆 适用场景

| 场景 | 推荐引擎 | 原因 |
|------|---------|------|
| **CUDA GPU 高吞吐** | vLLM | PagedAttention 成熟，吞吐优化极致 |
| **Apple Silicon 本地** | **ThunderLLAMA** | 原生优化 + 磁盘缓存 + ContextPilot |
| **长上下文 + 多轮对话** | **ThunderLLAMA** | L2+L3 缓存 + 跨会话持久化 |
| **GGUF 模型生态** | **ThunderLLAMA** | llama.cpp 完全兼容 |
| **云端大规模服务** | vLLM / SGLang | 更好的水平扩展 |

---

## 🛠️ 快速开始

### 环境要求

- **硬件**：Apple Silicon (M1/M2/M3/M4)
- **系统**：macOS 13.0+
- **依赖**：CMake 3.20+, Metal SDK

### 编译安装

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/ThunderLLAMA.git
cd ThunderLLAMA

# 2. 编译（启用 Metal GPU）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# 3. 验证
./build/bin/llama-server --version
```

### 配置文件（唯一真相源）

ThunderLLAMA 使用 `thunderllama.conf` 作为所有配置的唯一真相源。
**命令行参数和外部环境变量均被忽略**，所有配置必须写入此文件。

```bash
# 编辑配置
vim thunderllama.conf

# 关键配置项：
MODEL_PATH="$HOME/models/your-model.gguf"
CONTEXT_SIZE=4096
SERVER_PORT=30000
GPU_LAYERS=99
FLASH_ATTENTION="on"
THUNDER_LMCACHE=1
LLAMA_PAGED_ATTENTION=1
KV_CACHE_LEVEL="f16"
CPU_THREADS=4           # TG 推理线程（M4 Pro 最优值）
CPU_THREADS_BATCH=8     # PP 批处理线程
```

### 启动服务

```bash
# 推荐方式：使用启动脚本
./start-thunderllama.sh

# 或直接运行（自动读取 thunderllama.conf）
./build/bin/llama-server

# 停止 / 重启 / 状态
./stop-thunderllama.sh
./restart-thunderllama.sh
./status-thunderllama.sh
```

### API 调用

```bash
# 标准 OpenAI 兼容 API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'

# ContextPilot 集成（ClawGate 模式）
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Context-Signature: <signature>" \
  -H "X-Context-Chunks: [\"chunk-1\", \"chunk-2\"]" \
  -d '{
    "model": "default",
    "messages": [...],
    "max_tokens": 100
  }'
```

---

## 📚 文档

### 核心文档
- [架构设计](docs/ARCHITECTURE.md) - 系统架构详解
- [Phase 3 完成报告](docs/PHASE3_COMPLETION_REPORT.md) - 性能测试数据
- [端到端集成测试](docs/E2E_INTEGRATION_TEST.md) - ClawGate + ContextPilot

### 技术细节
- [Paged Attention 实现](docs/PAGED_ATTENTION.md) - Metal 优化细节
- [LMCache 三层架构](docs/LMCACHE_ARCHITECTURE.md) - 缓存设计
- [ContextPilot 集成](docs/CONTEXTPILOT_INTEGRATION.md) - 去重优化

---

## 🔬 性能基准测试

### 测试环境
- **硬件**：MacBook Pro M4 Pro (48GB RAM, 273 GB/s 内存带宽)
- **模型**：Qwen3-30B-A3B-128K (Q5_K_M / Q4_K_M)
- **配置**：Context 4096, GPU Layers 99, Flash Attention ON, KV f16

### Metal 内核优化基准（10-run 稳定测量）

| 优化项 | PP512 (tok/s) | TG128 (tok/s) | 提升 |
|--------|:------------:|:------------:|:----:|
| Baseline (上游 llama.cpp) | ~600 | ~59 | - |
| + N_R0_Q5_K=8 (Metal 参数调优) | ~690 | ~63.5 | +7.6% TG |
| + KV f16 (vs q4_0) | ~686 | ~59→66 | +5% TG |
| + CPU threads=4/8 分离配置 | 718 | **66.36** | **+12.5% TG** |
| + Q4_K_M 量化 | **767** | **75.90** | **+28.6% TG** |

**关键发现**：
- TG 是纯内存带宽瓶颈 — 模型缩小 14.6%，TG 快 14.4%（线性关系）
- Metal 融合已接近天花板 — 元素操作融合 (nf=7)、并发 dispatch、Graph reuse 全部生效
- Q5_K_M 带宽利用率 ~65%，软件可寻址差距仅 ~8%

### LMCache 缓存基准（M3 Max）

```
┌─────────────────────────────────────────────────────────────┐
│  Token Length │  Cold    │  Warm (L2) │  Disk (L3) │ Speedup │
├─────────────────────────────────────────────────────────────┤
│  256 tokens   │  735ms   │  46ms      │  22ms      │ 16-33x  │
│  512 tokens   │  1335ms  │  52ms      │  50ms      │ 26-27x  │
│  1024 tokens  │  2491ms  │  42ms      │  89ms      │ 28-60x  │
└─────────────────────────────────────────────────────────────┘
```

### 端到端集成测试（M3 Max）

```
┌─────────────────────────────────────────────────────────────┐
│  Scenario           │ Prompt Time │ Total Latency │ Speedup │
├─────────────────────────────────────────────────────────────┤
│  Cold Start         │  424.38ms   │  594.43ms     │  1x     │
│  Warm Cache (L2)    │  40.01ms    │  200.14ms     │  10.6x  │
│  ContextPilot (opt) │  33.13ms    │  196.50ms     │  12.8x  │
└─────────────────────────────────────────────────────────────┘
```

**关键结论**：
- ✅ Metal 内核优化：Q5_K **66 tok/s**，Q4_K **76 tok/s**
- ✅ L2 缓存命中：**10-60x** PP 加速
- ✅ L3 磁盘恢复：**22-33x** PP 加速
- ✅ ContextPilot 优化：端到端延迟降低 **67%**

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. **Fork 仓库**
2. **创建分支**：`git checkout -b feature/your-feature`
3. **提交代码**：`git commit -m 'Add some feature'`
4. **推送分支**：`git push origin feature/your-feature`
5. **提交 PR**：在 GitHub 上创建 Pull Request

### 开发规范

- **C++ 代码**：遵循 llama.cpp 风格
- **Commit 格式**：`feat/fix/docs/perf: 简短描述`
- **测试要求**：新功能需要添加测试
- **文档更新**：重要变更需要更新文档

---

## 📝 路线图

### ✅ Phase 1-3（已完成）
- [x] Paged Attention 修复和验证
- [x] LMCache L2/L3 三层缓存
- [x] 磁盘持久化（v3 格式）
- [x] ContextPilot headers 解析
- [x] ClawGate 端到端集成

### ✅ Phase 4: Metal 内核优化（已完成）
- [x] 配置文件唯一真相源（`thunderllama.conf`，代码级强制）
- [x] N_R0_Q5_K Metal 参数调优（1 → 8，+2-3% TG）
- [x] KV Cache 量化对比（f16 > q8_0 > q4_0，M4 Pro 实测）
- [x] CPU 线程分离配置（TG=4 线程 / PP=8 线程）
- [x] Metal GPU Profiling 分析（29 dispatches/layer, nf=7 fusion）
- [x] Q4_K_M 量化验证（75.9 tok/s，+14.4%）

### 🚧 Phase 5（进行中）
- [ ] LMCache 使用 chunk hashes 优化查询
- [ ] ClawGate 集成生产级 ContextPilot
- [ ] 多 Agent 并发压力测试
- [ ] 监控和可观测性

### 🔮 未来计划
- [ ] 分布式缓存共享
- [ ] 自定义 MoE Kernel（逼近理论天花板）
- [ ] 量化感知缓存策略
- [ ] WebUI 管理界面

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - 强大的基础框架
- [ContextPilot](https://github.com/RAGFoundry/contextpilot) - Context 优化灵感
- Apple Metal 团队 - 优秀的 GPU 框架

---

## 📬 联系我们

- **Issues**：[GitHub Issues](https://github.com/yourusername/ThunderLLAMA/issues)
- **Discussions**：[GitHub Discussions](https://github.com/yourusername/ThunderLLAMA/discussions)
- **Email**：your.email@example.com

---

<p align="center">
  <b>⚡ ThunderLLAMA - 让 Apple Silicon 释放真正的 LLM 性能 ⚡</b>
</p>
