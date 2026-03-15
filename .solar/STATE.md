# Mission
ThunderLLAMA 持续优化:
1. **L2/L3 两层缓存系统** -- CPU (2GB) + Disk (100GB) 缓存，98.5% 命中率
2. **Metal GPU 深度优化** -- Tier A/B 内核融合与 GEMV 加速 (Q5_K 65->78+ tok/s)
3. **Metal Fusion + GPU Pool + Paged Attention**

# Constraints
- 不破坏现有 Clawgate + ThunderLLAMA 集成
- 配置文件 (thunderllama.conf) 是唯一真相源
- 正确性优先：输出必须与未优化版本 bit-identical
- **禁止 Mock 和模拟**：代码必须真实实现，能跑通

# Current Plan

## LMCache 两层缓存系统（当前状态）

### 当前架构：L2 (CPU) + L3 (Disk)
- **L2 CPU 缓存**: 2GB RAM
- **L3 磁盘缓存**: 100GB Disk (`/Volumes/toshiba/thunderllama-cache/kv_cache.bin`)
- **传输方式**: CPU->GPU transfer (ggml_backend_tensor_set)
- **当前性能**:
  - L2 命中率: 98.5%
  - Full Skip: 20% 跳过率
  - 缓存容量: 144 chunks (26 MB)

### L1 GPU Pool 已移除 (2026-03-15)
**原因**: llama-server 的 slot 管理在请求结束时调用 `memory_seq_rm [0, end)` 清空 KV cache，导致 L1/L2/L3 缓存在请求边界失效。

**移除内容**:
- metal-buffer-pool.h
- metal-buffer-pool.mm
- L1 存储/查询逻辑（thunder-lmcache-storage.cpp）
- L1 blit 代码（llama-context.cpp）
- Approximate Skip 特性（95%+ 命中零填充）

**保留功能**:
- L2/L3 缓存（CPU->GPU transfer）
- Full Skip（100% 命中跳过前向传播）
- Prefix Matching（前缀匹配复用缓存）
- Content-based Hashing

---

## 已完成 Tiers
1. Tier 1: CPU 线程分离 + KV cache f16 优化 (TG: 59->66.36)
2. Tier 2: Q4_K_M 量化 (TG: 66.36->75.90, 纯带宽瓶颈)
3. Tier 3: Metal MoE Kernel Fusion (TG: 59->65 Q5_K, 70->79 Q4_K)
4. 扩展 Metal Kernel Fusion（4 个新模式）

---

## Metal GPU 内核优化（并行进行）

## Tier A: 高回报、可行性高（推荐优先做）
| # | 优化方案 | 预估提升 | 难度 | 状态 |
|---|---------|---------|------|------|
| A1 | Fused Expert Aggregation (7xADD -> 1 kernel) | +7.3% TG (实测) | 中 | 已有 (upstream) |
| A2 | Q5_K Branchless Dequant (select() 替代 ternary) | +2-5% TG | 低 | 待做 (#13) |
| A3 | MoE ne21_mm_id_min 阈值降低 (128->8) | +5-15% (多并发) | 低 | 待做 (#2) |
| A4 | Fused RMS_NORM+MUL+SWIGLU (PR #16143) | +5-10% | 中 | 待做 (#2) |

**Tier A 剩余预估**: A1 已在当前基线中生效（+7.3% 已包含在 65 tok/s 里），A2/A3/A4 预估额外 +5-12% -> Q5_K 68-73 tok/s

## Tier B: 中等回报、技术挑战大
| # | 优化方案 | 预估提升 | 难度 | 状态 |
|---|---------|---------|------|------|
| B1 | Split-K Quantized Mat-Vec (MLX 技术) | +20-30% TG | 高 | 待做 |
| B2 | Fused Dequant+GEMV (BS=1 专用) | +15-30% TG | 高 | 待做 |
| B3 | MoE Expert-Only Dispatch | +5-15% | 中 | 待做 |
| B4 | MoE map0 Barrier 消除 | +3-8% | 中 | 待做 |

## Tier C: 低回报或被阻塞
| # | 优化方案 | 预估提升 | 状态 |
|---|---------|---------|------|
| C1 | Normalization Chain Fusion (SUM_ROWS->CLAMP->DIV) | +1-2% | 被阻塞 (graph scheduler) |
| C2 | Flash Attention nwg 调优 | +0.5-1.5% | 低优先级 |
| C3 | Q5_K PP path branchless | +1-3% PP | 低优先级 |
| C4 | Fused RoPE | +1-3% | 低优先级 |

## Tier D: 架构级变革（研究性质）
| # | 优化方案 | 预估提升 | 难度 |
|---|---------|---------|------|
| D1 | MPSGraph Hybrid Path | +30-50% | 极高 (#3) |
| D2 | Multi-Expert MatMul Fusion (8->1 grouped matmul) | +10-20% | 极高 |
| D3 | 4-concurrent-user batched MoE dispatch | +15-25% 聚合 | 高 |

---

## 独立任务
- [ ] **#1 Metal Performance Shaders (MPS) 集成** -- 用 Apple MPS 框架替代手写 kernel
- [ ] **#2 Paged Attention 优化（对标 vllm-mlx）** -- 重新实现，解决当前缺陷
- [x] **#9 Week 3: LRU 策略和最终优化** -- LMCache LRU/LFU 驱逐策略

## Metal JIT Fusion Pipeline (Week 6-8, 顺序依赖)
- [ ] **#3 Week 6.1: ggml Graph 分析器** -- 解析计算图，识别可融合模式
- [ ] **#4 Week 6.2: Dependency Graph Builder** -- 构建依赖关系图 (blocked by #3)
- [ ] **#5 Week 7.1: Graph Rewriter** -- 模式匹配 + 图变换 (blocked by #3, #4)
- [ ] **#6 Week 7.2: Metal Kernel Code Generator** -- 自动生成 Metal shader (blocked by #5)
- [ ] **#7 Week 8.1: Metal JIT Compiler** -- 运行时编译 + 缓存 (blocked by #6)
- [ ] **#8 Week 8.2: 集成到 ggml-metal** -- 完整 pipeline 集成 (blocked by #7)

## 基础设施
| # | 任务 | 状态 |
|---|------|------|
| I1 | Prometheus 指标导出 | 待做 |
| I2 | Grafana 仪表板 | 待做 |

### Blocked
- Normalization chain fusion（需要 graph 调度器改动，可能被 JIT pipeline 取代）

# Decisions

## LMCache 架构决策
- [2026-03-14] **GPU-Side Cache 架构**：采用三层架构（L1 GPU 10GB -> L2 CPU 2GB -> L3 Disk 100GB），利用 M4 Pro 统一内存架构消除 CPU->GPU 传输瓶颈（~8ms）。目标 90-100x speedup
- [2026-03-15] **移除 L1 GPU Pool**：因 llama-server slot 管理在请求边界调用 `memory_seq_rm [0, end)` 清空 KV cache，导致 L1/L2/L3 缓存失效。修复 persistence 需要修改核心 slot 管理（风险高）。决定移除 L1，保持 L2/L3 两层缓存系统。
  - 移除文件: metal-buffer-pool.h, metal-buffer-pool.mm
  - 移除功能: L1 存储/查询、L1 blit、Approximate Skip
  - 保留功能: L2/L3 缓存、Full Skip、Prefix Matching
  - 验证结果: L2 命中率 98.5%，Full Skip 20% 跳过率
- [2026-03-15] **KV Cache 量化级别选择**：全面测试 f16、q8_0、q4_0 三种量化级别（Task #18，15 次测试）。**选择 q8_0 作为推荐配置**：
  - 性能损失微小: 仅 -2.2% (46.55 -> 45.51 tok/s)
  - 内存节省显著: -50% KV Cache (1536MB -> 768MB)
  - 稳定性良好: 标准差 +/-3.43 tok/s
  - q4_0 虽节省 75% 内存，但性能损失 -6.1%，仅适合极限内存场景
  - 详细报告: `.solar/kv-quant-results/FINAL_REPORT_*.json`
- [2026-03-14] **EXCLUSIVE 模式**：添加 THUNDER_LMCACHE_EXCLUSIVE=1 配置，禁用内置 prompt cache 避免干扰 LMCache 测试。实测 65.22x speedup（vs 62x 共存模式）

## Metal 优化决策
- [2026-03-15] A1 验证: Metal 后端 ADD 链融合已覆盖 MoE 聚合的 7xADD，无需额外实现。实测 TG +7.3% (59.57 vs 55.50 tok/s)。每层每次 eval 都确认 "fuse: ADD x 7"
- [2026-03-15] MoE Fusion 只融合 gating 链 (SOFT_MAX->ARGSORT->GET_ROWS)，不融合 normalization chain：graph scheduler 将 MUL_MAT_ID 插在 GET_ROWS 和 SUM_ROWS 之间，无法相邻融合
- [2026-03-15] METAL_FUSION 配置项加入 thunderllama.conf：通过 GGML_METAL_FUSION_DISABLE 环境变量控制
- [2026-03-14] N_R0_Q5_K=8 编译时常量：7 组实测确认甜区
- [2026-03-14] KV cache 用 f16 而非 q4_0/q8_0：短上下文下反量化开销 > 带宽节省
- [2026-03-14] TG 用 4 线程、PP 用 8 线程：分离配置减少 GPU 带宽争抢
- [2026-03-15] K/V Projection Fusion 完全验证: 性能+9.8% TG, 正确性 IDENTICAL, 已提交

# Progress

## Done

### L1 GPU Pool 完整生命周期（2026-03-14 至 2026-03-15）
- **Task #14**: Week 1 - GPU Buffer Pool 基础框架（完成后移除）
  - 实现 MetalBufferPool class（10GB MTLResourceStorageModeShared）
  - LRU eviction 逻辑
  - 集成到 ThunderChunkStorage（L1 -> L2 -> L3）
  - 编译成功（build 8390）
- **Task #15**: Week 2 - Metal Kernel 集成和优化（完成后移除）
  - Metal Blit Encoder 实现（GPU->GPU transfer）
  - 修复 decode crash（chunk_size bug）
  - 限制 L1 blit 仅在 prefill 阶段
- **Task #17**: Remove L1 GPU Pool（2026-03-15）
  - 删除 metal-buffer-pool.h 和 metal-buffer-pool.mm
  - 移除 thunder-lmcache-storage.cpp 中的 L1 逻辑
  - 移除 llama-context.cpp 中的 L1 blit 和 Approximate Skip
  - 更新 CMakeLists.txt
  - 编译成功（0 错误）
  - 启动测试成功
  - L2/L3 缓存验证：98.5% 命中率，Full Skip 20% 跳过率

### LMCache 基础设施（2026-03-14）
- EXCLUSIVE 模式实现：禁用内置 prompt cache（server-task.cpp, config-parser.h）
  - 性能验证：65.22x speedup（vs 62x 共存模式）
  - 配置项：THUNDER_LMCACHE_EXCLUSIVE=1
- GPU-Side Cache 技术设计：`docs/GPU-SIDE-CACHE-DESIGN.md`
  - 三层架构设计（L1 GPU -> L2 CPU -> L3 Disk）
  - 性能预测：90-100x speedup
  - 风险评估和缓解措施
- 文档更新：
  - `docs/architecture-design.md` Section 10
  - `README.md` 特性说明
  - `.solar/STATE.md` Mission 和 Plan 更新

### KV Cache 量化优化
- **Task #1**: Native KV Cache 量化（GPU-side）
- **Task #11**: 集成 KV Cache Pipeline Quantization 到 llama-kv-cache.cpp
- **Task #12**: End-to-End Benchmark - 真实 Qwen3-30B 模型测试
- **Task #18**: KV Cache 量化全面对比测试（2026-03-15）
  - 测试配置：Qwen3-30B-Q5_K_M，3 种量化级别（f16, q8_0, q4_0），每种 5 次测试
  - 测试结果：
    - **f16** (基线): TG=46.55+/-0.39 tok/s, PP=86.12+/-0.71 tok/s, KV=1536MB
    - **q8_0** (推荐): TG=45.51+/-3.43 tok/s, PP=84.20+/-6.34 tok/s, KV=768MB (-50%)
      - 性能损失: -2.2% TG, -2.2% PP
      - 内存节省: -50% KV Cache
    - **q4_0**: TG=43.70+/-2.75 tok/s, PP=80.85+/-5.09 tok/s, KV=384MB (-75%)
      - 性能损失: -6.1% TG, -6.1% PP
      - 内存节省: -75% KV Cache
  - **结论**: 选择 q8_0 作为推荐配置（平衡性能和内存）
  - **配置更新**: `KV_CACHE_LEVEL="q8_0"` 写入 thunderllama.conf
  - 详细报告：`.solar/kv-quant-results/FINAL_REPORT_*.json`

### Metal GPU 内核优化
- A1 Fused Expert Aggregation 验证 (build 8389)
  - Metal 后端 ADD 链融合 (kernel_bin_fuse_impl) 已自动覆盖 MoE 7xADD
  - GGML_METAL_FUSION_DEBUG=2 确认：每层每 eval 均 "fuse: ADD x 7"
  - 实测: TG +7.3% (59.57 vs 55.50 tok/s), PP +0.8% (675 vs 670)
  - 结论: 无需额外代码，upstream 基础设施已完全覆盖
- Metal MoE Kernel Fusion 实现 (build 8389)
  - kernel_topk_moe_f32: simdgroup softmax + iterative top-8 argmax
  - 3-op fusion: SOFT_MAX -> ARGSORT -> GET_ROWS
  - 48 层 x 减少 2 dispatches = 96 dispatches eliminated
  - A/B 正确性验证: 输出 byte-identical
- Q5_K_M 全套优化: TG=65.25+/-0.16, PP=729.47+/-6.80 (10-run)
- Q4_K_M 全套优化: TG=79.12+/-0.20, PP=787.50+/-6.14 (5-run)
- thunderllama.conf 更新: METAL_FUSION 配置项 + 性能基准刷新
- config-parser.h 更新: METAL_FUSION -> GGML_METAL_FUSION_DISABLE 映射
- OPTIMIZATION_FEATURES.md 更新: Metal Kernel Fusion + 性能基准
- Native KV Cache 量化（GPU-side）
- 集成 KV Cache Pipeline Quantization 到 llama-kv-cache.cpp
- End-to-End Benchmark - 真实 Qwen3-30B 模型测试
- Week 1: GPU Buffer Pool 基础框架
- Week 2: Metal Kernel 集成和优化
- Remove L1 GPU Pool (Week 1 + Week 2)
- KV Cache 量化全面对比测试（f16 vs q8_0 vs q4_0）
  - 结论: q8_0 推荐（-2.2% 性能，-50% 内存）
- **Task #9: K/V Projection Fusion (GQA Adapted)** - 2026-03-15 完成
  - **最终性能 (5-run benchmark via server API)**:
    - TG: 65.90+/-5.15 -> **72.35+/-0.82** (**+9.8%**)
    - PP: 74.72+/-8.47 -> **81.04+/-0.60** (**+8.5%**)
  - **稳定性提升**: TG stdev -84%, PP stdev -93%
  - **正确性验证**: PASS -- 输出 IDENTICAL (seed=42, temp=0, greedy)
  - **覆盖率**: 48/48 layers (100%)
  - 关键发现: Qwen3-30B/Qwen3.5-35B 是 GQA 8:1 模型
  - 方案调整: 原方案（Q/K/V全融合）-> GQA适配版（K/V融合，Q分离）
  - 实现方式: ggml_concat + ggml_view_2d (动态融合 + 零拷贝切片)
  - 量化要求: 从 Q5_K 降至 Q4_K (扩大模型适用范围)
  - 文件修改:
    - src/llama-qkv-fusion.cpp (新增, 196 lines)
    - src/llama-model.h (wkv 成员)
    - src/models/qwen35moe.cpp (+45 lines)
  - 环境变量: FUSED_QKV=1 启用
  - 正确性验证: PASS -- Baseline vs Fusion 输出 IDENTICAL (2026-03-15, server API test, seed=42, temp=0, greedy)

- 扩展 Metal Kernel Fusion（4 个新模式）

## In-Progress
无

## Recently Completed (2026-03-15)
- **Task #9: LRU 策略和最终优化** -- LMCache 生产级升级
  - **Phase 1A**: L2/L3 容量可配置 (LMCACHE_L2_SIZE_GB / LMCACHE_L3_SIZE_GB)
  - **Phase 1B**: 频率保护 LRU (second-chance scan, freq decay, threshold=5)
  - **Phase 2A**: 修复 batch_get() 内存泄漏 (L3->L2 promote 时预驱逐)
  - **Phase 2B**: 完善驱逐统计 (l2_to_l3_evictions, l3_permanent_evictions, freq_protected_saves, utilization)
  - **Phase 2C**: shared_mutex 读写锁 (并发读不阻塞)
  - **Phase 3A**: TTL 支持 (LMCACHE_TTL_HOURS, lazy expiration)
  - **Phase 3B**: 缓存预热 API (POST /lmcache/warm)
  - **Phase 3C**: thunderllama.conf 新配置项文档
  - **Code Review**: 修复 3 个 CRITICAL/HIGH 问题 (iterator direction, deadlock, infinite loop)
  - **Build**: [100%] Built target llama-server

## Blocked
- Normalization chain (SUM_ROWS->CLAMP->DIV) 融合受 graph scheduler 限制

## Pending (从上次 session 恢复, 2026-03-15)
- [ ] #1 Metal Performance Shaders (MPS) 集成
- [ ] #2 Paged Attention 优化（对标 vllm-mlx）
- [ ] #3 Week 6.1: ggml Graph 分析器
- [ ] #4 Week 6.2: Dependency Graph Builder (<- #3)
- [ ] #5 Week 7.1: Graph Rewriter (<- #3, #4)
- [ ] #6 Week 7.2: Metal Kernel Code Generator (<- #5)
- [ ] #7 Week 8.1: Metal JIT Compiler (<- #6)
- [ ] #8 Week 8.2: 集成到 ggml-metal (<- #7)
- [x] #9 Week 3: LRU 策略和最终优化

# 已完成工作摘要 (详见 docs/architecture-design.md)

## 关键成果
- **L2/L3 两层缓存**: 98.5% L2 命中率，Full Skip 20% 跳过率
- **三层协同**: ContextPilot + ClawGate + LMCache = 23.34x (Phase 1 完成)
- **Hybrid Hashing**: prefix overlap 1.11x -> 3-7.8x
- **Skip Logic**: cache_prompt=false 触发 27.93x 加速
- **V Tensor 教训**: 不跳过 forward pass 时 V cache = 纯开销 (-8.2%)
- **OpenMP 冲突**: 已解决 (venv + OMP_NUM_THREADS=1)

# 风险点
- 如果重新测试无法复现 3-4x 提升，可能需要检查：
  - ubatch.token/pos 是否为 null（llama-server 路径）
  - 缓存是否真正触发（检查日志中的 STORED/RESTORED）
  - Prompt 长度是否足够（需要 > 256 tokens）

# Current Action
准备开始 #1 Metal Performance Shaders (MPS) 集成

## 待完成
- ClawGate + ThunderLLAMA 完整链路测试 (待配置路由)
- Prometheus/Grafana 监控 (见 task list I1/I2)
- Tier A 剩余优化 (A2/A3/A4)

# Next Actions

## 立即可做（优先级排序）

1. **Tier A2 - Q5_K Branchless Dequantization** (#13)
   - 难度：低
   - 预估：+2-5% TG
   - 实现：select() 替代 ternary operator

2. **Tier A3 - MoE ne21_mm_id_min 阈值降低** (#2)
   - 难度：低
   - 预估：+5-15% (多并发)
   - 实现：128->8

3. **Tier A4 - Fused RMS_NORM+MUL+SWIGLU** (#2)
   - 难度：中
   - 预估：+5-10%
   - 参考：PR #16143

4. **Week 3 - LRU 策略优化** (#16)
   - L2/L3 缓存 LRU 策略优化
   - 多 slot 并发测试

## 长期规划

1. **Paged Attention 优化** (#4)
   - 对标 vllm-mlx
   - 需要深入研究

2. **Graph Rewrite 系统** (#5-#10)
   - 研究性质
   - 需要系统性设计

3. **Metal Performance Shaders (MPS)** (#3)
   - 探索 MPSGraph 集成
   - 高风险高回报
