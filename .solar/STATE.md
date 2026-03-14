# Mission
ThunderLLAMA Metal GPU 深度优化 — Tier A/B 内核融合与 GEMV 加速 (Q5_K 65→78+ tok/s)

# Constraints
- 不破坏现有 Clawgate + ThunderLLAMA 集成
- 配置文件 (thunderllama.conf) 是唯一真相源
- 正确性优先：输出必须与未优化版本 bit-identical

# Current Plan

## 已完成
1. ✅ Tier 1: CPU 线程分离 + KV cache f16 优化 (TG: 59→66.36)
2. ✅ Tier 2: Q4_K_M 量化 (TG: 66.36→75.90, 纯带宽瓶颈)
3. ✅ Tier 3: Metal MoE Kernel Fusion (TG: 59→65 Q5_K, 70→79 Q4_K)

## Tier A: 高回报、可行性高（推荐优先做）
| # | 优化方案 | 预估提升 | 难度 | 状态 |
|---|---------|---------|------|------|
| A1 | Fused Expert Aggregation (7×ADD → 1 kernel) | +3-5% TG | 中 | ⏳ 待做 |
| A2 | Q5_K Branchless Dequant (select() 替代 ternary) | +2-5% TG | 低 | ⏳ 待做 |
| A3 | MoE ne21_mm_id_min 阈值降低 (128→8) | +5-15% (多并发) | 低 | ⏳ 待做 |
| A4 | Fused RMS_NORM+MUL+SWIGLU (PR #16143) | +5-10% | 中 | ⏳ 待做 |

**Tier A 累计预估**: +10-20% → Q5_K 72-78 tok/s

## Tier B: 中等回报、技术挑战大
| # | 优化方案 | 预估提升 | 难度 | 状态 |
|---|---------|---------|------|------|
| B1 | Split-K Quantized Mat-Vec (MLX 技术) | +20-30% TG | 高 | ⏳ 待做 |
| B2 | Fused Dequant+GEMV (BS=1 专用) | +15-30% TG | 高 | ⏳ 待做 |
| B3 | MoE Expert-Only Dispatch | +5-15% | 中 | ⏳ 待做 |
| B4 | MoE map0 Barrier 消除 | +3-8% | 中 | ⏳ 待做 |

## Tier C: 低回报或被阻塞
| # | 优化方案 | 预估提升 | 状态 |
|---|---------|---------|------|
| C1 | Normalization Chain Fusion (SUM_ROWS→CLAMP→DIV) | +1-2% | ⛔ 被阻塞 (graph scheduler) |
| C2 | Flash Attention nwg 调优 | +0.5-1.5% | 低优先级 |
| C3 | Q5_K PP path branchless | +1-3% PP | 低优先级 |
| C4 | Fused RoPE | +1-3% | 低优先级 |

## Tier D: 架构级变革（研究性质）
| # | 优化方案 | 预估提升 | 难度 |
|---|---------|---------|------|
| D1 | MPSGraph Hybrid Path | +30-50% | 极高 |
| D2 | Multi-Expert MatMul Fusion (8→1 grouped matmul) | +10-20% | 极高 |
| D3 | 4-concurrent-user batched MoE dispatch | +15-25% 聚合 | 高 |

## 基础设施
| # | 任务 | 状态 |
|---|------|------|
| I1 | Prometheus 指标导出 | ⏳ 待做 |
| I2 | Grafana 仪表板 | ⏳ 待做 |

# Decisions
- [2026-03-15] MoE Fusion 只融合 gating 链 (SOFT_MAX→ARGSORT→GET_ROWS)，不融合 normalization chain：graph scheduler 将 MUL_MAT_ID 插在 GET_ROWS 和 SUM_ROWS 之间，无法相邻融合
- [2026-03-15] METAL_FUSION 配置项加入 thunderllama.conf：通过 GGML_METAL_FUSION_DISABLE 环境变量控制
- [2026-03-14] N_R0_Q5_K=8 编译时常量：7 组实测确认甜区
- [2026-03-14] KV cache 用 f16 而非 q4_0/q8_0：短上下文下反量化开销 > 带宽节省
- [2026-03-14] TG 用 4 线程、PP 用 8 线程：分离配置减少 GPU 带宽争抢

# Progress

## Done
- ✅ Metal MoE Kernel Fusion 实现 (build 8389)
  - kernel_topk_moe_f32: simdgroup softmax + iterative top-8 argmax
  - 3-op fusion: SOFT_MAX → ARGSORT → GET_ROWS
  - 48 层 × 减少 2 dispatches = 96 dispatches eliminated
  - A/B 正确性验证: 输出 byte-identical
- ✅ Q5_K_M 全套优化: TG=65.25±0.16, PP=729.47±6.80 (10-run)
- ✅ Q4_K_M 全套优化: TG=79.12±0.20, PP=787.50±6.14 (5-run)
- ✅ thunderllama.conf 更新: METAL_FUSION 配置项 + 性能基准刷新
- ✅ config-parser.h 更新: METAL_FUSION → GGML_METAL_FUSION_DISABLE 映射
- ✅ OPTIMIZATION_FEATURES.md 更新: Metal Kernel Fusion + 性能基准

## In-Progress
- 无

## Blocked
- Normalization chain (SUM_ROWS→CLAMP→DIV) 融合受 graph scheduler 限制

# 已完成工作摘要 (详见 docs/architecture-design.md)

## 关键成果
- **三层协同**: ContextPilot + ClawGate + LMCache = 23.34x (Phase 1 完成)
- **Hybrid Hashing**: prefix overlap 1.11x → 3-7.8x
- **Skip Logic**: cache_prompt=false 触发 27.93x 加速
- **Approximate Skip**: 95%+ 命中零填充，覆盖率 5%→30%
- **V Tensor 教训**: 不跳过 forward pass 时 V cache = 纯开销 (-8.2%)
- **OpenMP 冲突**: 已解决 (venv + OMP_NUM_THREADS=1)

## 待完成
- ⏳ ClawGate + ThunderLLAMA 完整链路测试 (待配置路由)
- ⏳ Prometheus/Grafana 监控 (见 task list I1/I2)

