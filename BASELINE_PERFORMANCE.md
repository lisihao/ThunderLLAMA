# ThunderLLAMA Performance Baseline Report

**测试时间:** 2026-03-12
**Commit:** 085c68a66 (feat: enable Paged Attention for Apple Silicon)
**配置:** Flash Attention ON, Paged Attention OFF
**硬件:** Apple M4 Pro
**模型:** Qwen3-0.6B-Q4_0

---

## 测试配置

- **Flash Attention:** ✅ Enabled
- **Paged Attention:** ❌ Disabled (use_paged_attention=0)
- **Context Size:** 40960
- **Offload Layers:** 99 (全GPU)
- **Cache Type:** f16

---

## 性能基线数据

### 1. Short Prompt (10 tokens → 50 tokens)

| 指标 | 数值 |
|------|------|
| **平均延迟** | 179ms |
| **P50延迟** | 170ms |
| **P95延迟** | 372ms |
| **P99延迟** | 372ms |
| **Token生成速度** | 28.2 tokens/s |
| **QPS (2并发)** | 4.95 req/s |
| **吞吐量 (2并发)** | 24.8 tokens/s |

**分析:** 短prompt性能稳定，P99延迟较高可能是首次推理的缓存预热。

---

### 2. Medium Prompt (50 tokens → 50 tokens)

| 指标 | 数值 |
|------|------|
| **平均延迟** | 351ms |
| **P50延迟** | 419ms |
| **P95延迟** | 536ms |
| **P99延迟** | 536ms |
| **Token生成速度** | 117.3 tokens/s |
| **QPS (2并发)** | 2.62 req/s |
| **吞吐量 (2并发)** | 102.1 tokens/s |

**分析:** 中等prompt，延迟增加但token生成速度提升显著 (28→117 tokens/s)。

---

### 3. Long Prompt (200 tokens → 50 tokens)

| 指标 | 数值 |
|------|------|
| **平均延迟** | 354ms |
| **P50延迟** | 508ms |
| **P95延迟** | 571ms |
| **P99延迟** | 571ms |
| **Token生成速度** | 647.3 tokens/s |
| **QPS (2并发)** | 4.30 req/s |
| **吞吐量 (2并发)** | 968.2 tokens/s |

**分析:** 长prompt下Flash Attention优势明显，吞吐量接近1000 tokens/s。

---

## 关键发现

### ✅ 优势

1. **Flash Attention有效:** 长prompt (200 tokens) 吞吐量达到 968 tokens/s
2. **延迟稳定:** P50延迟在 170-508ms 范围内
3. **GPU利用率高:** 全GPU offload，Metal优化生效

### ⚠️ 待优化点

1. **P99延迟波动:** Short prompt P99达到372ms，是P50的2.2倍
2. **并发QPS偏低:** 2并发下QPS只有2.6-5.0 req/s
3. **Paged Attention未启用:** 当前baseline不包含paged优化

---

## 对比目标 (未来测试)

| 优化方案 | 预期提升 | 测试状态 |
|----------|---------|----------|
| Paged Attention | 减少内存拷贝，提升吞吐10-20% | ⏳ 待修复 |
| Prefix Caching | 减少重复计算，提升50%+ | ⏳ 待测试 |
| Cache-RAM Auto Tuning | 优化cache大小，减少P99波动 | ⏳ 待集成 |

---

## 下一步行动

1. **修复Paged Attention bug** → 建立paged基线
2. **对比Paged vs Non-Paged** → 量化性能提升
3. **集成Context Shift** → 测试长上下文性能
4. **压力测试** → 高并发场景 (10+并发)

---

**基线已确立，可用于后续所有性能对比！**
