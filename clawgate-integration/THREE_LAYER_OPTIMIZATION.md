# 三层协同优化实现指南

> **状态**: Phase 1 准备中
> **最后更新**: 2026-03-12
> **目标**: 37x 综合加速（Multi-agent 场景）

## 📋 快速恢复指南

如果会话崩溃，按以下步骤恢复：

### 1. 读取状态文件
```bash
cat ~/.openclaw_backup_20260303_060004/agents/main/sessions/.solar/STATE.md
```

### 2. 检查当前分支
```bash
cd /Users/lisihao/ThunderLLAMA
git branch
# 应该在: merge-mac-mini-and-laptop (Hybrid hashing)
```

### 3. 验证 Skip Logic
```bash
python3 /tmp/test_skip_logic.py
# 预期: 27.93x 加速，6 次跳过
```

### 4. 继续 Phase 1
从本文档的 "Phase 1 实现步骤" 部分继续

---

## 🎯 架构概览

### 三层协同

```
┌─────────────────────────────────────────────────────────────────┐
│  L1: ContextPilot                                               │
│  功能: Reorder + Dedup                                          │
│  收益: 36% token 节省                                           │
│  状态: ✅ 已部署（ClawGate 集成）                               │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  L2: ClawGate **← Phase 1 实现目标**                            │
│  新增功能:                                                      │
│  - should_force_prefill() 决策逻辑                              │
│  - cache_prompt 参数透传                                        │
│  - ContextPilot overlap API 集成                                │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  L3: ThunderLLAMA + LMCache                                     │
│  功能:                                                          │
│  - Hybrid hashing (3x for 50% prefix overlap)                  │
│  - Skip logic (27x when triggered)                             │
│  状态: ✅ 已实现并验证                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 性能数据（已验证）

| 测试 | 方法 | 加速比 | 状态 |
|------|------|--------|------|
| K-only | Position-based hashing | 25.59x | ✅ Baseline |
| Hybrid hashing | Content-based (prefill) | 25.59x (100%), 3.01x (50%) | ✅ 已提交 |
| V tensor | K+V 缓存 | 23.50x | ❌ 负优化 (-8.2%) |
| Skip logic | cache_prompt=false | **27.93x** | ✅ 已验证 |

---

## 🔧 Phase 1: ClawGate 增强

### 目标

实现缓存感知路由，自动决策 `cache_prompt` 参数以触发 skip logic。

### 核心决策逻辑

```python
def should_force_prefill(
    request: AgentRequest,
    context_pilot_stats: ContextPilotStats,
    lmcache_stats: LMCacheStats
) -> tuple[bool, str]:
    """
    决策是否应该强制 prefill（禁用 session cache）

    Args:
        request: 当前请求
        context_pilot_stats: ContextPilot 统计信息
        lmcache_stats: LMCache 统计信息

    Returns:
        (should_force, reason)
        - should_force: True → cache_prompt=false
        - reason: 决策原因（用于日志）
    """

    # 1. 计算 prefix overlap
    overlap = context_pilot_stats.get_prefix_overlap(
        request.prompt,
        window_size=10  # 最近 10 个请求
    )

    # 2. 查询 LMCache 命中率
    cache_hit_rate = lmcache_stats.get_estimated_hit_rate(
        request.prompt_hash
    )

    # 决策规则
    if overlap > 0.8 and cache_hit_rate > 0.9:
        # 高 overlap + 高命中率 → 强制 prefill，触发 skip
        return True, f"high_overlap={overlap:.2f},hit_rate={cache_hit_rate:.2f}"

    elif overlap < 0.3:
        # 低 overlap → 允许 session cache
        return False, f"low_overlap={overlap:.2f}"

    else:
        # 中等 overlap → 使用 hybrid hashing
        return False, f"medium_overlap={overlap:.2f}"
```

### 实现步骤

#### Step 1: 扩展 ContextPilot Stats API

**文件**: `context_optimizer.py` (已存在)

**新增方法**:
```python
class ClawgateContextOptimizer:
    def get_prefix_overlap(
        self,
        prompt: str,
        window_size: int = 10
    ) -> float:
        """
        计算当前 prompt 与最近 N 个请求的 prefix overlap

        Returns:
            float: 平均 overlap 比例 (0.0 - 1.0)
        """
        # 使用 ContextPilot 的 embedding 计算相似度
        recent_prompts = self.recent_requests[-window_size:]
        if not recent_prompts:
            return 0.0

        overlaps = []
        for prev_prompt in recent_prompts:
            # 计算 token-level prefix overlap
            overlap = self._calculate_prefix_overlap(prompt, prev_prompt)
            overlaps.append(overlap)

        return sum(overlaps) / len(overlaps)

    def _calculate_prefix_overlap(
        self,
        prompt1: str,
        prompt2: str
    ) -> float:
        """
        计算两个 prompt 的 prefix overlap 比例

        Algorithm:
        1. Tokenize both prompts
        2. Find longest common prefix
        3. Return overlap ratio
        """
        tokens1 = self._tokenize(prompt1)
        tokens2 = self._tokenize(prompt2)

        # Find longest common prefix
        common_len = 0
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2:
                common_len += 1
            else:
                break

        # Return overlap ratio (relative to shorter prompt)
        min_len = min(len(tokens1), len(tokens2))
        return common_len / min_len if min_len > 0 else 0.0
```

#### Step 2: 添加 LMCache Stats Client

**新文件**: `lmcache_stats_client.py`

```python
import httpx
from typing import Optional

class LMCacheStatsClient:
    """
    LMCache 统计信息客户端

    通过 ThunderLLAMA 的 /stats 端点查询缓存命中率
    """

    def __init__(self, base_url: str = "http://localhost:30000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=5.0)

    async def get_estimated_hit_rate(
        self,
        prompt_hash: Optional[int] = None
    ) -> float:
        """
        估算 prompt 的 LMCache 命中率

        Args:
            prompt_hash: Prompt 的 hash (可选)
                        如果提供，返回该 prompt 的命中率
                        否则返回全局命中率

        Returns:
            float: 命中率 (0.0 - 1.0)
        """
        try:
            # TODO: 实现 /lmcache/stats 端点
            # 目前使用全局命中率估算
            resp = await self.client.get(f"{self.base_url}/slots")
            slots = resp.json()

            # 简单估算: 如果有活跃 slot，假设有缓存
            active_slots = [s for s in slots if s.get("is_processing")]
            if active_slots:
                return 0.9  # 假设高命中率
            else:
                return 0.5  # 假设中等命中率

        except Exception as e:
            # 降级: 无法查询时返回中等命中率
            return 0.5

    async def close(self):
        await self.client.aclose()
```

#### Step 3: 集成到 ClawGate Request Handler

**文件**: `context_optimizer.py`

**修改**: `optimize_agent_request()` 方法

```python
class ClawgateContextOptimizer:
    def __init__(self, ...):
        # ... 现有初始化 ...
        self.lmcache_client = LMCacheStatsClient()

    async def optimize_agent_request(
        self,
        agent_type: str,
        contexts: List[str],
        query: str,
        model: str,
        **kwargs
    ) -> AgentResponse:
        """
        优化并执行 agent 请求（新增 cache_prompt 决策）
        """

        # 1. ContextPilot 优化（现有）
        optimized_prompt = self.pilot.optimize(contexts, query)

        # 2. **新增**: 决策 cache_prompt
        overlap = self.get_prefix_overlap(optimized_prompt)
        cache_hit_rate = await self.lmcache_client.get_estimated_hit_rate()

        should_force, reason = self._should_force_prefill(
            overlap, cache_hit_rate
        )

        # 3. 调用 ThunderLLAMA（透传 cache_prompt）
        response = await self._call_thunderllama(
            prompt=optimized_prompt,
            cache_prompt=not should_force,  # False → 强制 prefill
            **kwargs
        )

        # 4. 记录统计
        self.stats.record_request(
            agent_type=agent_type,
            overlap=overlap,
            cache_hit_rate=cache_hit_rate,
            cache_prompt_used=not should_force,
            skip_triggered=response.get("skip_count", 0) > 0
        )

        return response

    def _should_force_prefill(
        self,
        overlap: float,
        cache_hit_rate: float
    ) -> tuple[bool, str]:
        """决策逻辑（见上文）"""
        if overlap > 0.8 and cache_hit_rate > 0.9:
            return True, f"high_overlap={overlap:.2f},hit={cache_hit_rate:.2f}"
        elif overlap < 0.3:
            return False, f"low_overlap={overlap:.2f}"
        else:
            return False, f"medium_overlap={overlap:.2f}"
```

#### Step 4: 修改 ThunderLLAMA 调用

**文件**: `context_optimizer.py`

**方法**: `_call_thunderllama()`

```python
async def _call_thunderllama(
    self,
    prompt: str,
    cache_prompt: bool = True,  # 新增参数
    **kwargs
) -> dict:
    """
    调用 ThunderLLAMA API

    Args:
        prompt: 优化后的 prompt
        cache_prompt: 是否允许 session cache
                      True → 允许（默认）
                      False → 禁用，触发 full prefill
    """

    payload = {
        "prompt": prompt,
        "cache_prompt": cache_prompt,  # 透传到 llama-server
        **kwargs
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{self.thunderllama_url}/completion",
            json=payload,
            timeout=120.0
        )
        resp.raise_for_status()
        return resp.json()
```

### 测试验证

**测试脚本**: `/tmp/test_phase1_integration.py`

```python
#!/usr/bin/env python3
"""Phase 1 集成测试"""

import asyncio
from context_optimizer import ClawgateContextOptimizer

async def test_cache_aware_routing():
    optimizer = ClawgateContextOptimizer(
        thunderllama_url="http://localhost:30000",
        lmcache_enabled=True
    )

    # 场景 1: 10 个相似 agents（>80% overlap）
    common_system = "你是一个专业的代码审查专家。" * 100
    tasks = [f"审查任务 {i}" for i in range(10)]

    results = []
    for i, task in enumerate(tasks):
        result = await optimizer.optimize_agent_request(
            agent_type="reviewer",
            contexts=[common_system],
            query=task,
            model="default"
        )
        results.append(result)

        print(f"Agent {i}: cache_prompt={result.cache_prompt_used}, "
              f"skip={result.skip_triggered}, latency={result.latency:.2f}ms")

    # 验证
    assert results[0].cache_prompt_used == True   # Round 1: 允许 cache
    assert all(not r.cache_prompt_used for r in results[1:])  # Round 2+: 强制 prefill
    assert all(r.skip_triggered for r in results[1:])  # Round 2+: skip

    # 性能验证
    avg_latency = sum(r.latency for r in results[1:]) / len(results[1:])
    speedup = results[0].latency / avg_latency

    print(f"\n✅ 测试通过")
    print(f"Round 1: {results[0].latency:.2f}ms")
    print(f"Round 2+ avg: {avg_latency:.2f}ms")
    print(f"加速比: {speedup:.2f}x (预期 > 20x)")

    assert speedup > 20, f"Expected 20x+, got {speedup:.2f}x"

if __name__ == "__main__":
    asyncio.run(test_cache_aware_routing())
```

---

## 📊 Phase 2: 监控与调优

### 监控指标

| 指标 | 来源 | 用途 |
|------|------|------|
| `skip_rate` | ThunderLLAMA logs | 跳过比例（目标 > 80%） |
| `cache_hit_rate` | LMCache stats | 缓存命中率 |
| `overlap_distribution` | ContextPilot | Overlap 分布 |
| `latency_p50/p99` | ClawGate | 延迟分布 |

### Dashboard 设计

```python
# Prometheus metrics
skip_rate = Gauge("lmcache_skip_rate", "Forward pass skip rate")
cache_hit_rate = Gauge("lmcache_hit_rate", "Cache hit rate")
overlap_avg = Histogram("context_overlap", "Prefix overlap distribution")
```

---

## 🚀 Phase 3: 高级优化

### Eviction-Aware Scheduling

**目标**: LMCache 驱逐时，重新排序请求队列

**实现**:
```python
@app.post("/eviction_webhook")
async def on_lmcache_eviction(evicted_hashes: List[int]):
    # 通知 ContextPilot
    context_pilot.invalidate_cache(evicted_hashes)

    # 重新排序
    queue.reorder_by_cache_warmth()
```

### Prefix-Group Batching

**目标**: 同组请求批量处理

**实现**:
```python
groups = context_pilot.group_by_prefix(requests, threshold=0.8)
for group in groups:
    # 第一个: cache_prompt=true
    process(group[0], cache_prompt=True)

    # 后续: cache_prompt=false
    for req in group[1:]:
        process(req, cache_prompt=False)
```

---

## 📝 提交清单

在开始 Phase 1 前，确保以下文件已保存：

- [x] STATE.md - 完整状态
- [x] THREE_LAYER_OPTIMIZATION.md - 本文档
- [x] /tmp/test_skip_logic.py - Skip logic 验证
- [x] /tmp/三层协同优化方案.md - 方案文档
- [ ] Git commit - 保存到版本库

---

## 🔄 故障恢复流程

### 如果 Claude 崩溃

1. **读取状态**:
   ```bash
   cat ~/.openclaw_backup_20260303_060004/agents/main/sessions/.solar/STATE.md
   ```

2. **检查 git 分支**:
   ```bash
   cd /Users/lisihao/ThunderLLAMA
   git branch  # 应该在 merge-mac-mini-and-laptop
   git log --oneline -5
   ```

3. **读取本文档**:
   ```bash
   cat /Users/lisihao/ThunderLLAMA/clawgate-integration/THREE_LAYER_OPTIMIZATION.md
   ```

4. **继续实现**: 从 "Phase 1 实现步骤" 继续

### 如果测试失败

1. **验证 Skip Logic 仍然有效**:
   ```bash
   python3 /tmp/test_skip_logic.py
   # 预期: 27.93x
   ```

2. **检查 ThunderLLAMA 日志**:
   ```bash
   tail -100 /tmp/llama_skip_test.log | grep "SKIPPING"
   ```

3. **回滚到已知状态**:
   ```bash
   git checkout merge-mac-mini-and-laptop
   git log --oneline -3
   # 应该看到: 6a1e8b8df feat(lmcache): implement hybrid hashing
   ```

---

**准备就绪，可以开始 Phase 1 实现。**
