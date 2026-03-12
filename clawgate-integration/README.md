# Clawgate Integration: ContextPilot + ThunderLLAMA + LMCache

> **Complete integration for OpenClaw multi-agent architecture**

## 🎯 Overview

This integration combines four powerful technologies to maximize performance for OpenClaw's multi-agent system:

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenClaw Multi-Agent Layer                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              ContextPilot (Prompt-level Optimization)           │
│  • Context reordering for prefix sharing                        │
│  • Deduplication (~36% token savings)                           │
│  • Cache-aware scheduling                                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Clawgate (Service Layer)                       │
│  • Request orchestration                                        │
│  • Eviction sync (LMCache ← → ContextPilot)                     │
│  • Per-agent statistics                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              ThunderLLAMA (Inference Engine)                    │
│  • Paged Attention for Apple Silicon                            │
│  • LMCache integration                                          │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                  LMCache (Storage-level Optimization)           │
│  • L2 (8GB RAM): Hot KV cache                                   │
│  • L3 (256GB Disk): Persistent cache                            │
│  • Smart prefetch with parallel I/O                             │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install ContextPilot
pip install contextpilot

# 2. Install dependencies
pip install httpx asyncio

# 3. Build ThunderLLAMA with LMCache
cd /Users/lisihao/ThunderLLAMA/build
cmake .. && make llama -j8
```

### Basic Usage

```python
from context_optimizer import ClawgateContextOptimizer
from openclaw_prompt_builder import OpenClawPromptBuilder

# Initialize optimizer
optimizer = ClawgateContextOptimizer(
    thunderllama_url="http://localhost:30000",
    lmcache_enabled=True,
    enable_eviction_sync=True
)

# Build standardized prompt
prompt_builder = OpenClawPromptBuilder()
contexts = prompt_builder.build(
    agent_type="reviewer",
    task="Review authentication module"
)

# Execute with optimization
response = await optimizer.optimize_agent_request(
    agent_type="reviewer",
    contexts=contexts,
    query="Review this code",
    model="default"
)

# Check cache efficiency
optimizer.print_stats()
```

## 📚 Components

### 1. `context_optimizer.py`

**ClawgateContextOptimizer** - Main integration layer

**Features:**
- Online mode: Single agent requests with context tracking
- Offline mode: Batch optimization with scheduling
- Eviction sync: Webhook or polling
- Per-agent statistics

**API:**
```python
# Online mode (multi-turn chat)
response = await optimizer.optimize_agent_request(
    agent_type="reviewer",
    contexts=[...],
    query="Review this",
    model="default"
)

# Offline mode (batch processing)
results = await optimizer.optimize_batch_requests(
    agent_requests=[{
        "agent_type": "reviewer",
        "contexts": [...],
        "query": "..."
    }],
    model="default"
)

# Statistics
stats = optimizer.get_stats("reviewer")
# {'cache_hit_rate': 0.85, 'tokens_saved': 12345, ...}
```

### 2. `openclaw_prompt_builder.py`

**OpenClawPromptBuilder** - Standardized prompt construction

**Why it matters:**
- Ensures common blocks (tools, config) come first
- Maximizes cache hits across agents
- Deterministic ordering for ContextPilot

**Example:**
```python
builder = OpenClawPromptBuilder()

# Single agent
contexts = builder.build(
    agent_type="reviewer",
    task="Review PR #123",
    tool_categories=[ToolCategory.FILE_OPS, ToolCategory.CODE_ANALYSIS]
)

# Batch
batch = builder.build_batch([
    {"agent_type": "reviewer", "task": "Review PR #123"},
    {"agent_type": "architect", "task": "Design system"},
])
```

**Structure:**
```
1. Global Config       (shared by ALL agents)
2. Tool Definitions    (shared by agents with same tools)
3. Agent Role          (unique per agent type)
4. Additional Context  (optional)
5. Current Task        (unique per request)
```

### 3. `thunder-lmcache-eviction.h/cpp`

**ThunderEvictionNotifier** - Eviction sync mechanism

**How it works:**
1. ThunderLLAMA evicts chunks from L3
2. `evict_one_l3()` notifies `ThunderEvictionNotifier`
3. Webhook sends `POST /evict` to ContextPilot
4. ContextPilot updates its context index

**Configuration:**
```cpp
// Set webhook URL (environment variable)
export CONTEXTPILOT_INDEX_URL="http://localhost:8000/evict"

// Or programmatically
thunder_lmcache_set_eviction_webhook("http://localhost:8000/evict");
```

### 4. Test Suite

**`test_integration.py`** - Comprehensive integration tests

**Run tests:**
```bash
python test_integration.py --thunderllama-url http://localhost:30000

# Or specific test
python test_integration.py --test basic
python test_integration.py --test online
python test_integration.py --test batch
```

### 5. Benchmark Suite

**`benchmark_multi_agent.py`** - Performance benchmarks

**Run benchmark:**
```bash
python benchmark_multi_agent.py --thunderllama-url http://localhost:30000
```

**Benchmarks:**
1. Single agent repeated calls (memory effect)
2. Multi-agent shared tools (prefix sharing)
3. Batch scheduling optimization
4. Large-scale orchestration (realistic workload)

## 📊 Performance Expectations

Based on ContextPilot paper and our LMCache implementation:

### Single Agent (Repeated Calls)

| Metric | First Call | Subsequent Calls | Improvement |
|--------|-----------|------------------|-------------|
| Latency | 100 ms | <1 ms | **100x** |
| Cache Hit | 0% | 95%+ | - |
| Tokens Processed | 5000 | 250 | **20x reduction** |

### Multi-Agent (Shared Tools)

| Scenario | Without Optimization | With ContextPilot + LMCache | Improvement |
|----------|---------------------|----------------------------|-------------|
| 10 agents (sequential) | 1000 ms | 200 ms | **5x** |
| 10 agents (parallel) | 100 ms | 30 ms | **3.3x** |
| Cache hit rate | 5% | 60%+ | **12x** |

### Large-Scale (OpenClaw Realistic)

Assuming 10 agents, 5 rounds:

- **Without optimization**: 50 × 100ms = 5000ms
- **With ContextPilot + LMCache**:
  - Round 1: 10 × 100ms = 1000ms (cold)
  - Round 2-5: 40 × 10ms = 400ms (hot)
  - **Total**: 1400ms vs 5000ms = **3.6x speedup**

## 🔧 Configuration

### Environment Variables

```bash
# ThunderLLAMA endpoint
export THUNDERLLAMA_URL="http://localhost:30000"

# ContextPilot server (optional, default = embedded mode)
export CONTEXTPILOT_URL="http://localhost:8000"

# LMCache settings
export THUNDER_LMCACHE_DISK_PATH="/path/to/openclaw_cache.bin"
export LMCACHE_ENABLED="true"

# Eviction sync
export EVICTION_SYNC_ENABLED="true"
export CONTEXTPILOT_INDEX_URL="http://localhost:8000/evict"

# ContextPilot GPU (optional)
export CONTEXTPILOT_GPU="false"
```

### Clawgate Config

```python
# Load from environment
from context_optimizer import create_optimizer_from_env

optimizer = create_optimizer_from_env()

# Or explicit config
optimizer = ClawgateContextOptimizer(
    thunderllama_url="http://localhost:30000",
    contextpilot_url=None,  # Embedded mode
    lmcache_enabled=True,
    enable_eviction_sync=True,
    use_gpu=False
)
```

## 🎯 Best Practices for OpenClaw

### 1. Standardize Prompt Structure

✅ **DO**: Use `OpenClawPromptBuilder` for all agents
```python
builder = OpenClawPromptBuilder()
contexts = builder.build("reviewer", "Review PR #123")
```

❌ **DON'T**: Manually construct prompts
```python
# This breaks cache reuse!
contexts = [
    f"Time: {now()}",  # Different every time!
    "Tools: read_file, write_file",
    "You are a reviewer"
]
```

### 2. Group Agents by Tool Needs

Agents with the same tool categories will share more cache:

```python
# Good: All code analysis agents use same tools
reviewer_contexts = builder.build("reviewer", task,
    tool_categories=[FILE_OPS, CODE_ANALYSIS])

architect_contexts = builder.build("architect", task,
    tool_categories=[FILE_OPS, CODE_ANALYSIS])  # Same!

coder_contexts = builder.build("coder", task,
    tool_categories=[FILE_OPS, CODE_ANALYSIS, TESTING])
```

### 3. Use Batch Mode for Parallel Agents

When running multiple agents in parallel:

```python
# ✅ Good: Batch optimization
results = await optimizer.optimize_batch_requests(agent_requests)

# ❌ Bad: Sequential without optimization
for req in agent_requests:
    await optimizer.optimize_agent_request(...)
```

### 4. Monitor Cache Hit Rates

Regularly check cache efficiency:

```python
# Print stats
optimizer.print_stats()

# Get specific agent stats
reviewer_stats = optimizer.get_stats("reviewer")
if reviewer_stats['cache_hit_rate'] < 0.5:
    print("⚠️ Low cache hit rate for reviewer!")
```

### 5. Warm Up Cache on Startup

Pre-warm cache with common prompts:

```python
async def warmup_openclaw_cache():
    """Pre-warm cache with common agent prompts"""
    builder = OpenClawPromptBuilder()

    for agent_type in ["reviewer", "architect", "coder", "tester"]:
        contexts = builder.build(agent_type, "Warmup task")
        await optimizer.optimize_agent_request(
            agent_type=agent_type,
            contexts=contexts,
            query="Warmup",
            model="default"
        )

    print("✓ Cache warmed up")
```

## 🐛 Troubleshooting

### Issue 1: Low Cache Hit Rate

**Symptoms**: Cache hit rate < 30%

**Causes:**
- Prompts not standardized (use `OpenClawPromptBuilder`)
- Tool categories inconsistent
- ContextPilot not optimizing (check logs)

**Solution:**
```python
# Check prompt consistency
builder = OpenClawPromptBuilder()
contexts_1 = builder.build("reviewer", "Task 1")
contexts_2 = builder.build("reviewer", "Task 2")

# First 2 blocks should be identical
assert contexts_1[0] == contexts_2[0]  # Global config
assert contexts_1[1] == contexts_2[1]  # Tools
```

### Issue 2: Eviction Sync Not Working

**Symptoms**: ContextPilot index stale, performance degrades over time

**Causes:**
- Webhook URL not configured
- ThunderLLAMA not calling eviction callback

**Solution:**
```bash
# 1. Check webhook URL
echo $CONTEXTPILOT_INDEX_URL

# 2. Check ThunderLLAMA logs for eviction notifications
# Should see: [ThunderEvictionNotifier] Notifying...

# 3. Verify ContextPilot receives evictions
# Check ContextPilot logs
```

### Issue 3: High Latency Despite Cache

**Symptoms**: Cache hit rate high (>80%) but latency still slow

**Causes:**
- LMCache L2 full, hitting L3 (disk)
- Network latency to ThunderLLAMA

**Solution:**
```bash
# Check LMCache stats
thunder-cache stats

# If L2 usage > 90%, increase L2 size:
# Modify ThunderChunkStorage constructor:
#   cpu_limit_bytes = 16ULL * 1024 * 1024 * 1024  // 16GB instead of 8GB
```

## 📖 Additional Documentation

- **LMCache Features**: See `/Users/lisihao/ThunderLLAMA/LMCACHE_FEATURES.md`
- **ContextPilot Paper**: https://arxiv.org/abs/2511.03475
- **ThunderLLAMA README**: `/Users/lisihao/ThunderLLAMA/README.md`

## 🎓 Example: Complete OpenClaw Integration

```python
#!/usr/bin/env python3
"""
Example: Complete OpenClaw integration
"""

import asyncio
from context_optimizer import ClawgateContextOptimizer
from openclaw_prompt_builder import OpenClawPromptBuilder, ToolCategory


class OpenClawOrchestrator:
    def __init__(self):
        self.optimizer = ClawgateContextOptimizer(
            thunderllama_url="http://localhost:30000",
            lmcache_enabled=True,
            enable_eviction_sync=True
        )
        self.prompt_builder = OpenClawPromptBuilder()

    async def execute_review_workflow(self, pr_number: int):
        """
        Execute a code review workflow with multiple agents
        """
        # Define agents and tasks
        agents = [
            ("reviewer", f"Review PR #{pr_number} for bugs"),
            ("security_checker", f"Check PR #{pr_number} for vulnerabilities"),
            ("test_verifier", f"Verify tests for PR #{pr_number}"),
        ]

        # Build batch request
        agent_requests = []
        for agent_type, task in agents:
            contexts = self.prompt_builder.build(
                agent_type=agent_type,
                task=task,
                tool_categories=[
                    ToolCategory.FILE_OPS,
                    ToolCategory.CODE_ANALYSIS
                ]
            )
            agent_requests.append({
                "agent_type": agent_type,
                "contexts": contexts,
                "query": task
            })

        # Execute batch
        print(f"🚀 Executing review workflow for PR #{pr_number}...")
        results = await self.optimizer.optimize_batch_requests(
            agent_requests,
            model="default"
        )

        # Process results
        for (agent_type, task), result in zip(agents, results):
            print(f"\n✓ {agent_type}:")
            print(f"  Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')[:100]}...")
            print(f"  Cache hit: {result.get('cache_hit', False)}")

        # Print stats
        self.optimizer.print_stats()

    async def close(self):
        await self.optimizer.close()


async def main():
    orchestrator = OpenClawOrchestrator()

    try:
        # Run multiple PR reviews (cache should help)
        for pr_num in [123, 124, 125]:
            await orchestrator.execute_review_workflow(pr_num)
            print("\n" + "="*70 + "\n")

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## 🤝 Contributing

This integration is designed specifically for OpenClaw. If you have improvements or find issues, please contact the maintainer.

## 📄 License

Follows ThunderLLAMA project license.

---

**🎉 Ready for Production!**

Start using: `python test_integration.py --test all`
