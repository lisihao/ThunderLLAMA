# ClawGate KV Cache Strategy API Integration Guide

## Overview

The ThunderLLAMA KV Cache Strategy API provides dynamic control over KV cache quantization levels, allowing ClawGate to optimize memory usage and performance based on real-time workload characteristics.

**Key Capabilities:**
- **Dynamic quantization switching**: Switch between f16 (full precision), q8_0 (8-bit), and q4_0 (4-bit) quantization
- **Strategy injection**: Configure custom decision strategies via HTTP API
- **Automatic adaptation**: Built-in strategies for hands-off optimization
- **Metrics exposure**: Real-time visibility into cache utilization and strategy decisions

**Benefits:**
- **Memory savings**: Up to 72% reduction in KV cache memory (q4_0 vs f16)
- **Maintained quality**: Minimal quality degradation with quantization
- **Flexible control**: From fully automatic to explicit manual control

---

## API Endpoints

### 1. GET /thunder/kv-strategy

Retrieve current strategy configuration and status.

**Request:**
```bash
curl http://localhost:30000/thunder/kv-strategy
```

**Response:**
```json
{
  "strategy": {
    "name": "adaptive",
    "params": {
      "ctx_weight": 0.4,
      "memory_weight": 0.3,
      "load_weight": 0.2,
      "prompt_weight": 0.1,
      "memory_emergency_threshold": 0.85,
      "high_load_threshold": 0.7,
      "medium_load_threshold": 0.4,
      "hysteresis": 0.05
    },
    "version": 1,
    "source": "default"
  },
  "current_level": "f16",
  "metrics": {
    "n_ctx_total": 4096,
    "n_ctx_used": 512,
    "ctx_utilization": 0.125,
    "kv_cache_size": 384000000,
    "kv_cache_max": 1000000000,
    "memory_pressure": 0.384,
    "n_slots_total": 4,
    "n_slots_active": 1,
    "slot_load": 0.25,
    "avg_prompt_length": 128,
    "timestamp_us": 1234567890
  },
  "history": [
    {
      "timestamp_us": 1234567000,
      "decision": {
        "level": "f16",
        "requires_rebuild": false,
        "reason": "adaptive: score=0.25 -> f16",
        "metadata": { "score": 0.25 }
      },
      "metrics": { ... }
    }
  ],
  "available_strategies": ["fixed", "threshold", "adaptive"]
}
```

---

### 2. POST /thunder/kv-strategy

Set a new strategy configuration (ClawGate injection point).

**Request:**
```bash
curl -X POST http://localhost:30000/thunder/kv-strategy \
  -H "Content-Type: application/json" \
  -d '{
    "name": "threshold",
    "params": {
      "thresholds": [
        {"ctx_utilization": 0.3, "level": "f16"},
        {"ctx_utilization": 0.6, "level": "q8_0"},
        {"ctx_utilization": 0.8, "level": "q4_0"}
      ],
      "hysteresis": 0.05
    },
    "version": 1
  }'
```

**Response:**
```json
{
  "strategy": { ... },
  "current_level": "q8_0",
  "metrics": { ... },
  "rebuild_performed": true,
  "rebuild_success": true,
  "decision": {
    "level": "q8_0",
    "requires_rebuild": true,
    "reason": "threshold strategy: ctx_util=0.65 -> q8_0",
    "metadata": { "ctx_utilization": 0.65, "hysteresis_applied": 0.05 }
  }
}
```

**Error responses:**
- `400 Bad Request`: Invalid strategy name or parameters
- `503 Service Unavailable`: Rebuild failed (slots not idle)

---

### 3. GET /thunder/kv-strategy/evaluate

Dry-run evaluation of current strategy (does not apply changes).

**Request:**
```bash
curl http://localhost:30000/thunder/kv-strategy/evaluate
```

**Response:**
```json
{
  "current_level": "f16",
  "metrics": { ... },
  "decision": {
    "level": "q8_0",
    "requires_rebuild": true,
    "reason": "adaptive: score=0.55 -> q8_0",
    "metadata": { "score": 0.55 }
  }
}
```

**Use case:** Preview what decision would be made without actually switching quantization levels.

---

### 4. GET /thunder/kv-strategy/available

List all available strategies and their parameter schemas.

**Request:**
```bash
curl http://localhost:30000/thunder/kv-strategy/available
```

**Response:**
```json
[
  {
    "name": "fixed",
    "params_schema": {
      "level": {
        "type": "string",
        "enum": ["f16", "q8_0", "q4_0"],
        "description": "Target quantization level"
      }
    }
  },
  {
    "name": "threshold",
    "params_schema": {
      "thresholds": {
        "type": "array",
        "description": "Array of {ctx_utilization: number, level: string} objects"
      },
      "hysteresis": {
        "type": "number",
        "default": 0.05,
        "description": "Hysteresis band to prevent oscillation"
      }
    }
  },
  {
    "name": "adaptive",
    "params_schema": {
      "ctx_weight": {"type": "number", "default": 0.4},
      "memory_weight": {"type": "number", "default": 0.3},
      "load_weight": {"type": "number", "default": 0.2},
      "prompt_weight": {"type": "number", "default": 0.1},
      "memory_emergency_threshold": {"type": "number", "default": 0.85},
      "high_load_threshold": {"type": "number", "default": 0.7},
      "medium_load_threshold": {"type": "number", "default": 0.4},
      "hysteresis": {"type": "number", "default": 0.05}
    }
  }
]
```

---

## Strategy Types

### 1. Fixed Strategy

Always use the specified quantization level.

**Parameters:**
```json
{
  "name": "fixed",
  "params": {
    "level": "q8_0"  // "f16", "q8_0", or "q4_0"
  }
}
```

**Use case:** Manual control, testing, or when you know the optimal level for your workload.

---

### 2. Threshold Strategy

Switch quantization based on context utilization thresholds.

**Parameters:**
```json
{
  "name": "threshold",
  "params": {
    "thresholds": [
      {"ctx_utilization": 0.3, "level": "f16"},
      {"ctx_utilization": 0.6, "level": "q8_0"},
      {"ctx_utilization": 0.8, "level": "q4_0"}
    ],
    "hysteresis": 0.05
  }
}
```

**Logic:**
- When `ctx_utilization >= 0.8`: use `q4_0`
- When `ctx_utilization >= 0.6`: use `q8_0`
- When `ctx_utilization >= 0.3`: use `f16`
- Hysteresis prevents oscillation near threshold boundaries

**Use case:** Simple rule-based control based on context usage.

---

### 3. Adaptive Strategy (Default)

Multi-signal intelligent decision making with automatic emergency protection.

**Parameters:**
```json
{
  "name": "adaptive",
  "params": {
    "ctx_weight": 0.4,
    "memory_weight": 0.3,
    "load_weight": 0.2,
    "prompt_weight": 0.1,
    "memory_emergency_threshold": 0.85,
    "high_load_threshold": 0.7,
    "medium_load_threshold": 0.4,
    "hysteresis": 0.05
  }
}
```

**Logic:**
```
score = ctx_weight * ctx_utilization
      + memory_weight * memory_pressure
      + load_weight * (active_slots / total_slots)
      + prompt_weight * (avg_prompt_length / n_ctx_total)

if memory_pressure > 0.85:
    decision = q4_0  (emergency memory protection)
elif score > 0.7:
    decision = q4_0  (high load)
elif score > 0.4:
    decision = q8_0  (medium load)
else:
    decision = f16   (low load)
```

**Features:**
- Cooldown timer (30s minimum between rebuilds)
- Hysteresis to prevent oscillation
- Emergency memory protection

**Use case:** Fully automatic optimization for dynamic workloads.

---

## Integration Patterns

### Pattern 1: Monitor and Override

ClawGate monitors the default adaptive strategy and overrides when needed.

```python
# Monitor current state
response = requests.get('http://localhost:30000/thunder/kv-strategy')
current_level = response.json()['current_level']
metrics = response.json()['metrics']

# Override when conditions are met
if metrics['memory_pressure'] > 0.9:
    requests.post('http://localhost:30000/thunder/kv-strategy', json={
        'name': 'fixed',
        'params': {'level': 'q4_0'}
    })
```

---

### Pattern 2: Configure and Forget

ClawGate sets a custom strategy once and lets ThunderLLAMA manage it autonomously.

```python
# Set custom threshold strategy
requests.post('http://localhost:30000/thunder/kv-strategy', json={
    'name': 'threshold',
    'params': {
        'thresholds': [
            {'ctx_utilization': 0.4, 'level': 'f16'},
            {'ctx_utilization': 0.7, 'level': 'q8_0'},
            {'ctx_utilization': 0.9, 'level': 'q4_0'}
        ],
        'hysteresis': 0.1
    }
})
# No further intervention needed - strategy runs autonomously
```

**Note:** Strategies set via API (`source: "api"`) **do not auto-evaluate** in periodic slots updates. They only evaluate when explicitly triggered via POST or when slots become idle then active again.

---

### Pattern 3: Full Control

ClawGate makes all quantization decisions explicitly.

```python
# Dry-run to preview decision
eval_response = requests.get('http://localhost:30000/thunder/kv-strategy/evaluate')
proposed_level = eval_response.json()['decision']['level']

# Decide whether to apply
if should_apply(proposed_level):
    requests.post('http://localhost:30000/thunder/kv-strategy', json={
        'name': 'fixed',
        'params': {'level': proposed_level}
    })
```

---

## Error Handling

### Common Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| `400` | Invalid request | Check JSON format and required fields |
| `503` | Rebuild unavailable | Wait for slots to become idle |
| `500` | Internal error | Check server logs |

### Rebuild Failure

When rebuild fails (`rebuild_success: false`):
- **Cause**: Active slots prevent context rebuild
- **Resolution**: Retry after current requests complete
- **Prevention**: Monitor slot activity before changing strategy

```python
# Check if rebuild is safe
response = requests.get('http://localhost:30000/thunder/kv-strategy')
if response.json()['metrics']['n_slots_active'] == 0:
    # Safe to rebuild
    requests.post(...)
```

---

## Environment Variables

### `THUNDERLLAMA_KV_STRATEGY`

Set startup strategy via environment variable.

**Example:**
```bash
export THUNDERLLAMA_KV_STRATEGY='{"name":"threshold","params":{"thresholds":[{"ctx_utilization":0.5,"level":"q8_0"}]},"version":1}'

./build/bin/llama-server -m model.gguf -c 4096 -ngl 99 --port 30000
```

**Priority:**
- Environment variable sets initial strategy (`source: "env"`)
- API calls override with `source: "api"`
- Default is `adaptive` strategy (`source: "default"`)

---

## Performance Characteristics

### Memory Savings

| Level | KV Cache Size (4096 ctx, 30B model) | Reduction vs f16 |
|-------|--------------------------------------|------------------|
| f16   | ~384 MiB                             | 0%               |
| q8_0  | ~204 MiB                             | **-46.9%**       |
| q4_0  | ~108 MiB                             | **-71.9%**       |

### Quality Impact

Based on testing with Qwen3-30B:
- **q8_0**: Near-identical quality to f16
- **q4_0**: Surprisingly good quality, suitable for most use cases

### Rebuild Cost

- **Time**: ~5-10 seconds (model dependent)
- **Availability**: Server blocks new requests during rebuild
- **Mitigation**: Cooldown timer prevents frequent rebuilds

---

## Extending

### Custom Strategies (Future)

Currently, strategies are built-in. To add custom strategies:

1. **Via parameter tuning**: Adjust `adaptive` strategy weights
2. **Via plugin (future)**: Register custom evaluator functions via C++ API

**Example custom adaptive tuning:**
```json
{
  "name": "adaptive",
  "params": {
    "ctx_weight": 0.6,      // Prioritize context utilization
    "memory_weight": 0.2,   // De-prioritize memory pressure
    "load_weight": 0.1,
    "prompt_weight": 0.1,
    "high_load_threshold": 0.5  // More aggressive switching
  }
}
```

---

## FAQ

**Q: Can I change strategies while requests are in-flight?**
A: Yes, but rebuild will be deferred until slots are idle. The API returns `rebuild_success: false` in this case.

**Q: How often does the adaptive strategy evaluate?**
A: Every 10 seconds, but only for `default` or `env` strategies. API-set strategies do not auto-evaluate.

**Q: What happens if rebuild fails?**
A: The server remains on the current quantization level. Retry after slots become idle.

**Q: Can I disable automatic evaluation?**
A: Yes, set a `fixed` strategy via API. This disables periodic auto-evaluation.

**Q: Does quantization affect LMCache hit rates?**
A: LMCache uses content-based hashing, which is independent of quantization level. Hit rates should remain similar.

---

## Example: ClawGate Integration Workflow

```python
import requests
import time

BASE_URL = "http://localhost:30000"

def monitor_and_optimize():
    while True:
        # 1. Get current state
        state = requests.get(f"{BASE_URL}/thunder/kv-strategy").json()
        metrics = state['metrics']

        # 2. Make decision based on metrics
        memory_pressure = metrics['memory_pressure']
        ctx_utilization = metrics['ctx_utilization']

        if memory_pressure > 0.9:
            # Emergency: switch to q4_0
            requests.post(f"{BASE_URL}/thunder/kv-strategy", json={
                'name': 'fixed',
                'params': {'level': 'q4_0'}
            })
            print("🚨 Emergency: switched to q4_0")

        elif memory_pressure > 0.7 and ctx_utilization > 0.6:
            # High load: switch to q8_0
            requests.post(f"{BASE_URL}/thunder/kv-strategy", json={
                'name': 'fixed',
                'params': {'level': 'q8_0'}
            })
            print("⚡ High load: switched to q8_0")

        elif memory_pressure < 0.4 and ctx_utilization < 0.3:
            # Low load: use full precision
            requests.post(f"{BASE_URL}/thunder/kv-strategy", json={
                'name': 'fixed',
                'params': {'level': 'f16'}
            })
            print("✨ Low load: switched to f16")

        # 3. Sleep before next check
        time.sleep(30)

if __name__ == "__main__":
    monitor_and_optimize()
```

---

## References

- [KV Cache Quantization Research](https://arxiv.org/abs/2410.19724)
- [ThunderLLAMA Documentation](../README.md)
- [llama.cpp KV Cache Types](https://github.com/ggerganov/llama.cpp/pull/XXXXX)

---

**Version:** 1.0
**Last Updated:** 2026-03-13
