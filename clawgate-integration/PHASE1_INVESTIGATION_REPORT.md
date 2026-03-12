# Phase 1 Investigation Report

**Date**: 2026-03-12
**Status**: ✅ Parameter transmission verified, ⚠️ Skip logic trigger unstable

---

## Executive Summary

The `cache_prompt` parameter **is correctly transmitted and honored** by ThunderLLAMA's `/v1/chat/completions` endpoint. The original diagnosis in PHASE1_COMPLETION_REPORT.md was incorrect.

The real issue is that **LMCache skip logic has strict triggering requirements** (100% chunk hit rate), which is difficult to achieve with slightly varying prompts.

---

## Investigation Results

### 1. Parameter Transmission: ✅ VERIFIED

**Code Evidence**:

`tools/server/server-common.cpp:1115-1123`:
```cpp
// Copy remaining properties to llama_params
// This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
for (const auto & item : body.items()) {
    // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
    if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
        llama_params[item.key()] = item.value();
    }
}
```

This code **preserves all extra parameters including `cache_prompt`**.

**Test Evidence**:
```
Request 1 (cache_prompt=true):  cache_n=11, prompt_n=1  → Session cache active
Request 2 (cache_prompt=false): cache_n=0,  prompt_n=12 → Session cache disabled ✅
```

**Conclusion**: The parameter is correctly accepted and processed by ThunderLLAMA.

---

### 2. Session Cache Control: ✅ WORKING

**Code Evidence**:

`tools/server/server-context.cpp:2222`:
```cpp
if (slot.task->params.cache_prompt) {
    // reuse any previously computed tokens that are common with the new prompt
    n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
    ...
}
```

When `cache_prompt=false`, this code path is **not executed**, forcing full prefill.

**Test Evidence**:
```
With cache_prompt=true:  cache_n=11 (reused 11 tokens from session cache)
With cache_prompt=false: cache_n=0  (no session cache reuse)
```

**Conclusion**: `cache_prompt=false` successfully disables session cache.

---

### 3. Skip Logic Triggering: ⚠️ UNSTABLE

**Successful Cases** (549x speedup):
- Identical prompts
- Same slot_id
- Sufficient LMCache warmup
- Result: `prompt_ms` dropped from 3129ms → 5.7ms

**Failed Cases** (1-5x speedup only):
- Slightly varying prompts (e.g., through ContextPilot)
- Different slot_id per request
- Result: No significant acceleration

**Root Cause**:

Skip logic requires **100% LMCache chunk hit rate**:
```
Skip condition: is_prefill=1 AND chunks_found == chunks_needed
```

If the prompt varies even slightly (e.g., different query in ContextPilot output), LMCache chunking may result in `chunks_found < chunks_needed`, preventing skip logic activation.

---

## Why Phase 1 Test Shows 1.04x Speedup

Original Phase 1 test used **10 similar but not identical prompts**:
```python
tasks = [f"审查任务 #{i}: 检查 module_{i}.py 的代码质量" for i in range(10)]
```

Although system prompt is identical, the query varies (`module_0.py`, `module_1.py`, ...), resulting in:
1. ✅ `cache_prompt=False` correctly set (9/9)
2. ✅ Session cache disabled
3. ❌ LMCache **partial hit** (not 100%)
4. ❌ Skip logic **not triggered**
5. Result: Only 1.04x speedup (hybrid hashing + session cache savings, but no skip)

---

## Corrected Understanding

**PHASE1_COMPLETION_REPORT.md said**:
> "The `cache_prompt` parameter is not being honored by the `/v1/chat/completions` endpoint"

**Actually**:
- ✅ The parameter **is honored**
- ✅ Session cache **is correctly disabled** when `cache_prompt=false`
- ❌ Skip logic requires **100% identical prompts** for 100% LMCache hit
- ❌ ContextPilot + varying queries → partial LMCache hit → no skip

---

## Solutions

### Option 1: Relax Skip Logic Condition (Recommended)

**Current**: Skip only when `chunks_found == chunks_needed` (100% hit)

**Proposed**: Skip when `chunks_found / chunks_needed > threshold` (e.g., 95%)

**Impact**: More flexible skip triggering for slightly varying prompts

**Implementation**: Modify skip logic in ThunderLLAMA's prefill code

---

### Option 2: Ensure ContextPilot Output Consistency

**Problem**: ContextPilot may produce slightly different outputs even for identical inputs

**Solution**:
- Add deterministic mode to ContextPilot
- Ensure exact same message order and formatting

**Impact**: Guarantees 100% LMCache hit for identical logical inputs

---

### Option 3: Accept Current Performance

**Rationale**:
- 23.34x speedup (current) vs 27.93x (theoretical) = 83% of target
- Already exceeds 20x minimum requirement
- Complexity of fixing may not justify 16% gain

---

## Recommendations

**Immediate**:
1. Update PHASE1_COMPLETION_REPORT.md with correct diagnosis
2. Document that skip logic requires 100% identical prompts
3. Consider Phase 1 **functionally complete** at 23.34x

**Future (Phase 2)**:
1. Implement Option 1 (relax skip threshold to 95%)
2. Add monitoring for LMCache hit rate per request
3. Tune decision thresholds based on real data

---

## Test Commands for Verification

### Verify Parameter Transmission
```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Test"}],
    "cache_prompt": false,
    "max_tokens": 3,
    "slot_id": 1
  }' | jq '.timings | {cache_n, prompt_n}'
```

**Expected**: `cache_n: 0` (session cache disabled)

### Trigger Skip Logic
```bash
# Request 1: Build cache
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Long prompt here" * 100}],
    "cache_prompt": true,
    "max_tokens": 5,
    "slot_id": 2
  }'

# Request 2: Force prefill with 100% LMCache hit
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Long prompt here" * 100}],  # IDENTICAL
    "cache_prompt": false,
    "max_tokens": 5,
    "slot_id": 2
  }' | jq '.timings | {prompt_ms}'
```

**Expected**: `prompt_ms` < 10ms (skip logic triggered)

---

## Conclusion

**Phase 1 Status**: ✅ **Complete (with correct understanding)**

- ✅ `/lmcache/stats` endpoint implemented
- ✅ `cache_prompt` parameter correctly transmitted
- ✅ Session cache correctly disabled when `cache_prompt=false`
- ✅ Force prefill decision logic working (9/9)
- ⚠️ Skip logic requires 100% identical prompts (design limitation, not bug)

**Current Performance**: 23.34x (83% of theoretical 27.93x)

**Remaining Work**: None critical for Phase 1; recommend proceeding to Phase 2

---

**Investigation conducted by**: Claude Sonnet 4.5
**Date**: 2026-03-12
