# LMCache Performance Benchmark Report

> **Date**: 2026-03-12
> **Model**: Qwen3-30B-A3B-128K-Q5_K_M (30B parameters, Q5_K_M quantization)
> **Hardware**: Apple M4 Max (40GB RAM, Metal GPU)

---

## 📊 Test Results Summary

### Configuration

| Config | LMCACHE_ENABLED | GPU Layers | Context Size | Batch Size |
|--------|----------------|------------|--------------|------------|
| **Test 1** | `false` | 99 | 8192 | 512 (default) |
| **Test 2** | `true` | 20 | 4096 | 256 |

**Note**: GPU layers reduced in Test 2 due to memory constraints (30B model OOM with 99 layers)

---

## 🚀 Performance Comparison

### Test 1: Repeated Identical Prompts
Simulates single agent asking the same question multiple times (memory effect)

| Metric | No LMCache | With LMCache | Difference |
|--------|-----------|--------------|------------|
| **First Request (Cold)** | 1215 ms | 1117 ms | **-8.1%** (faster) |
| **Average Warm** | 804 ms | 813 ms | **+1.1%** (slower) |
| **Speedup (2nd vs 1st)** | 1.5x | 1.4x | -0.1x |
| **Total Time (5 requests)** | 4.4 s | 4.4 s | Same |

### Test 2: Shared Prefix Prompts
Simulates multi-agent scenario with shared system prompts

| Metric | No LMCache | With LMCache | Difference |
|--------|-----------|--------------|------------|
| **First Request (Cold)** | 924 ms | 948 ms | **+2.6%** (slower) |
| **Average Warm** | 885 ms | 875 ms | **-1.1%** (faster) |
| **Speedup (2nd vs 1st)** | 1.0x | 1.1x | +0.1x |
| **Total Time (5 requests)** | 4.5 s | 4.4 s | **-2.2%** (faster) |

---

## 🔍 Analysis

### Key Findings

1. **Minimal Performance Difference**
   - No significant speedup observed with LMCache enabled
   - Results are within margin of error (~1-2%)

2. **Possible Explanations**
   - **LMCache Not Enabled**: No LMCache-specific logs in server output
   - **Configuration Issue**: Environment variable may not be read correctly
   - **Compilation Issue**: LMCache code may not be compiled into binary
   - **Test Scenario Too Simple**: Short prompts may not trigger L3 cache benefits

3. **GPU Memory Constraints**
   - 30B model requires reduced GPU layers (99 → 20) when LMCache enabled
   - This configuration change may offset LMCache benefits
   - OOM error: `kIOGPUCommandBufferCallbackErrorOutOfMemory`

### Expected vs Actual Performance

According to ContextPilot paper and LMCache design:

| Scenario | Expected Speedup | Actual Speedup |
|----------|-----------------|----------------|
| Repeated prompts | **10-100x** (warm cache) | **1.4-1.5x** |
| Shared prefix | **3-5x** (prefix reuse) | **1.0-1.1x** |

**Conclusion**: LMCache is likely **NOT** functioning as expected.

---

## ⚠️ Issues Discovered

### Issue 1: LMCache Not Logging
**Symptom**: No LMCache-related logs in server output
**Expected**: Should see logs like:
```
[ThunderChunkStorage] Initialized: L2=8GB, L3=256GB
[ThunderChunkStorage] Cache hit on content_hash=...
[ThunderChunkStorage] Evicting chunk from L3
```

**Actual**: Only standard llama.cpp logs

### Issue 2: GPU Memory Insufficient (30B Model)
**Error**:
```
ggml_metal_synchronize: error: command buffer 0 failed with status 5
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

**Impact**: Had to reduce GPU layers, affecting performance baseline

### Issue 3: Environment Variable Not Effective
**Set**: `LMCACHE_ENABLED=true`
**Effect**: No visible change in behavior

---

## 🛠️ Recommended Next Steps

### 1. Verify LMCache Compilation

Check if LMCache code was compiled into `llama-server`:

```bash
cd /Users/lisihao/ThunderLLAMA/build
nm bin/llama-server | grep -i "thunder.*cache\|lmcache"
```

If no symbols found, **rebuild with LMCache**:

```bash
cd /Users/lisihao/ThunderLLAMA
rm -rf build
mkdir build && cd build
cmake .. -DGGML_METAL=ON -DTHUNDER_LMCACHE=ON  # Enable LMCache
make llama-server -j8
```

### 2. Add LMCache Logging

Verify LMCache initialization in code:

```cpp
// In llama.cpp or llama-server
if (std::getenv("LMCACHE_ENABLED") &&
    std::string(std::getenv("LMCACHE_ENABLED")) == "true") {
    fprintf(stderr, "[LMCache] Enabled: L2=%zu, L3=%s\n", l2_size, l3_path);
} else {
    fprintf(stderr, "[LMCache] Disabled\n");
}
```

### 3. Test with Smaller Model

Avoid GPU memory constraints:

```bash
# Use 7B-14B model for testing
./bin/llama-server \
  --model Qwen-7B-Q5_K_M.gguf \
  --n-gpu-layers 99 \
  --ctx-size 8192
```

### 4. Longer Prompt Test

Current test uses very short prompts (15-21 tokens). LMCache benefits increase with:
- Longer prompts (1000+ tokens)
- More repetitions (10+ requests)
- Larger prefix overlap

**Suggested test prompt**:
```python
# 2000+ token shared prefix (tool definitions + system prompt)
long_prefix = """
You are a helpful AI assistant with access to the following tools:

[... 1500 tokens of tool definitions ...]

Current task: {task}
"""
```

---

## 📈 Expected Performance (When LMCache Works)

Based on LMCache design and ContextPilot paper:

### Scenario: Multi-Agent Workflow (10 agents, 5 rounds)

| Round | Without LMCache | With LMCache | Speedup |
|-------|----------------|--------------|---------|
| Round 1 (cold) | 1000 ms | 1000 ms | 1x |
| Round 2 (warm) | 1000 ms | **100 ms** | **10x** |
| Round 3 (warm) | 1000 ms | **100 ms** | **10x** |
| Round 4 (warm) | 1000 ms | **100 ms** | **10x** |
| Round 5 (warm) | 1000 ms | **100 ms** | **10x** |
| **Total** | **5000 ms** | **1400 ms** | **3.6x** |

### Benefits:
- **Cache hit rate**: 85-95% (with standardized prompts)
- **Token savings**: 60-70% (shared tool definitions)
- **Latency reduction**: 10-100x (warm cache, L2 hit)
- **Disk persistence**: Cache survives server restarts

---

## 🎯 Conclusion

### Current Status: ⚠️ LMCache Not Verified

The performance benchmark shows **no significant improvement** with LMCache enabled, suggesting:

1. **LMCache is not functioning**: No logs, no speedup
2. **Compilation issue**: Code may not be compiled into binary
3. **Configuration issue**: Environment variable not effective

### Action Required:

1. ✅ Verify LMCache compilation (`nm` symbol check)
2. ✅ Rebuild with `-DTHUNDER_LMCACHE=ON` if needed
3. ✅ Add debug logging to confirm initialization
4. ✅ Re-run benchmark with verified build
5. ✅ Test with smaller model (7B-14B) to avoid GPU OOM

### Expected Outcome (When Fixed):

- **10-100x speedup** on repeated prompts (warm cache)
- **3-5x speedup** on multi-agent workflows
- **LMCache logs** showing cache hits/misses

---

## 📝 Test Details

### Test Environment

```
Model: Qwen3-30B-A3B-128K-Q5_K_M.gguf
Size: ~20GB
Quantization: Q5_K_M
Hardware: Apple M4 Max
GPU: MTL0 (MTLGPUFamilyApple9)
RAM: 40GB
```

### Test Prompts

**Test 1 (Identical)**:
```
"Explain how photosynthesis works." × 5
```

**Test 2 (Shared Prefix)**:
```
"You are a helpful AI.\n\nQuestion: What is machine learning?"
"You are a helpful AI.\n\nQuestion: What is deep learning?"
"You are a helpful AI.\n\nQuestion: What is neural network?"
"You are a helpful AI.\n\nQuestion: What is computer vision?"
"You are a helpful AI.\n\nQuestion: What is NLP?"
```

### Measurement

- **Latency**: End-to-end request time (HTTP POST to response)
- **Tokens**: Prompt tokens from `/v1/chat/completions` response
- **Speedup**: First request latency / warm request latency

---

*Report generated: 2026-03-12*
*Next: Verify compilation and re-test*
