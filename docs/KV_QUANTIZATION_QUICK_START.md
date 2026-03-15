# KV Cache Quantization - Quick Start Guide

> **TL;DR**: 8.47 GB/s GPU-side quantization using CPU-GPU pipeline on Apple Silicon

---

## Quick Start

### 1. Include Header

```cpp
#include "ggml-metal-device.h"
```

### 2. Basic Usage

```cpp
// Initialize Metal device
ggml_metal_device_t dev = ggml_metal_device_get(0);

// Prepare data
int n_layers = 32;
int elements_per_layer = 5120 * 256 * 2;  // hidden × seq × (K+V)
int group_size = 64;

ggml_fp16_t * src = ...;    // Input: FP16 KV cache
int8_t * dst = ...;         // Output: INT8 quantized
ggml_fp16_t * scales = ...; // Output: FP16 scales

// Quantize (Pipeline - Fastest!)
ggml_metal_quantize_kv_cache_q8_pipeline_cpu(
    dev, src, dst, scales,
    n_layers, elements_per_layer, group_size
);
```

### 3. Run Tests

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build -j

# Test
DYLD_LIBRARY_PATH=./build/bin ./build/bin/test-kv-quantize-pipeline

# Expected: 18.9 ms (8.47 GB/s) ✅
```

---

## Performance Comparison

| Method | API | Speed | Use Case |
|--------|-----|-------|----------|
| **Pipeline** | `..._pipeline_cpu()` | 8.47 GB/s | ✅ Best performance |
| Offline | `..._offline_cpu()` | 5.30 GB/s | One-time quantization |
| Batch | `..._batch_cpu()` | 4.85 GB/s | All layers at once |
| Per-layer | `..._q8_cpu()` | 4.14 GB/s | Layer-by-layer |

---

## Memory Savings

```
Original (FP16):  160 MB
Quantized (INT8): 82.5 MB
Savings:          1.9x (48% reduction)
```

---

## Accuracy

```
Max Error: 0.023 (< 0.05 threshold) ✅
Avg Error: 0.005 (< 0.01 threshold) ✅
```

---

## Configuration

Enable in `thunderllama.conf`:

```bash
# KV Cache Quantization
KV_CACHE_LEVEL="q8_0"  # Options: none, q8_0, q4_0
```

---

## Architecture

```
CPU (NEON)              GPU (Metal)
┌──────────┐            ┌──────────┐
│ Find Max │──scales──→ │ Quantize │
│ (16 ms)  │            │ (16 ms)  │
└──────────┘            └──────────┘
     ↓ overlap                ↓
Total Time: max(16, 16) + 2ms = 18.9ms
```

**Key Features**:
- ✅ **Zero-Copy** (UMA shared memory)
- ✅ **CPU-GPU Pipeline** (parallel execution)
- ✅ **NEON Acceleration** (8×FP16 SIMD)

---

## Advanced Options

### Offline Quantization (Model Load)

```cpp
// Quantize once at model load
ggml_metal_kv_cache_quantize_offline_cpu(
    dev, kv_cache,
    n_layers, hidden_dim, seq_len, group_size
);
// Later: only dequantization needed (faster inference)
```

### Batch Quantization (Multiple Layers)

```cpp
// All layers in one kernel call
int total_elements = n_layers * elements_per_layer;
ggml_metal_quantize_kv_cache_q8_batch_cpu(
    dev, src, dst, scales,
    total_elements, group_size
);
```

---

## Troubleshooting

### Issue: Slow performance

**Check**:
1. Using `_pipeline_cpu()` API? (fastest)
2. DYLD_LIBRARY_PATH set correctly?
3. Running on Apple Silicon? (required)

### Issue: High error

**Check**:
1. Group size = 64? (recommended)
2. Input data range? (extreme values → higher error)

### Issue: Compilation errors

**Check**:
1. CMake flags: `-DGGML_METAL=ON`
2. macOS version: ≥ 12.0 required
3. Xcode Command Line Tools installed

---

## Performance Tips

1. **Use Pipeline API** for online quantization (8.47 GB/s)
2. **Use Offline API** for model loading (5.30 GB/s, one-time cost)
3. **Batch multiple layers** when possible
4. **Zero-copy buffers** (already done in pipeline API)

---

## Next Steps

- ✅ **Integrate into llama.cpp** - See `llama-kv-cache.cpp`
- 🚀 **Kernel Fusion** - Integrate into attention kernel
- 📊 **Benchmark on Qwen3-30B** - Measure end-to-end impact

---

**Full Documentation**: [KV_CACHE_QUANTIZATION.md](./KV_CACHE_QUANTIZATION.md)

**Source Code**: `ggml/src/ggml-metal/ggml-metal-context.m`

**Tests**: `tests/test-kv-quantize-pipeline.cpp`
