# Metal Performance Shaders (MPS) Integration

## Overview

ThunderLLAMA integrates Apple's Metal Performance Shaders framework alongside hand-written Metal kernels for GPU-accelerated LLM inference. This document covers the architecture, research findings, and optimization strategy.

## Phase 1: MPS Foundation + FP16 GEMM (Completed)

### Architecture

MPS integration uses a hybrid approach: MPSMatrixMultiplication for eligible FP16 matmul operations, with fallback to existing hand-written Metal kernels for all other operations.

```
ggml_metal_op_mul_mat()
├── MPS path: USE_MPS_GRAPH=1, src0=F16, src1=F32, 2D, not transposed
│   ├── pause_encoder() — end compute encoder
│   ├── F32→F16 convert src1 (custom Metal kernel)
│   ├── MPSMatrixMultiplication encode
│   ├── F16→F32 convert result (custom Metal kernel)
│   └── resume_encoder() — restart compute encoder
└── Metal kernel path: all other cases
    ├── kernel_mul_mv_ext (BS 2-8, fused dequant+matmul)
    ├── kernel_mul_mv_q5_K_f32 (BS=1 decode GEMV)
    └── kernel_mul_mm (BS>8, matrix-matrix)
```

### Files Modified/Added

| File | Change | Lines |
|------|--------|-------|
| `ggml/src/ggml-metal/ggml-metal-mps.h` | NEW - C-compatible MPS wrapper header | 52 |
| `ggml/src/ggml-metal/ggml-metal-mps.mm` | NEW - Obj-C++ MPS implementation | 318 |
| `ggml/src/ggml-metal/CMakeLists.txt` | ADD MetalPerformanceShaders framework + .mm source | +5 |
| `ggml/src/ggml-metal/ggml-metal-device.h` | ADD mps_ctx getter declaration | +1 |
| `ggml/src/ggml-metal/ggml-metal-device.m` | ADD MPS context init/free/getter | +20 |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | ADD MPS dispatch branch + encoder pause/resume | +50 |
| `common/config-parser.h` | ADD USE_MPS_GRAPH env var mapping | +5 |
| `thunderllama.conf` | ADD USE_MPS_GRAPH=0 | +1 |

### MPS Implementation Details

**MPSMatrixMultiplication** (not MPSGraph):
- Chosen over MPSGraph because it supports buffer offsets natively
- Operates on FP16 matrices (A × B^T = C)
- src1 (FP32) is converted to FP16 via custom Metal compute kernel
- Result (FP16) is converted back to FP32

**Encoder Pause/Resume**:
- MPS encoding requires the compute command encoder to be ended first
- `pause_encoder()` ends the current compute encoder
- MPS encodes on the same command buffer
- `resume_encoder()` creates a new compute encoder for subsequent ops

**Temp Buffer Management**:
- Cached temporary buffers for F32↔F16 conversion
- 20% headroom to avoid frequent reallocations
- Manual retain/release (MRC, not ARC)

### Phase 1 Verification

| Check | Result |
|-------|--------|
| Build | 0 errors, 0 warnings |
| Server start | MPS init messages confirmed |
| USE_MPS_GRAPH=0 | No behavioral change (fallback path) |
| Inference | TG=56.6, PP=130.0 (with MPS disabled) |

### Configuration

```bash
# thunderllama.conf
USE_MPS_GRAPH=0   # 0=disabled (default), 1=enabled
```

---

## Phase 2 Research: Decode GEMV Optimization

### Key Finding: MPSGraph Does NOT Have quantizedMatmul

The original plan assumed MPSGraph provides a `quantizedMatmul` API for fused dequantization + matmul. **This is incorrect.**

- `quantizedMatmul` belongs to **MLX** (Apple's ML framework), not MPSGraph
- MLX uses **custom Metal kernels** (STEEL GEMM framework) for quantized matmul
- MLX's approach: `QuantizedBlockLoader` + `qdot` template for fused dequant+dot product

**Sources**:
- MLX `mlx/backend/metal/kernels/quantized.h` — custom Metal kernels, not MPSGraph
- MLX STEEL GEMM — `BlockLoader`, `mma_t` (simdgroup_matrix), Split-K tiling
- vllm-mlx uses MLX's `mx.quantized_matmul()` which calls these custom kernels

### Bandwidth Analysis: Current Decode GEMV

**BS=1 Kernel**: `kernel_mul_mv_q5_K_f32` (ggml-metal.metal:7465)

Architecture:
```
32 threads per simdgroup
├── 4 threads parallel along K dimension (ix = tiisg % 4)
├── 8 rows processed simultaneously (N_R0_Q5_K = 8)
├── Each thread reads nb/4 quantized blocks sequentially
└── simd_sum() for final reduction
```

**Bandwidth utilization**:

| Metric | Value |
|--------|-------|
| Q5_K bits per weight | 5.5 bpw (176 bytes / 256 elements) |
| Qwen3-30B-A3B active params | ~3B (MoE) |
| Data read per token | ~2.06 GB |
| M4 Pro memory bandwidth | ~273 GB/s |
| Theoretical max TG | ~132 tok/s |
| Current TG | 65-72 tok/s |
| **Bandwidth utilization** | **~50-55%** |

**Root cause of 45-50% gap**: Only 4-way K-dimension parallelism within each simdgroup. For large matrices (e.g., MLP up_proj K=14336), each thread serially reads 14 blocks x 176 bytes = 2464 bytes. Insufficient memory-level parallelism to hide DRAM latency.

### Optimization Strategy: Split-K GEMV

**Core idea**: Split the K dimension across multiple threadgroups instead of processing it within a single simdgroup.

```
Current ggml kernel (4-way K parallel):
  1 threadgroup → 4 threads read entire K → simd_sum

Split-K (target 16-32 way K parallel):
  8-16 threadgroups each read K/8~K/16 → atomic_add or 2-pass reduce
  More concurrent memory requests → better bandwidth utilization
```

**Expected improvement**:
- Bandwidth utilization: 50% → 70-80%
- TG: 65-72 → 85-105 tok/s (+25-45%)

**Implementation approach** (Plan B — MLX-style fused kernel):
- Reuse existing `dequantize_q5_K` logic (no need to rewrite)
- Add Split-K dispatch: partition K blocks across threadgroups
- Add inter-threadgroup reduction (atomic float add or 2-pass kernel)
- Tune: split factor, threadgroup size, rows per threadgroup

**Validation plan**:
1. Day 1: Q4_0 Split-K PoC (simplest format) → benchmark vs existing
2. Day 2: Q5_K Split-K → benchmark
3. Day 3: Tuning + integration

**Go/No-Go**: Q4_0 PoC must show >= 10% improvement over existing kernel.

### Alternative Approaches Considered

| Approach | Description | Verdict |
|----------|-------------|---------|
| A: GEMM path only | dequant→FP16→MPS matmul for prefill | Rejected (prefill cached by LMCache) |
| B: Split-K fused kernel | Reuse dequant, split K across threadgroups | **Selected** |
| C: MLX kernel port | Convert Q4_K to MLX affine format | Rejected (format conversion overhead, precision loss) |

### Q5_K Block Format Reference

```c
typedef struct {
    half    d;           // 2 bytes — super-block scale
    half    dmin;        // 2 bytes — super-block min
    uint8_t scales[12];  // 12 bytes — packed 6-bit sub-block scales/mins
    uint8_t qh[32];      // 32 bytes — high bits (5th bit per weight)
    uint8_t qs[128];     // 128 bytes — low 4 bits per weight
} block_q5_K;            // Total: 176 bytes per 256 elements (5.5 bpw)

// Dequantization:
// value = d * sc * (quant_4bit | (high_bit << 4)) - dmin * m
// where sc, m are 6-bit values extracted from scales[] with bit manipulation
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Split-K atomic_add contention | Medium | Medium | Use 2-pass reduce if atomic is slow |
| Kernel not faster than existing | Medium | High | Q4_0 PoC on Day 1 as Go/No-Go gate |
| Register pressure from dequant | Low | Medium | Reuse existing dequant code |
| Split-K tuning difficult | Medium | Low | Start with MLX defaults (split=8) |
| macOS version compatibility | Low | Low | @available check, fallback |

---

## References

- MLX quantized kernels: `mlx/backend/metal/kernels/quantized.h`
- MLX STEEL GEMM: `mlx/backend/metal/kernels/steel/`
- ggml Metal kernels: `ggml/src/ggml-metal/ggml-metal.metal`
- Apple MPS documentation: Metal Performance Shaders framework
