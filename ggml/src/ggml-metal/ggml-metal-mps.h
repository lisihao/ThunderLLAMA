// ThunderLLAMA: Metal Performance Shaders integration
// Phase 1: FP16 GEMM via MPSMatrixMultiplication
// Phase 2: Quantized GEMV via MPSGraph (future)
#pragma once

#include "ggml-metal-device.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_mps_context * ggml_metal_mps_ctx_t;

// Lifecycle
ggml_metal_mps_ctx_t ggml_metal_mps_init(void * device, void * queue);
void                 ggml_metal_mps_free(ggml_metal_mps_ctx_t ctx);

// Check if MPS is available on this system (macOS 13.0+)
bool ggml_metal_mps_available(void);

// FP16 GEMM: C = A * B^T
//
//   A = src0: FP16 weight matrix [M rows, K cols]
//   B = src1: FP32 input  matrix [N rows, K cols]  (transposed internally)
//   C = dst:  FP32 output matrix [M rows, N cols]
//
// Internally converts src1 F32→F16, runs MPS matmul in F16, converts result F16→F32.
//
// IMPORTANT: the command buffer must NOT have an active compute encoder when this
// is called. The caller must end the encoder before calling and re-create after.
//
// Returns true on success, false on error (dimensions mismatch, MPS not available, etc.)
//
bool ggml_metal_mps_mul_mat_f16(
    ggml_metal_mps_ctx_t ctx,
    void * cmd_buf,                     // id<MTLCommandBuffer>
    struct ggml_metal_buffer_id bid_a,  // FP16 src0 buffer + offset
    struct ggml_metal_buffer_id bid_b,  // FP32 src1 buffer + offset
    struct ggml_metal_buffer_id bid_c,  // FP32 dst  buffer + offset
    int M, int N, int K,               // matrix dimensions
    uint64_t row_bytes_a,              // nb01: bytes per row of A
    uint64_t row_bytes_b,              // nb11: bytes per row of B
    uint64_t row_bytes_c               // nb1:  bytes per row of C
);

#ifdef __cplusplus
}
#endif
