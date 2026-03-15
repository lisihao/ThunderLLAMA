#pragma once

#include "ggml.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// CPU-side NEON accelerated reduction for finding max values in groups
// Used for computing quantization scales
void ggml_metal_cpu_compute_scales_neon(
        const ggml_fp16_t * src,
        ggml_fp16_t       * scales,
        int                 n_groups,
        int                 group_size);

#ifdef __cplusplus
}
#endif
