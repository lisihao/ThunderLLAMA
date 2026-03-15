#include "ggml-metal-cpu-neon.h"
#include <math.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// CPU-side NEON accelerated reduction for computing quantization scales
void ggml_metal_cpu_compute_scales_neon(
        const ggml_fp16_t * src,
        ggml_fp16_t       * scales,
        int                 n_groups,
        int                 group_size) {

#ifdef __ARM_NEON
    // Use OpenMP for multi-core parallelism if available
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int g = 0; g < n_groups; g++) {
        const ggml_fp16_t * group_src = src + g * group_size;

        // NEON: Process 8 FP16 elements at a time
        float16x8_t max_vec = vdupq_n_f16(0);

        int i = 0;
        // Vectorized loop: process 8 elements per iteration
        for (; i + 8 <= group_size; i += 8) {
            float16x8_t val = vld1q_f16((const __fp16 *)(group_src + i));
            val = vabsq_f16(val);  // abs(val)
            max_vec = vmaxq_f16(max_vec, val);  // max
        }

        // Horizontal max reduction (find max among 8 lanes)
        float max_val = vmaxvq_f16(max_vec);

        // Handle remaining elements (if group_size not divisible by 8)
        for (; i < group_size; i++) {
            float val = fabsf(ggml_fp16_to_fp32(group_src[i]));
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute scale: max / 127.0
        scales[g] = ggml_fp32_to_fp16(max_val / 127.0f);
    }
#else
    // Fallback for non-NEON platforms
    for (int g = 0; g < n_groups; g++) {
        const ggml_fp16_t * group_src = src + g * group_size;

        float max_val = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float val = fabsf(ggml_fp16_to_fp32(group_src[i]));
            if (val > max_val) {
                max_val = val;
            }
        }

        scales[g] = ggml_fp32_to_fp16(max_val / 127.0f);
    }
#endif
}
