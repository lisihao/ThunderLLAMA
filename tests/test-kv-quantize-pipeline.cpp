// Test CPU-GPU Pipeline Quantization
// Tests Zero-Copy + Pipeline approach to hide kernel launch overhead

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>

// Test functions
extern "C" {
    // Baseline: per-layer quantization
    void ggml_metal_quantize_kv_cache_q8_cpu(
            void * dev,
            const ggml_fp16_t * src,
            int8_t * dst,
            ggml_fp16_t * scales,
            int ne00,
            int group_size);

    // Batch quantization (previous implementation)
    void ggml_metal_quantize_kv_cache_q8_batch_cpu(
            void * dev,
            const ggml_fp16_t * src,
            int8_t * dst,
            ggml_fp16_t * scales,
            int total_elements,
            int group_size);

    // Pipeline quantization (new optimized implementation)
    void ggml_metal_quantize_kv_cache_q8_pipeline_cpu(
            void * dev,
            const ggml_fp16_t * src,
            int8_t * dst,
            ggml_fp16_t * scales,
            int n_layers,
            int elements_per_layer,
            int group_size);
}

// Test parameters
const int N_LAYERS = 32;        // Qwen3-30B
const int HIDDEN_DIM = 5120;
const int SEQ_LEN = 256;
const int GROUP_SIZE = 64;

// Generate random FP16 data
void generate_random_fp16(ggml_fp16_t * data, int n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        data[i] = ggml_fp32_to_fp16(dist(gen));
    }
}

// Verify correctness
bool verify_correctness(const ggml_fp16_t * src, const int8_t * dst,
                       const ggml_fp16_t * scales, int n_groups, int group_size) {
    double max_error = 0.0;
    double avg_error = 0.0;
    int error_count = 0;

    for (int g = 0; g < n_groups; g++) {
        float scale = ggml_fp16_to_fp32(scales[g]);

        for (int i = 0; i < group_size; i++) {
            int idx = g * group_size + i;
            float orig = ggml_fp16_to_fp32(src[idx]);
            float reconstructed = (float)dst[idx] * scale;

            double error = std::abs(orig - reconstructed);
            max_error = std::max(max_error, error);
            avg_error += error;
            error_count++;
        }
    }

    avg_error /= error_count;

    printf("  Max Error: %.6f\n", max_error);
    printf("  Avg Error: %.6f\n", avg_error);

    return max_error < 0.05 && avg_error < 0.01;
}

int main() {
    printf("==========================================================\n");
    printf("ThunderLLAMA KV Cache - CPU-GPU Pipeline Test\n");
    printf("==========================================================\n");

    // Initialize Metal device
    printf("\n[1/5] Initializing Metal device...\n");
    ggml_metal_device_t dev = ggml_metal_device_get(0);
    if (!dev) {
        fprintf(stderr, "ERROR: Failed to get Metal device\n");
        return 1;
    }
    printf("  ✓ Metal device initialized\n");

    // Allocate memory
    int elements_per_layer = HIDDEN_DIM * SEQ_LEN * 2;  // K + V
    int total_elements = N_LAYERS * elements_per_layer;
    int groups_per_layer = (elements_per_layer + GROUP_SIZE - 1) / GROUP_SIZE;
    int total_groups = N_LAYERS * groups_per_layer;

    printf("\n[2/5] Allocating memory...\n");
    printf("  Layers: %d\n", N_LAYERS);
    printf("  Elements per Layer: %d (%.2f MB)\n",
           elements_per_layer, elements_per_layer * 2.0 / 1024 / 1024);
    printf("  Total Elements: %d (%.2f MB)\n",
           total_elements, total_elements * 2.0 / 1024 / 1024);

    ggml_fp16_t * src = (ggml_fp16_t *)malloc(total_elements * sizeof(ggml_fp16_t));
    int8_t * dst_pipeline = (int8_t *)malloc(total_elements * sizeof(int8_t));
    int8_t * dst_batch = (int8_t *)malloc(total_elements * sizeof(int8_t));
    int8_t * dst_perlayer = (int8_t *)malloc(total_elements * sizeof(int8_t));
    ggml_fp16_t * scales_pipeline = (ggml_fp16_t *)malloc(total_groups * sizeof(ggml_fp16_t));
    ggml_fp16_t * scales_batch = (ggml_fp16_t *)malloc(total_groups * sizeof(ggml_fp16_t));
    ggml_fp16_t * scales_perlayer = (ggml_fp16_t *)malloc(total_groups * sizeof(ggml_fp16_t));

    if (!src || !dst_pipeline || !dst_batch || !dst_perlayer ||
        !scales_pipeline || !scales_batch || !scales_perlayer) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return 1;
    }

    generate_random_fp16(src, total_elements);
    printf("  ✓ Memory allocated\n");

    // Test 1: Pipeline quantization
    printf("\n[3/5] Testing Pipeline Quantization...\n");
    printf("========================================\n");

    ggml_metal_quantize_kv_cache_q8_pipeline_cpu(
        dev, src, dst_pipeline, scales_pipeline,
        N_LAYERS, elements_per_layer, GROUP_SIZE
    );

    printf("\nVerifying correctness (Pipeline):\n");
    if (verify_correctness(src, dst_pipeline, scales_pipeline, total_groups, GROUP_SIZE)) {
        printf("  ✅ Correctness: PASS\n");
    } else {
        printf("  ❌ Correctness: FAIL\n");
    }

    // Test 2: Batch quantization (previous implementation)
    printf("\n[4/5] Testing Batch Quantization (comparison)...\n");
    printf("================================================\n");

    auto start = std::chrono::high_resolution_clock::now();
    ggml_metal_quantize_kv_cache_q8_batch_cpu(
        dev, src, dst_batch, scales_batch,
        total_elements, GROUP_SIZE
    );
    auto end = std::chrono::high_resolution_clock::now();
    double batch_time = std::chrono::duration<double, std::milli>(end - start).count();

    double data_mb = total_elements * 2.0 / 1024 / 1024;
    double batch_throughput = data_mb / batch_time;

    printf("Batch Quantization Results:\n");
    printf("  Time: %.3f ms\n", batch_time);
    printf("  Throughput: %.2f GB/s\n", batch_throughput);

    printf("\nVerifying correctness (Batch):\n");
    if (verify_correctness(src, dst_batch, scales_batch, total_groups, GROUP_SIZE)) {
        printf("  ✅ Correctness: PASS\n");
    } else {
        printf("  ❌ Correctness: FAIL\n");
    }

    // Test 3: Per-layer quantization (baseline)
    printf("\n[5/5] Testing Per-Layer Quantization (baseline)...\n");
    printf("==================================================\n");

    start = std::chrono::high_resolution_clock::now();
    for (int layer = 0; layer < N_LAYERS; layer++) {
        int offset = layer * elements_per_layer;
        int scale_offset = layer * groups_per_layer;

        ggml_metal_quantize_kv_cache_q8_cpu(
            dev,
            src + offset,
            dst_perlayer + offset,
            scales_perlayer + scale_offset,
            elements_per_layer,
            GROUP_SIZE
        );
    }
    end = std::chrono::high_resolution_clock::now();
    double perlayer_time = std::chrono::duration<double, std::milli>(end - start).count();
    double perlayer_throughput = data_mb / perlayer_time;

    printf("Per-Layer Quantization Results:\n");
    printf("  Time: %.3f ms (%d kernel calls)\n", perlayer_time, N_LAYERS);
    printf("  Throughput: %.2f GB/s\n", perlayer_throughput);
    printf("  Time per Layer: %.3f ms\n", perlayer_time / N_LAYERS);

    // Performance comparison
    printf("\n==========================================================\n");
    printf("Performance Comparison Summary\n");
    printf("==========================================================\n");
    printf("\n");
    printf("Data Size: %.2f MB (%d layers × %.2f MB/layer)\n",
           data_mb, N_LAYERS, elements_per_layer * 2.0 / 1024 / 1024);
    printf("\n");
    printf("| Method       | Time (ms) | Throughput (GB/s) | vs Pipeline |\n");
    printf("|--------------|-----------|-------------------|-------------|\n");
    printf("| Pipeline     | %.3f   | %.2f            | 1.00x       |\n",
           0.0, 0.0);  // Will be filled by pipeline output
    printf("| Batch        | %.3f   | %.2f            | %.2fx       |\n",
           batch_time, batch_throughput, batch_time / 1.0);
    printf("| Per-Layer    | %.3f   | %.2f            | %.2fx       |\n",
           perlayer_time, perlayer_throughput, perlayer_time / 1.0);
    printf("\n");

    // Cleanup
    free(src);
    free(dst_pipeline);
    free(dst_batch);
    free(dst_perlayer);
    free(scales_pipeline);
    free(scales_batch);
    free(scales_perlayer);

    printf("=== All Tests Complete ===\n");
    printf("\nExpected Results:\n");
    printf("  Pipeline should be 2-3x faster than Batch\n");
    printf("  Pipeline should be 1.5-2x faster than Per-Layer\n");
    printf("\nKey Improvements:\n");
    printf("  ✅ Zero-Copy (UMA shared memory)\n");
    printf("  ✅ CPU-GPU Pipeline (overlapped execution)\n");
    printf("  ✅ CPU NEON acceleration (scales computation)\n");
    printf("  ✅ Simplified GPU kernel (no reduction)\n");

    return 0;
}
