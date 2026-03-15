// Test Batch and Offline KV Cache Quantization
// Tests three stackable optimization approaches:
// - Batch Processing (reduce kernel launch overhead)
// - Offline Quantization (quantize once, use many times)
// - Combined approach

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>

// Test functions declared in ggml-metal-context.m
extern "C" {
    void ggml_metal_quantize_kv_cache_q8_cpu(
            void * dev,
            const ggml_fp16_t * src,
            int8_t * dst,
            ggml_fp16_t * scales,
            int ne00,
            int group_size);

    void ggml_metal_dequantize_kv_cache_q8_cpu(
            void * dev,
            const int8_t * src,
            const ggml_fp16_t * scales,
            ggml_fp16_t * dst,
            int ne00,
            int group_size);

    void ggml_metal_quantize_kv_cache_q8_batch_cpu(
            void * dev,
            const ggml_fp16_t * src,
            int8_t * dst,
            ggml_fp16_t * scales,
            int total_elements,
            int group_size);

    void ggml_metal_dequantize_kv_cache_q8_batch_cpu(
            void * dev,
            const int8_t * src,
            const ggml_fp16_t * scales,
            ggml_fp16_t * dst,
            int total_elements,
            int group_size);

    void ggml_metal_kv_cache_quantize_offline_cpu(
            void * dev,
            ggml_fp16_t * kv_cache,
            int n_layers,
            int hidden_dim,
            int seq_len,
            int group_size);
}

// Simulation parameters
const int N_LAYERS = 32;        // Qwen3-30B-like
const int HIDDEN_DIM = 5120;    // Hidden dimension
const int SEQ_LEN = 256;        // Sequence length
const int GROUP_SIZE = 64;      // Quantization group size

// Generate random FP16 data
void generate_random_fp16(ggml_fp16_t * data, int n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        data[i] = ggml_fp32_to_fp16(dist(gen));
    }
}

// Test 1: Batch quantization (all layers at once)
void test_batch_quantization(ggml_metal_device_t dev) {
    printf("\n=== Test 1: Batch Quantization ===\n");

    // Total elements across all layers (K + V)
    int elements_per_layer = HIDDEN_DIM * SEQ_LEN;
    int total_elements = N_LAYERS * elements_per_layer * 2;
    int n_groups = (total_elements + GROUP_SIZE - 1) / GROUP_SIZE;

    printf("Configuration:\n");
    printf("  Layers: %d\n", N_LAYERS);
    printf("  Hidden Dim: %d\n", HIDDEN_DIM);
    printf("  Seq Len: %d\n", SEQ_LEN);
    printf("  Total Elements: %d (%.2f MB FP16)\n",
           total_elements, total_elements * 2.0 / 1024 / 1024);
    printf("  Groups: %d\n", n_groups);

    // Allocate buffers
    ggml_fp16_t * src_data = (ggml_fp16_t *)malloc(total_elements * sizeof(ggml_fp16_t));
    int8_t * dst_data = (int8_t *)malloc(total_elements * sizeof(int8_t));
    ggml_fp16_t * scales = (ggml_fp16_t *)malloc(n_groups * sizeof(ggml_fp16_t));
    ggml_fp16_t * reconstructed = (ggml_fp16_t *)malloc(total_elements * sizeof(ggml_fp16_t));

    if (!src_data || !dst_data || !scales || !reconstructed) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }

    // Generate random input
    generate_random_fp16(src_data, total_elements);

    // Timing - Batch quantization
    auto start = std::chrono::high_resolution_clock::now();
    ggml_metal_quantize_kv_cache_q8_batch_cpu(dev, src_data, dst_data, scales,
                                              total_elements, GROUP_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    double quant_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Timing - Batch dequantization
    start = std::chrono::high_resolution_clock::now();
    ggml_metal_dequantize_kv_cache_q8_batch_cpu(dev, dst_data, scales, reconstructed,
                                                total_elements, GROUP_SIZE);
    end = std::chrono::high_resolution_clock::now();
    double dequant_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Verify correctness
    double max_error = 0.0;
    double avg_error = 0.0;

    for (int i = 0; i < total_elements; i++) {
        float orig = ggml_fp16_to_fp32(src_data[i]);
        float recon = ggml_fp16_to_fp32(reconstructed[i]);
        double error = std::abs(orig - recon);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= total_elements;

    // Results
    double data_mb = total_elements * 2.0 / 1024 / 1024;
    double quant_throughput = data_mb / quant_time;
    double dequant_throughput = data_mb / dequant_time;

    printf("\nResults:\n");
    printf("  Quantization Time: %.3f ms\n", quant_time);
    printf("  Quantization Throughput: %.2f GB/s\n", quant_throughput);
    printf("  Dequantization Time: %.3f ms\n", dequant_time);
    printf("  Dequantization Throughput: %.2f GB/s\n", dequant_throughput);
    printf("  Max Error: %.3f\n", max_error);
    printf("  Avg Error: %.3f\n", avg_error);
    printf("  Memory Reduction: %.1fx (%.2f MB → %.2f MB)\n",
           (double)(total_elements * 2) / (total_elements + n_groups * 2),
           data_mb,
           (total_elements + n_groups * 2.0) / 1024 / 1024);

    // Correctness check
    if (max_error < 0.05) {
        printf("  ✅ Correctness: PASS (Max Error < 0.05)\n");
    } else {
        printf("  ❌ Correctness: FAIL (Max Error = %.3f)\n", max_error);
    }

    free(src_data);
    free(dst_data);
    free(scales);
    free(reconstructed);
}

// Test 2: Per-layer quantization (baseline for comparison)
void test_per_layer_quantization(ggml_metal_device_t dev) {
    printf("\n=== Test 2: Per-Layer Quantization (Baseline) ===\n");

    int elements_per_layer = HIDDEN_DIM * SEQ_LEN * 2;  // K + V
    int n_groups = (elements_per_layer + GROUP_SIZE - 1) / GROUP_SIZE;

    printf("Configuration:\n");
    printf("  Layers: %d\n", N_LAYERS);
    printf("  Elements per Layer: %d (%.2f MB)\n",
           elements_per_layer, elements_per_layer * 2.0 / 1024 / 1024);

    // Allocate buffers for one layer
    ggml_fp16_t * src_data = (ggml_fp16_t *)malloc(elements_per_layer * sizeof(ggml_fp16_t));
    int8_t * dst_data = (int8_t *)malloc(elements_per_layer * sizeof(int8_t));
    ggml_fp16_t * scales = (ggml_fp16_t *)malloc(n_groups * sizeof(ggml_fp16_t));

    if (!src_data || !dst_data || !scales) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }

    generate_random_fp16(src_data, elements_per_layer);

    // Timing for N_LAYERS sequential calls
    auto start = std::chrono::high_resolution_clock::now();

    for (int layer = 0; layer < N_LAYERS; layer++) {
        ggml_metal_quantize_kv_cache_q8_cpu(dev, src_data, dst_data, scales,
                                            elements_per_layer, GROUP_SIZE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end - start).count();

    double total_mb = N_LAYERS * elements_per_layer * 2.0 / 1024 / 1024;
    double throughput = total_mb / total_time;

    printf("\nResults:\n");
    printf("  Total Time: %.3f ms (%d kernel calls)\n", total_time, N_LAYERS);
    printf("  Throughput: %.2f GB/s\n", throughput);
    printf("  Time per Layer: %.3f ms\n", total_time / N_LAYERS);

    free(src_data);
    free(dst_data);
    free(scales);
}

// Test 3: Offline quantization
void test_offline_quantization(ggml_metal_device_t dev) {
    printf("\n=== Test 3: Offline Quantization ===\n");

    // Allocate KV Cache buffer
    int elements_per_layer = HIDDEN_DIM * SEQ_LEN;
    int total_elements = N_LAYERS * elements_per_layer * 2;

    ggml_fp16_t * kv_cache = (ggml_fp16_t *)malloc(total_elements * sizeof(ggml_fp16_t));
    if (!kv_cache) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return;
    }

    generate_random_fp16(kv_cache, total_elements);

    // Offline quantization (one-time operation at model load)
    ggml_metal_kv_cache_quantize_offline_cpu(dev, kv_cache, N_LAYERS,
                                             HIDDEN_DIM, SEQ_LEN, GROUP_SIZE);

    free(kv_cache);
}

// Test 4: Comparison
void test_comparison() {
    printf("\n=== Performance Comparison ===\n");
    printf("\nExpected Results:\n");
    printf("  Batch Processing (方案 B):\n");
    printf("    - Reduces kernel launch overhead from O(layers) to O(1)\n");
    printf("    - Expected speedup: 2-3x for %d layers\n", N_LAYERS);
    printf("    - Best for: Online quantization during generation\n");
    printf("\n");
    printf("  Offline Quantization (方案 C):\n");
    printf("    - Quantize once at model load, use many times\n");
    printf("    - Expected overhead: 0 during generation (only dequantization)\n");
    printf("    - Best for: Long conversations, RAG applications\n");
    printf("\n");
    printf("  Combined (B + C):\n");
    printf("    - Use offline for preloaded cache\n");
    printf("    - Use batch for new tokens during generation\n");
    printf("    - Best for: Hybrid scenarios\n");
}

int main() {
    printf("==========================================================\n");
    printf("ThunderLLAMA KV Cache Quantization - Batch & Offline Tests\n");
    printf("==========================================================\n");

    // Initialize Metal device
    printf("\n[1/5] Initializing Metal device...\n");
    ggml_metal_device_t dev = ggml_metal_device_get(0);
    if (!dev) {
        fprintf(stderr, "ERROR: Failed to get Metal device\n");
        return 1;
    }
    printf("  ✓ Metal device initialized\n");

    // Test 1: Batch quantization
    printf("\n[2/5] Testing batch quantization...\n");
    test_batch_quantization(dev);

    // Test 2: Per-layer quantization (baseline)
    printf("\n[3/5] Testing per-layer quantization (baseline)...\n");
    test_per_layer_quantization(dev);

    // Test 3: Offline quantization
    printf("\n[4/5] Testing offline quantization...\n");
    test_offline_quantization(dev);

    // Test 4: Comparison
    printf("\n[5/5] Performance comparison...\n");
    test_comparison();

    printf("\n=== All Tests Complete ===\n");
    printf("Next Step: Implement Kernel Fusion (方案 A)\n");

    return 0;
}
