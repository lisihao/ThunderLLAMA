// ThunderLLAMA KV Cache 量化测试 (v2 - 高性能版本)
// 测试 GPU 侧 KV Cache 量化的正确性和性能

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cmath>

// 测试专用 API（v2 版本）
extern "C" {
    void ggml_metal_quantize_kv_cache_q8_v2_cpu(
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
}

// 测试参数
constexpr int N_ELEMENTS = 4096 * 512;  // 2M elements = 4 MB FP16
constexpr int GROUP_SIZE = 64;
constexpr int N_ITER = 100;

// 生成随机 FP16 数据
void generate_random_fp16(ggml_fp16_t * data, int n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        data[i] = ggml_fp32_to_fp16(dist(gen));
    }
}

// 验证量化/反量化的正确性
bool verify_correctness(const ggml_fp16_t * original, const ggml_fp16_t * restored, int n) {
    double max_error = 0.0;
    double avg_error = 0.0;

    for (int i = 0; i < n; i++) {
        float orig = ggml_fp16_to_fp32(original[i]);
        float rest = ggml_fp16_to_fp32(restored[i]);
        double error = std::abs(orig - rest);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= n;

    printf("  Max Error: %.6f\n", max_error);
    printf("  Avg Error: %.6f\n", avg_error);

    return max_error < 0.05 && avg_error < 0.01;
}

// 性能测试
double benchmark_quantization(
        ggml_metal_device_t dev,
        const ggml_fp16_t * src_fp16,
        int8_t * dst_q8,
        ggml_fp16_t * scales,
        int n,
        int group_size,
        int n_iter) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iter; i++) {
        ggml_metal_quantize_kv_cache_q8_v2_cpu(dev, src_fp16, dst_q8, scales, n, group_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    return elapsed.count() / n_iter;
}

double benchmark_dequantization(
        ggml_metal_device_t dev,
        const int8_t * src_q8,
        const ggml_fp16_t * scales,
        ggml_fp16_t * dst_fp16,
        int n,
        int group_size,
        int n_iter) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iter; i++) {
        ggml_metal_dequantize_kv_cache_q8_cpu(dev, src_q8, scales, dst_fp16, n, group_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    return elapsed.count() / n_iter;
}

int main() {
    printf("===========================================\n");
    printf("ThunderLLAMA KV Cache 量化测试 (v2)\n");
    printf("===========================================\n");
    printf("Test Config:\n");
    printf("  Elements: %d\n", N_ELEMENTS);
    printf("  Group Size: %d\n", GROUP_SIZE);
    printf("  Iterations: %d\n", N_ITER);
    printf("  Version: v2 (Threadgroup Memory + Parallel Reduction)\n");
    printf("\n");

    // 1. 初始化 Metal device
    printf("[1/5] 初始化 Metal device...\n");
    ggml_metal_device_t dev = ggml_metal_device_get(0);
    if (!dev) {
        fprintf(stderr, "ERROR: Failed to get Metal device\n");
        return 1;
    }
    printf("  ✓ Metal device initialized\n\n");

    // 2. 分配内存
    printf("[2/5] 分配内存...\n");
    int n_groups = (N_ELEMENTS + GROUP_SIZE - 1) / GROUP_SIZE;

    ggml_fp16_t * src_fp16  = (ggml_fp16_t *)malloc(N_ELEMENTS * sizeof(ggml_fp16_t));
    int8_t      * dst_q8    = (int8_t *)malloc(N_ELEMENTS * sizeof(int8_t));
    ggml_fp16_t * scales    = (ggml_fp16_t *)malloc(n_groups * sizeof(ggml_fp16_t));
    ggml_fp16_t * restored  = (ggml_fp16_t *)malloc(N_ELEMENTS * sizeof(ggml_fp16_t));

    if (!src_fp16 || !dst_q8 || !scales || !restored) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return 1;
    }

    generate_random_fp16(src_fp16, N_ELEMENTS);
    printf("  ✓ Memory allocated and initialized\n\n");

    // 3. 正确性测试
    printf("[3/5] 正确性测试...\n");
    ggml_metal_quantize_kv_cache_q8_v2_cpu(dev, src_fp16, dst_q8, scales, N_ELEMENTS, GROUP_SIZE);
    ggml_metal_dequantize_kv_cache_q8_cpu(dev, dst_q8, scales, restored, N_ELEMENTS, GROUP_SIZE);

    bool correct = verify_correctness(src_fp16, restored, N_ELEMENTS);
    if (correct) {
        printf("  ✓ 正确性验证通过\n\n");
    } else {
        fprintf(stderr, "  ✗ 正确性验证失败\n\n");
        return 1;
    }

    // 4. 性能测试 - 量化
    printf("[4/5] 性能测试 - 量化 (v2)...\n");
    double quant_time = benchmark_quantization(dev, src_fp16, dst_q8, scales,
                                                 N_ELEMENTS, GROUP_SIZE, N_ITER);
    double quant_throughput = (N_ELEMENTS * sizeof(ggml_fp16_t)) / (quant_time * 1e6);
    printf("  Time per quantization: %.3f ms\n", quant_time);
    printf("  Throughput: %.2f GB/s\n", quant_throughput);
    printf("\n");

    // 5. 性能测试 - 反量化
    printf("[5/5] 性能测试 - 反量化...\n");
    double dequant_time = benchmark_dequantization(dev, dst_q8, scales, restored,
                                                     N_ELEMENTS, GROUP_SIZE, N_ITER);
    double dequant_throughput = (N_ELEMENTS * sizeof(int8_t)) / (dequant_time * 1e6);
    printf("  Time per dequantization: %.3f ms\n", dequant_time);
    printf("  Throughput: %.2f GB/s\n", dequant_throughput);
    printf("\n");

    // 总结
    printf("===========================================\n");
    printf("测试总结 (v2 优化版本)\n");
    printf("===========================================\n");
    printf("✓ 正确性: 通过\n");
    printf("✓ 量化性能: %.3f ms (%.2f GB/s)\n", quant_time, quant_throughput);
    printf("✓ 反量化性能: %.3f ms (%.2f GB/s)\n", dequant_time, dequant_throughput);
    printf("✓ 内存压缩: %.1fx (FP16 → INT8 + scales)\n",
           (float)(N_ELEMENTS * sizeof(ggml_fp16_t)) /
           (N_ELEMENTS * sizeof(int8_t) + n_groups * sizeof(ggml_fp16_t)));
    printf("\n");

    double theoretical_bandwidth = 273.0;  // M4 Pro
    printf("带宽利用率: %.1f%% (%.2f / %.0f GB/s)\n",
           (quant_throughput / theoretical_bandwidth) * 100,
           quant_throughput, theoretical_bandwidth);
    printf("\n");

    // 清理
    free(src_fp16);
    free(dst_q8);
    free(scales);
    free(restored);

    printf("✓ 测试完成！\n");
    return 0;
}
