// ThunderLLAMA KV Cache 量化性能分析工具
// 分解测量每个阶段的时间

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>

// 测试专用 API
extern "C" {
    void ggml_metal_quantize_kv_cache_q8_cpu(
            void * dev,
            const ggml_fp16_t * src,
            int8_t * dst,
            ggml_fp16_t * scales,
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

// 性能分析
void profile_quantization() {
    printf("===========================================\n");
    printf("KV Cache 量化性能分析\n");
    printf("===========================================\n");
    printf("测试规模: %d elements (%.2f MB FP16)\n", N_ELEMENTS, N_ELEMENTS * 2.0 / 1024 / 1024);
    printf("迭代次数: %d\n\n", N_ITER);

    // 1. 初始化
    ggml_metal_device_t dev = ggml_metal_device_get(0);
    if (!dev) {
        fprintf(stderr, "ERROR: Failed to get Metal device\n");
        return;
    }

    int n_groups = (N_ELEMENTS + GROUP_SIZE - 1) / GROUP_SIZE;

    // 2. 分配 CPU 内存
    auto t_start = std::chrono::high_resolution_clock::now();

    ggml_fp16_t * src_fp16 = (ggml_fp16_t *)malloc(N_ELEMENTS * sizeof(ggml_fp16_t));
    int8_t      * dst_q8   = (int8_t *)malloc(N_ELEMENTS * sizeof(int8_t));
    ggml_fp16_t * scales   = (ggml_fp16_t *)malloc(n_groups * sizeof(ggml_fp16_t));

    generate_random_fp16(src_fp16, N_ELEMENTS);

    auto t_malloc = std::chrono::high_resolution_clock::now();
    double malloc_time = std::chrono::duration<double, std::milli>(t_malloc - t_start).count();
    printf("[1] CPU 内存分配: %.3f ms\n", malloc_time);

    // 3. 第一次调用（warm-up + Metal 初始化）
    t_start = std::chrono::high_resolution_clock::now();
    ggml_metal_quantize_kv_cache_q8_cpu(dev, src_fp16, dst_q8, scales, N_ELEMENTS, GROUP_SIZE);
    auto t_warmup = std::chrono::high_resolution_clock::now();
    double warmup_time = std::chrono::duration<double, std::milli>(t_warmup - t_start).count();
    printf("[2] Warm-up (首次调用): %.3f ms\n", warmup_time);

    // 4. 性能测试（包含 CPU-GPU 传输）
    t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITER; i++) {
        ggml_metal_quantize_kv_cache_q8_cpu(dev, src_fp16, dst_q8, scales, N_ELEMENTS, GROUP_SIZE);
    }
    auto t_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(t_total - t_start).count() / N_ITER;
    double total_throughput = (N_ELEMENTS * sizeof(ggml_fp16_t)) / (total_time * 1e6);

    printf("[3] 总时间 (CPU wrapper): %.3f ms/iter (%.2f GB/s)\n", total_time, total_throughput);

    // 5. 理论分析
    printf("\n===========================================\n");
    printf("理论分析\n");
    printf("===========================================\n");

    double data_in_mb = N_ELEMENTS * sizeof(ggml_fp16_t) / 1024.0 / 1024.0;
    double data_out_mb = N_ELEMENTS * sizeof(int8_t) / 1024.0 / 1024.0;
    double scales_mb = n_groups * sizeof(ggml_fp16_t) / 1024.0 / 1024.0;

    printf("数据量:\n");
    printf("  输入 (FP16):  %.2f MB\n", data_in_mb);
    printf("  输出 (INT8):  %.2f MB\n", data_out_mb);
    printf("  Scales:       %.2f MB\n", scales_mb);
    printf("  总计:         %.2f MB\n", data_in_mb + data_out_mb + scales_mb);

    double theoretical_bandwidth = 273.0;  // GB/s for M4 Pro
    double theoretical_time_ms = (data_in_mb + data_out_mb + scales_mb) / theoretical_bandwidth;

    printf("\n理论最优时间 (纯内存带宽):\n");
    printf("  @ %.0f GB/s: %.6f ms\n", theoretical_bandwidth, theoretical_time_ms);

    printf("\n实际性能:\n");
    printf("  实际时间:     %.3f ms\n", total_time);
    printf("  实际吞吐:     %.2f GB/s\n", total_throughput);
    printf("  带宽利用率:   %.1f%%\n", (total_throughput / theoretical_bandwidth) * 100);
    printf("  差距倍数:     %.1fx\n", total_time / theoretical_time_ms);

    // 6. 时间分解估算
    printf("\n===========================================\n");
    printf("时间分解估算\n");
    printf("===========================================\n");

    // 假设 unified memory memcpy 速度约 40-50 GB/s
    double memcpy_bandwidth = 40.0;  // GB/s (保守估计)
    double cpu_to_gpu_time = data_in_mb / memcpy_bandwidth;
    double gpu_to_cpu_time = (data_out_mb + scales_mb) / memcpy_bandwidth;
    double total_memcpy_time = cpu_to_gpu_time + gpu_to_cpu_time;
    double kernel_time = total_time - total_memcpy_time;

    printf("假设 memcpy @ %.0f GB/s:\n", memcpy_bandwidth);
    printf("  CPU → GPU (%.2f MB):  %.3f ms\n", data_in_mb, cpu_to_gpu_time);
    printf("  GPU → CPU (%.2f MB):  %.3f ms\n", data_out_mb + scales_mb, gpu_to_cpu_time);
    printf("  Memcpy 总计:          %.3f ms (%.1f%%)\n",
           total_memcpy_time, (total_memcpy_time / total_time) * 100);
    printf("  Kernel 执行:          %.3f ms (%.1f%%)\n",
           kernel_time, (kernel_time / total_time) * 100);

    if (kernel_time > 0) {
        double kernel_effective_bandwidth = (data_in_mb + data_out_mb) / kernel_time;
        printf("  Kernel 有效带宽:      %.2f GB/s\n", kernel_effective_bandwidth);
    }

    // 7. 瓶颈分析
    printf("\n===========================================\n");
    printf("瓶颈分析\n");
    printf("===========================================\n");

    if (total_memcpy_time > total_time * 0.5) {
        printf("⚠️  主要瓶颈: CPU-GPU 数据传输 (%.1f%%)\n",
               (total_memcpy_time / total_time) * 100);
        printf("    建议: 避免 CPU wrapper，直接在 GPU 上处理\n");
    } else if (kernel_time > total_time * 0.5) {
        printf("⚠️  主要瓶颈: Kernel 执行 (%.1f%%)\n",
               (kernel_time / total_time) * 100);

        if (total_throughput < 10.0) {
            printf("    可能原因:\n");
            printf("    - Kernel 启动延迟\n");
            printf("    - 内存访问模式不优 (cache miss)\n");
            printf("    - Threadgroup 调度不充分\n");
            printf("    - Max reduction 算法效率低\n");
        }
    }

    printf("\n下一步优化方向:\n");
    if (total_memcpy_time > total_time * 0.3) {
        printf("  1. 消除 CPU-GPU 传输（集成到 attention kernel）\n");
    }
    if (kernel_time > theoretical_time_ms * 10) {
        printf("  2. 优化 kernel 算法（使用 threadgroup memory）\n");
        printf("  3. 增加并行度（每个 threadgroup 更多 threads）\n");
    }

    // 清理
    free(src_fp16);
    free(dst_q8);
    free(scales);

    printf("\n✓ 分析完成！\n");
}

int main() {
    profile_quantization();
    return 0;
}
