// ThunderLLAMA: Metal Performance Shaders integration
// Phase 1: FP16 GEMM via MPSMatrixMultiplication
//
// Strategy: Use MPSMatrixMultiplication (F16×F16→F16) with F32↔F16 conversion
// kernels for compatibility with ggml's F32 activation / F16 weight convention.
//
// Phase 2 will replace this with MPSGraph quantizedMatmul for Q4_K/Q5_K.

#import "ggml-metal-mps.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdlib>
#include <cstdio>

struct ggml_metal_mps_context {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;

    // F32↔F16 conversion compute pipelines (compiled once at init)
    id<MTLComputePipelineState> pipeline_f32_to_f16;
    id<MTLComputePipelineState> pipeline_f16_to_f32;

    // Cached temporary buffers (private GPU memory, grown as needed)
    id<MTLBuffer> temp_src1_f16;   // src1 after F32→F16 conversion
    size_t        temp_src1_f16_size;

    id<MTLBuffer> temp_dst_f16;    // MPS result before F16→F32 conversion
    size_t        temp_dst_f16_size;
};

// Metal shader source for type conversion kernels
static NSString * const kConvertShaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void f32_to_f16(device const float * src [[buffer(0)]],
                       device half        * dst [[buffer(1)]],
                       constant int       & n   [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid < (uint)n) {
        dst[tid] = half(src[tid]);
    }
}

kernel void f16_to_f32(device const half  * src [[buffer(0)]],
                       device float       * dst [[buffer(1)]],
                       constant int       & n   [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid < (uint)n) {
        dst[tid] = float(src[tid]);
    }
}
)";

// Compile a named function from the conversion shader source
static id<MTLComputePipelineState> compile_convert_pipeline(
        id<MTLDevice> device,
        NSString * func_name) {
    NSError * error = nil;

    MTLCompileOptions * options = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:kConvertShaderSource
                                              options:options
                                                error:&error];
    [options release];

    if (!lib) {
        fprintf(stderr, "[MPS] error: failed to compile conversion shaders: %s\n",
                [[error description] UTF8String]);
        return nil;
    }

    id<MTLFunction> func = [lib newFunctionWithName:func_name];
    [lib release];

    if (!func) {
        fprintf(stderr, "[MPS] error: function '%s' not found\n",
                [func_name UTF8String]);
        return nil;
    }

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func
                                                                            error:&error];
    [func release];

    if (!pso) {
        fprintf(stderr, "[MPS] error: failed to create pipeline for '%s': %s\n",
                [func_name UTF8String],
                [[error description] UTF8String]);
        return nil;
    }

    return pso;
}

// Ensure a temporary buffer is at least `required_size` bytes
static void ensure_temp_buffer(id<MTLDevice> device,
                               id<MTLBuffer> __strong * buf,
                               size_t * current_size,
                               size_t   required_size) {
    if (*current_size >= required_size && *buf != nil) {
        return;
    }

    if (*buf) {
        [*buf release];
    }

    // Round up to next 4KB page and add 20% headroom to avoid frequent reallocs
    size_t alloc_size = (size_t)((double)required_size * 1.2);
    alloc_size = (alloc_size + 4095) & ~(size_t)4095;

    *buf = [device newBufferWithLength:alloc_size
                               options:MTLResourceStorageModePrivate];
    *current_size = alloc_size;

    if (!*buf) {
        fprintf(stderr, "[MPS] error: failed to allocate temp buffer (%zu bytes)\n", alloc_size);
        *current_size = 0;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool ggml_metal_mps_available(void) {
    if (@available(macOS 13.0, *)) {
        return true;
    }
    return false;
}

ggml_metal_mps_ctx_t ggml_metal_mps_init(void * device_raw, void * queue_raw) {
    if (!ggml_metal_mps_available()) {
        fprintf(stderr, "[MPS] warning: MPS not available on this system\n");
        return NULL;
    }

    id<MTLDevice>       device = (__bridge id<MTLDevice>)device_raw;
    id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>)queue_raw;

    ggml_metal_mps_ctx_t ctx = (ggml_metal_mps_ctx_t)calloc(1, sizeof(struct ggml_metal_mps_context));
    if (!ctx) {
        return NULL;
    }

    ctx->device = [device retain];
    ctx->queue  = [queue  retain];

    // Compile F32↔F16 conversion pipelines
    ctx->pipeline_f32_to_f16 = compile_convert_pipeline(device, @"f32_to_f16");
    ctx->pipeline_f16_to_f32 = compile_convert_pipeline(device, @"f16_to_f32");

    if (!ctx->pipeline_f32_to_f16 || !ctx->pipeline_f16_to_f32) {
        fprintf(stderr, "[MPS] error: failed to compile conversion pipelines\n");
        ggml_metal_mps_free(ctx);
        return NULL;
    }

    // Pre-allocate temp buffers (will grow as needed)
    // Initial size: 4 MB each (enough for BS=1 decode with hidden_dim=4096)
    const size_t initial_size = 4 * 1024 * 1024;
    ensure_temp_buffer(device, &ctx->temp_src1_f16, &ctx->temp_src1_f16_size, initial_size);
    ensure_temp_buffer(device, &ctx->temp_dst_f16,  &ctx->temp_dst_f16_size,  initial_size);

    fprintf(stderr, "[MPS] initialized: conversion pipelines compiled, temp buffers allocated\n");

    return ctx;
}

void ggml_metal_mps_free(ggml_metal_mps_ctx_t ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->pipeline_f32_to_f16) { [ctx->pipeline_f32_to_f16 release]; }
    if (ctx->pipeline_f16_to_f32) { [ctx->pipeline_f16_to_f32 release]; }
    if (ctx->temp_src1_f16)       { [ctx->temp_src1_f16 release]; }
    if (ctx->temp_dst_f16)        { [ctx->temp_dst_f16  release]; }
    if (ctx->device)              { [ctx->device release]; }
    if (ctx->queue)               { [ctx->queue  release]; }

    free(ctx);
}

bool ggml_metal_mps_mul_mat_f16(
        ggml_metal_mps_ctx_t ctx,
        void * cmd_buf_raw,
        struct ggml_metal_buffer_id bid_a,
        struct ggml_metal_buffer_id bid_b,
        struct ggml_metal_buffer_id bid_c,
        int M, int N, int K,
        uint64_t row_bytes_a,
        uint64_t row_bytes_b,
        uint64_t row_bytes_c) {
    if (!ctx || !cmd_buf_raw) {
        return false;
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        return false;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = (__bridge id<MTLCommandBuffer>)cmd_buf_raw;

        id<MTLBuffer> mtl_a = (__bridge id<MTLBuffer>)bid_a.metal;
        id<MTLBuffer> mtl_b = (__bridge id<MTLBuffer>)bid_b.metal;
        id<MTLBuffer> mtl_c = (__bridge id<MTLBuffer>)bid_c.metal;

        const int n_elements_b = N * K;
        const int n_elements_c = M * N;

        // Step 1: Ensure temp buffers are large enough
        const size_t src1_f16_size = (size_t)n_elements_b * sizeof(uint16_t);
        const size_t dst_f16_size  = (size_t)n_elements_c * sizeof(uint16_t);

        ensure_temp_buffer(ctx->device, &ctx->temp_src1_f16, &ctx->temp_src1_f16_size, src1_f16_size);
        ensure_temp_buffer(ctx->device, &ctx->temp_dst_f16,  &ctx->temp_dst_f16_size,  dst_f16_size);

        if (!ctx->temp_src1_f16 || !ctx->temp_dst_f16) {
            fprintf(stderr, "[MPS] error: temp buffer allocation failed\n");
            return false;
        }

        // Step 2: Convert src1 from F32 to F16
        // Encode a compute pass for the conversion
        {
            id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
            [enc setComputePipelineState:ctx->pipeline_f32_to_f16];
            [enc setBuffer:mtl_b             offset:bid_b.offs atIndex:0];
            [enc setBuffer:ctx->temp_src1_f16 offset:0          atIndex:1];
            [enc setBytes:&n_elements_b       length:sizeof(int) atIndex:2];

            const int threads_per_tg = (int)ctx->pipeline_f32_to_f16.maxTotalThreadsPerThreadgroup;
            const int n_tg = (n_elements_b + threads_per_tg - 1) / threads_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_tg, 1, 1)];
            [enc endEncoding];
        }

        // Step 3: MPSMatrixMultiplication (F16 × F16 → F16)
        // C_f16 = A_f16 * B_f16^T
        {
            MPSMatrixDescriptor * desc_a = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M
                                 columns:K
                                rowBytes:(NSUInteger)row_bytes_a
                                dataType:MPSDataTypeFloat16];

            MPSMatrixDescriptor * desc_b = [MPSMatrixDescriptor
                matrixDescriptorWithRows:N
                                 columns:K
                                rowBytes:(NSUInteger)(K * sizeof(uint16_t))  // temp buffer is contiguous
                                dataType:MPSDataTypeFloat16];

            MPSMatrixDescriptor * desc_c = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M
                                 columns:N
                                rowBytes:(NSUInteger)(N * sizeof(uint16_t))  // temp buffer is contiguous
                                dataType:MPSDataTypeFloat16];

            MPSMatrix * mat_a = [[MPSMatrix alloc] initWithBuffer:mtl_a
                                                           offset:(NSUInteger)bid_a.offs
                                                       descriptor:desc_a];

            MPSMatrix * mat_b = [[MPSMatrix alloc] initWithBuffer:ctx->temp_src1_f16
                                                           offset:0
                                                       descriptor:desc_b];

            MPSMatrix * mat_c = [[MPSMatrix alloc] initWithBuffer:ctx->temp_dst_f16
                                                           offset:0
                                                       descriptor:desc_c];

            MPSMatrixMultiplication * mm = [[MPSMatrixMultiplication alloc]
                initWithDevice:ctx->device
                 transposeLeft:false
                transposeRight:true
                    resultRows:M
                 resultColumns:N
               interiorColumns:K
                         alpha:1.0
                          beta:0.0];

            [mm encodeToCommandBuffer:cmd_buf
                           leftMatrix:mat_a
                          rightMatrix:mat_b
                         resultMatrix:mat_c];

            [mm    release];
            [mat_a release];
            [mat_b release];
            [mat_c release];
        }

        // Step 4: Convert result from F16 to F32 into dst
        {
            id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
            [enc setComputePipelineState:ctx->pipeline_f16_to_f32];
            [enc setBuffer:ctx->temp_dst_f16 offset:0          atIndex:0];
            [enc setBuffer:mtl_c             offset:bid_c.offs atIndex:1];
            [enc setBytes:&n_elements_c      length:sizeof(int) atIndex:2];

            const int threads_per_tg = (int)ctx->pipeline_f16_to_f32.maxTotalThreadsPerThreadgroup;
            const int n_tg = (n_elements_c + threads_per_tg - 1) / threads_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_tg, 1, 1)];
            [enc endEncoding];
        }

        return true;
    }
}
