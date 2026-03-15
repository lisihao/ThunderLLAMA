#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#include <mach/mach_time.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_compute = 0;
    res->capture_started = false;
    res->capture_scope = nil;

    {
        const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
        if (val) {
            res->capture_compute = atoi(val);
        }
    }

    res->has_error = false;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}

// ============================================================================
// ThunderLLAMA KV Cache Quantization (GPU-side, 对标 MLX mx.quantize)
// ============================================================================

// 测试专用：从 CPU 内存创建 Metal buffers
void ggml_metal_quantize_kv_cache_q8_cpu(
        ggml_metal_device_t dev,
        const ggml_fp16_t * src_cpu,   // CPU FP16 input
        int8_t            * dst_cpu,   // CPU INT8 output
        ggml_fp16_t       * scales_cpu,// CPU FP16 scales
        int                 ne00,
        int                 group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    int n_groups = (ne00 + group_size - 1) / group_size;

    // Create Metal buffers
    id<MTLBuffer> src_buf = [device newBufferWithLength:ne00 * sizeof(ggml_fp16_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst_buf = [device newBufferWithLength:ne00 * sizeof(int8_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales_buf = [device newBufferWithLength:n_groups * sizeof(ggml_fp16_t)
                                                    options:MTLResourceStorageModeShared];

    // Copy input to GPU
    memcpy([src_buf contents], src_cpu, ne00 * sizeof(ggml_fp16_t));

    // Execute GPU quantization kernel
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_quantize_kv_cache_q8"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_quantize_kv_cache_q8 function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create quantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buf    offset:0 atIndex:0];
    [encoder setBuffer:dst_buf    offset:0 atIndex:1];
    [encoder setBuffer:scales_buf offset:0 atIndex:2];
    [encoder setBytes:&ne00       length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size length:sizeof(int) atIndex:4];

    int n_threadgroups = n_groups;
    MTLSize threadgroups = MTLSizeMake(n_threadgroups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    // Copy output back to CPU
    memcpy(dst_cpu, [dst_buf contents], ne00 * sizeof(int8_t));
    memcpy(scales_cpu, [scales_buf contents], n_groups * sizeof(ggml_fp16_t));
}

void ggml_metal_dequantize_kv_cache_q8_cpu(
        ggml_metal_device_t dev,
        const int8_t      * src_cpu,
        const ggml_fp16_t * scales_cpu,
        ggml_fp16_t       * dst_cpu,
        int                 ne00,
        int                 group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    int n_groups = (ne00 + group_size - 1) / group_size;

    // Create Metal buffers
    id<MTLBuffer> src_buf = [device newBufferWithLength:ne00 * sizeof(int8_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales_buf = [device newBufferWithLength:n_groups * sizeof(ggml_fp16_t)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst_buf = [device newBufferWithLength:ne00 * sizeof(ggml_fp16_t)
                                                 options:MTLResourceStorageModeShared];

    // Copy input to GPU
    memcpy([src_buf contents], src_cpu, ne00 * sizeof(int8_t));
    memcpy([scales_buf contents], scales_cpu, n_groups * sizeof(ggml_fp16_t));

    // Execute GPU dequantization kernel
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_dequantize_kv_cache_q8"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_dequantize_kv_cache_q8 function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create dequantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buf    offset:0 atIndex:0];
    [encoder setBuffer:scales_buf offset:0 atIndex:1];
    [encoder setBuffer:dst_buf    offset:0 atIndex:2];
    [encoder setBytes:&ne00       length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size length:sizeof(int) atIndex:4];

    // Changed to threadgroup-level dispatch (like quantization kernel)
    // Each threadgroup processes one quantization group
    int threads_per_threadgroup = 32;  // One simdgroup

    MTLSize threadgroups = MTLSizeMake(n_groups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threads_per_threadgroup, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    // Copy output back to CPU
    memcpy(dst_cpu, [dst_buf contents], ne00 * sizeof(ggml_fp16_t));
}

// ============================================================================

void ggml_metal_quantize_kv_cache_q8(
        ggml_metal_device_t dev,
        const void * src,      // FP16 input
        void       * dst,      // INT8 output
        void       * scales,   // FP16 scales (1 per group)
        int          ne00,     // Total elements
        int          group_size) {

    // Get Metal objects
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    // Create command buffer
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    // Get quantization kernel
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_quantize_kv_cache_q8"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_quantize_kv_cache_q8 function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create quantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)src    offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)dst    offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scales offset:0 atIndex:2];
    [encoder setBytes:&ne00       length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size length:sizeof(int) atIndex:4];

    // Dispatch
    int n_groups = (ne00 + group_size - 1) / group_size;
    MTLSize threadgroups = MTLSizeMake(n_groups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);  // 1 simdgroup

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

void ggml_metal_dequantize_kv_cache_q8(
        ggml_metal_device_t dev,
        const void * src,      // INT8 input
        const void * scales,   // FP16 scales
        void       * dst,      // FP16 output
        int          ne00,
        int          group_size) {

    // Get Metal objects
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    // Create command buffer
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    // Get dequantization kernel
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_dequantize_kv_cache_q8"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_dequantize_kv_cache_q8 function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create dequantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)src    offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)scales offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)dst    offset:0 atIndex:2];
    [encoder setBytes:&ne00       length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size length:sizeof(int) atIndex:4];

    // Dispatch
    int n_threads = ne00;
    int threads_per_threadgroup = 256;
    int n_threadgroups = (n_threads + threads_per_threadgroup - 1) / threads_per_threadgroup;

    MTLSize threadgroups = MTLSizeMake(n_threadgroups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threads_per_threadgroup, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

// ============================================================================
// High-Performance Version (v2) with Threadgroup Memory
// ============================================================================

void ggml_metal_quantize_kv_cache_q8_v2(
        ggml_metal_device_t dev,
        const void * src,      // FP16 input
        void       * dst,      // INT8 output
        void       * scales,   // FP16 scales (1 per group)
        int          ne00,     // Total elements
        int          group_size) {

    // Get Metal objects
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    // Create command buffer
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    // Get v2 quantization kernel
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_quantize_kv_cache_q8_v2"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_quantize_kv_cache_q8_v2 function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create v2 quantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)src    offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)dst    offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scales offset:0 atIndex:2];
    [encoder setBytes:&ne00       length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size length:sizeof(int) atIndex:4];

    // Dispatch with 128 threads per threadgroup
    // Each threadgroup processes 1 quantization group (64 elements)
    int n_groups = (ne00 + group_size - 1) / group_size;

    MTLSize threadgroups = MTLSizeMake(n_groups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);  // 4 simdgroups

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

// Test wrapper for v2 (CPU-side)
void ggml_metal_quantize_kv_cache_q8_v2_cpu(
        ggml_metal_device_t dev,
        const ggml_fp16_t * src_cpu,
        int8_t            * dst_cpu,
        ggml_fp16_t       * scales_cpu,
        int                 ne00,
        int                 group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    int n_groups = (ne00 + group_size - 1) / group_size;

    // Create Metal buffers
    id<MTLBuffer> src_buf = [device newBufferWithLength:ne00 * sizeof(ggml_fp16_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst_buf = [device newBufferWithLength:ne00 * sizeof(int8_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales_buf = [device newBufferWithLength:n_groups * sizeof(ggml_fp16_t)
                                                    options:MTLResourceStorageModeShared];

    // Copy input to GPU
    memcpy([src_buf contents], src_cpu, ne00 * sizeof(ggml_fp16_t));

    // Execute GPU quantization kernel (v2)
    ggml_metal_quantize_kv_cache_q8_v2(dev, src_buf, dst_buf, scales_buf, ne00, group_size);

    // Copy output back to CPU
    memcpy(dst_cpu, [dst_buf contents], ne00 * sizeof(int8_t));
    memcpy(scales_cpu, [scales_buf contents], n_groups * sizeof(ggml_fp16_t));
}

// ============================================================================
// Batch Quantization (方案 B: Reduce Kernel Launch Overhead)
// ============================================================================

void ggml_metal_quantize_kv_cache_q8_batch(
        ggml_metal_device_t dev,
        const void * src,      // FP16 input (all layers concatenated)
        void       * dst,      // INT8 output
        void       * scales,   // FP16 scales
        int          total_elements,
        int          group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_quantize_kv_cache_q8_batch"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get batch quantization kernel\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create batch pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)src    offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)dst    offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scales offset:0 atIndex:2];
    [encoder setBytes:&total_elements length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size     length:sizeof(int) atIndex:4];

    int n_groups = (total_elements + group_size - 1) / group_size;
    MTLSize threadgroups = MTLSizeMake(n_groups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

void ggml_metal_dequantize_kv_cache_q8_batch(
        ggml_metal_device_t dev,
        const void * src,
        const void * scales,
        void       * dst,
        int          total_elements,
        int          group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_dequantize_kv_cache_q8_batch"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get batch dequantization kernel\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create batch deq pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)src    offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)scales offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)dst    offset:0 atIndex:2];
    [encoder setBytes:&total_elements length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size     length:sizeof(int) atIndex:4];

    int n_groups = (total_elements + group_size - 1) / group_size;
    MTLSize threadgroups = MTLSizeMake(n_groups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

// ============================================================================
// Offline Quantization (方案 C: Quantize Once, Use Many Times)
// ============================================================================

void ggml_metal_kv_cache_quantize_offline(
        ggml_metal_device_t dev,
        void       * kv_cache,     // KV cache buffer
        int          n_layers,
        int          hidden_dim,
        int          seq_len,
        int          group_size) {

    // Calculate total elements
    int elements_per_layer = hidden_dim * seq_len;
    int total_elements = n_layers * elements_per_layer * 2;  // K + V

    fprintf(stderr, "Offline KV Cache Quantization:\n");
    fprintf(stderr, "  Layers: %d\n", n_layers);
    fprintf(stderr, "  Hidden Dim: %d\n", hidden_dim);
    fprintf(stderr, "  Seq Len: %d\n", seq_len);
    fprintf(stderr, "  Total Elements: %d (%.2f MB FP16)\n",
            total_elements, total_elements * 2.0 / 1024 / 1024);

    // Use batch quantization
    int n_groups = (total_elements + group_size - 1) / group_size;

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    id<MTLBuffer> dst_buf = [device newBufferWithLength:total_elements * sizeof(int8_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales_buf = [device newBufferWithLength:n_groups * sizeof(ggml_fp16_t)
                                                    options:MTLResourceStorageModeShared];

    // Timing using mach_absolute_time (Objective-C compatible)
    uint64_t start = mach_absolute_time();

    ggml_metal_quantize_kv_cache_q8_batch(dev, kv_cache, dst_buf, scales_buf,
                                          total_elements, group_size);

    uint64_t end = mach_absolute_time();

    // Convert to milliseconds
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t elapsed_ns = (end - start) * timebase.numer / timebase.denom;
    double elapsed_ms = elapsed_ns / 1e6;

    double data_mb = total_elements * 2.0 / 1024 / 1024;
    double throughput_gb_s = data_mb / elapsed_ms;

    fprintf(stderr, "Offline Quantization Complete:\n");
    fprintf(stderr, "  Time: %.3f ms\n", elapsed_ms);
    fprintf(stderr, "  Throughput: %.2f GB/s\n", throughput_gb_s);
    fprintf(stderr, "  Memory Reduction: %.1fx (%.2f MB → %.2f MB)\n",
            (float)(total_elements * 2) / (total_elements + n_groups * 2),
            data_mb,
            (total_elements + n_groups * 2.0) / 1024 / 1024);
}

// Batch quantization wrapper (CPU interface for testing)
void ggml_metal_quantize_kv_cache_q8_batch_cpu(
        ggml_metal_device_t dev,
        const ggml_fp16_t * src_cpu,
        int8_t            * dst_cpu,
        ggml_fp16_t       * scales_cpu,
        int                 total_elements,
        int                 group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    int n_groups = (total_elements + group_size - 1) / group_size;

    // Create Metal buffers
    id<MTLBuffer> src_buf = [device newBufferWithLength:total_elements * sizeof(ggml_fp16_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst_buf = [device newBufferWithLength:total_elements * sizeof(int8_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales_buf = [device newBufferWithLength:n_groups * sizeof(ggml_fp16_t)
                                                    options:MTLResourceStorageModeShared];

    // Copy input to GPU
    memcpy([src_buf contents], src_cpu, total_elements * sizeof(ggml_fp16_t));

    // Execute GPU batch quantization kernel
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_quantize_kv_cache_q8_batch"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_quantize_kv_cache_q8_batch function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create batch quantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buf    offset:0 atIndex:0];
    [encoder setBuffer:dst_buf    offset:0 atIndex:1];
    [encoder setBuffer:scales_buf offset:0 atIndex:2];
    [encoder setBytes:&total_elements length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size     length:sizeof(int) atIndex:4];

    int n_threadgroups = n_groups;
    MTLSize threadgroups = MTLSizeMake(n_threadgroups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    // Copy output back to CPU
    memcpy(dst_cpu, [dst_buf contents], total_elements * sizeof(int8_t));
    memcpy(scales_cpu, [scales_buf contents], n_groups * sizeof(ggml_fp16_t));
}

// Batch dequantization wrapper (CPU interface for testing)
void ggml_metal_dequantize_kv_cache_q8_batch_cpu(
        ggml_metal_device_t dev,
        const int8_t      * src_cpu,
        const ggml_fp16_t * scales_cpu,
        ggml_fp16_t       * dst_cpu,
        int                 total_elements,
        int                 group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    int n_groups = (total_elements + group_size - 1) / group_size;

    // Create Metal buffers
    id<MTLBuffer> src_buf = [device newBufferWithLength:total_elements * sizeof(int8_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scales_buf = [device newBufferWithLength:n_groups * sizeof(ggml_fp16_t)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst_buf = [device newBufferWithLength:total_elements * sizeof(ggml_fp16_t)
                                                 options:MTLResourceStorageModeShared];

    // Copy input to GPU
    memcpy([src_buf contents], src_cpu, total_elements * sizeof(int8_t));
    memcpy([scales_buf contents], scales_cpu, n_groups * sizeof(ggml_fp16_t));

    // Execute GPU batch dequantization kernel
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_dequantize_kv_cache_q8_batch"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_dequantize_kv_cache_q8_batch function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create batch dequantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buf    offset:0 atIndex:0];
    [encoder setBuffer:scales_buf offset:0 atIndex:1];
    [encoder setBuffer:dst_buf    offset:0 atIndex:2];
    [encoder setBytes:&total_elements length:sizeof(int) atIndex:3];
    [encoder setBytes:&group_size     length:sizeof(int) atIndex:4];

    int n_threadgroups = n_groups;
    MTLSize threadgroups = MTLSizeMake(n_threadgroups, 1, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    // Copy output back to CPU
    memcpy(dst_cpu, [dst_buf contents], total_elements * sizeof(ggml_fp16_t));
}

// Offline quantization wrapper (CPU interface for testing)
void ggml_metal_kv_cache_quantize_offline_cpu(
        ggml_metal_device_t dev,
        ggml_fp16_t       * kv_cache,
        int                 n_layers,
        int                 hidden_dim,
        int                 seq_len,
        int                 group_size) {

    int elements_per_layer = hidden_dim * seq_len;
    int total_elements = n_layers * elements_per_layer * 2;
    int n_groups = (total_elements + group_size - 1) / group_size;

    fprintf(stderr, "Offline KV Cache Quantization:\n");
    fprintf(stderr, "  Layers: %d\n", n_layers);
    fprintf(stderr, "  Hidden Dim: %d\n", hidden_dim);
    fprintf(stderr, "  Seq Len: %d\n", seq_len);
    fprintf(stderr, "  Total Elements: %d (%.2f MB FP16)\n",
            total_elements, total_elements * 2.0 / 1024 / 1024);

    // Use batch quantization
    int8_t * dst_buf_cpu = (int8_t *)malloc(total_elements * sizeof(int8_t));
    ggml_fp16_t * scales_cpu = (ggml_fp16_t *)malloc(n_groups * sizeof(ggml_fp16_t));

    // Timing
    uint64_t start = mach_absolute_time();

    ggml_metal_quantize_kv_cache_q8_batch_cpu(dev, kv_cache, dst_buf_cpu, scales_cpu,
                                              total_elements, group_size);

    uint64_t end = mach_absolute_time();

    // Convert to milliseconds
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t elapsed_ns = (end - start) * timebase.numer / timebase.denom;
    double elapsed_ms = elapsed_ns / 1e6;

    double data_mb = total_elements * 2.0 / 1024 / 1024;
    double throughput_gb_s = data_mb / elapsed_ms;

    fprintf(stderr, "Offline Quantization Complete:\n");
    fprintf(stderr, "  Time: %.3f ms\n", elapsed_ms);
    fprintf(stderr, "  Throughput: %.2f GB/s\n", throughput_gb_s);
    fprintf(stderr, "  Memory Reduction: %.1fx (%.2f MB → %.2f MB)\n",
            (double)(total_elements * 2) / (total_elements + n_groups * 2),
            data_mb,
            (total_elements + n_groups * 2.0) / 1024 / 1024);

    free(dst_buf_cpu);
    free(scales_cpu);
}

// ============================================================================
// CPU-GPU Pipeline Quantization (Zero-Copy + Pipeline)
// ============================================================================

// Include CPU NEON header
#include "ggml-metal-cpu-neon.h"

// Pipeline quantization: CPU computes scales (NEON), GPU quantizes (Metal)
// Overlaps CPU and GPU work to hide kernel launch overhead
void ggml_metal_quantize_kv_cache_q8_pipeline_cpu(
        ggml_metal_device_t dev,
        const ggml_fp16_t * src_cpu,
        int8_t            * dst_cpu,
        ggml_fp16_t       * scales_cpu,
        int                 n_layers,
        int                 elements_per_layer,
        int                 group_size) {

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    int groups_per_layer = (elements_per_layer + group_size - 1) / group_size;
    int total_elements = n_layers * elements_per_layer;
    int total_groups = n_layers * groups_per_layer;

    fprintf(stderr, "CPU-GPU Pipeline Quantization:\n");
    fprintf(stderr, "  Layers: %d\n", n_layers);
    fprintf(stderr, "  Elements per Layer: %d (%.2f MB)\n",
            elements_per_layer, elements_per_layer * 2.0 / 1024 / 1024);
    fprintf(stderr, "  Total Elements: %d (%.2f MB)\n",
            total_elements, total_elements * 2.0 / 1024 / 1024);

    // Zero-Copy buffers (UMA shared memory)
    id<MTLBuffer> src_buf = [device 
        newBufferWithBytesNoCopy:(void*)src_cpu
        length:total_elements * sizeof(ggml_fp16_t)
        options:MTLResourceStorageModeShared
        deallocator:nil];
    
    id<MTLBuffer> dst_buf = [device 
        newBufferWithBytesNoCopy:dst_cpu
        length:total_elements * sizeof(int8_t)
        options:MTLResourceStorageModeShared
        deallocator:nil];
    
    id<MTLBuffer> scales_buf = [device 
        newBufferWithBytesNoCopy:scales_cpu
        length:total_groups * sizeof(ggml_fp16_t)
        options:MTLResourceStorageModeShared
        deallocator:nil];

    // Get Metal resources
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    id<MTLLibrary> library = ggml_metal_library_get_obj(lib);

    // Prepare pipeline kernel
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:@"kernel_quantize_kv_cache_q8_pipeline"];
    if (!function) {
        GGML_LOG_ERROR("%s: failed to get kernel_quantize_kv_cache_q8_pipeline function\n", __func__);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        GGML_LOG_ERROR("%s: failed to create pipeline quantization pipeline: %s\n",
                       __func__, [[error localizedDescription] UTF8String]);
        return;
    }

    // Timing
    uint64_t start_time = mach_absolute_time();

    // Pipeline: CPU computes scales for layer N, GPU quantizes layer N-1
    for (int layer = 0; layer < n_layers; layer++) {
        int offset = layer * elements_per_layer;
        int scale_offset = layer * groups_per_layer;

        // Stage 1: CPU computes scales for current layer (NEON accelerated)
        ggml_metal_cpu_compute_scales_neon(
            src_cpu + offset,
            scales_cpu + scale_offset,
            groups_per_layer,
            group_size
        );

        // Stage 2: GPU quantizes previous layer (async, overlapped with CPU)
        if (layer > 0) {
            int prev_offset = (layer - 1) * elements_per_layer;
            int prev_scale_offset = (layer - 1) * groups_per_layer;

            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:src_buf offset:prev_offset * sizeof(ggml_fp16_t) atIndex:0];
            [encoder setBuffer:dst_buf offset:prev_offset * sizeof(int8_t) atIndex:1];
            [encoder setBuffer:scales_buf offset:prev_scale_offset * sizeof(ggml_fp16_t) atIndex:2];
            [encoder setBytes:&elements_per_layer length:sizeof(int) atIndex:3];
            [encoder setBytes:&group_size length:sizeof(int) atIndex:4];

            MTLSize threadgroups = MTLSizeMake(groups_per_layer, 1, 1);
            MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);

            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];

            [command_buffer commit];
            // Don't wait - let it run async (pipeline overlap!)
        }
    }

    // Process last layer (synchronous)
    {
        int last_offset = (n_layers - 1) * elements_per_layer;
        int last_scale_offset = (n_layers - 1) * groups_per_layer;

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src_buf offset:last_offset * sizeof(ggml_fp16_t) atIndex:0];
        [encoder setBuffer:dst_buf offset:last_offset * sizeof(int8_t) atIndex:1];
        [encoder setBuffer:scales_buf offset:last_scale_offset * sizeof(ggml_fp16_t) atIndex:2];
        [encoder setBytes:&elements_per_layer length:sizeof(int) atIndex:3];
        [encoder setBytes:&group_size length:sizeof(int) atIndex:4];

        MTLSize threadgroups = MTLSizeMake(groups_per_layer, 1, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);

        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];  // Wait for last layer
    }

    // All GPU work is complete (last command buffer waited)
    uint64_t end_time = mach_absolute_time();

    // Convert to milliseconds
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t elapsed_ns = (end_time - start_time) * timebase.numer / timebase.denom;
    double elapsed_ms = elapsed_ns / 1e6;

    double data_mb = total_elements * 2.0 / 1024 / 1024;
    double throughput_gb_s = data_mb / elapsed_ms;

    fprintf(stderr, "Pipeline Quantization Complete:\n");
    fprintf(stderr, "  Total Time: %.3f ms\n", elapsed_ms);
    fprintf(stderr, "  Throughput: %.2f GB/s\n", throughput_gb_s);
    fprintf(stderr, "  Time per Layer: %.3f ms\n", elapsed_ms / n_layers);
}
