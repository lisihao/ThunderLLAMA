#include "metal-buffer-pool.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <algorithm>

// ============================================================================
// Initialization
// ============================================================================

bool MetalBufferPool::init(id device, size_t pool_size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pool_buffer_ != nullptr) {
        fprintf(stderr, "[MetalBufferPool] Already initialized\n");
        return false;
    }

    pool_size_bytes_ = pool_size_bytes;

    // Allocate Metal buffer with shared storage mode (CPU/GPU both accessible)
    // M4 Pro unified memory: zero-copy access
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
    pool_buffer_ = (__bridge_retained id)[mtl_device newBufferWithLength:pool_size_bytes_
                                                                 options:MTLResourceStorageModeShared];

    if (pool_buffer_ == nullptr) {
        fprintf(stderr, "[MetalBufferPool] ERROR: Failed to allocate %zu bytes\n", pool_size_bytes_);
        return false;
    }

    fprintf(stderr, "[MetalBufferPool] Initialized: %.2f GB pool\n",
            pool_size_bytes_ / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "[MetalBufferPool] Storage mode: Shared (CPU/GPU unified memory)\n");

    return true;
}

void MetalBufferPool::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pool_buffer_ != nullptr) {
        CFRelease(pool_buffer_);
        pool_buffer_ = nullptr;
    }

    offset_map_.clear();
    lru_list_.clear();
    lru_index_.clear();
    free_offsets_.clear();

    next_offset_ = 0;
    used_bytes_ = 0;

    fprintf(stderr, "[MetalBufferPool] Cleaned up\n");
}

// ============================================================================
// Store
// ============================================================================

size_t MetalBufferPool::store(uint64_t key_hash,
                               const void* k_data, size_t k_size,
                               const void* v_data, size_t v_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pool_buffer_ == nullptr) {
        fprintf(stderr, "[MetalBufferPool] ERROR: Not initialized\n");
        return (size_t)-1;
    }

    size_t total_size = k_size + v_size;

    // Check if chunk already exists (update in place)
    auto it = offset_map_.find(key_hash);
    if (it != offset_map_.end()) {
        ChunkMetadata & meta = it->second;

        // If size matches, update in place
        if (meta.total_size == total_size) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)pool_buffer_;
            void* dst = (char*)[buffer contents] + meta.offset;

            memcpy(dst, k_data, k_size);
            memcpy((char*)dst + k_size, v_data, v_size);

            // Update LRU
            update_lru(key_hash);

            fprintf(stderr, "[MetalBufferPool] Updated in place: hash=%016llx, offset=%zu\n",
                    (unsigned long long)key_hash, meta.offset);

            return meta.offset;
        } else {
            // Size changed, need to reallocate
            // Free old space
            used_bytes_ -= meta.total_size;
            free_offsets_.push_back(meta.offset);

            // Remove from LRU
            auto lru_it = lru_index_.find(key_hash);
            if (lru_it != lru_index_.end()) {
                lru_list_.erase(lru_it->second);
                lru_index_.erase(lru_it);
            }

            // Remove from offset map
            offset_map_.erase(it);
        }
    }

    // Allocate space (may trigger eviction)
    size_t offset = allocate(total_size);
    if (offset == (size_t)-1) {
        fprintf(stderr, "[MetalBufferPool] ERROR: Allocation failed for %zu bytes\n", total_size);
        return (size_t)-1;
    }

    // Copy data to GPU buffer
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)pool_buffer_;
    void* dst = (char*)[buffer contents] + offset;

    memcpy(dst, k_data, k_size);
    memcpy((char*)dst + k_size, v_data, v_size);

    // Store metadata
    ChunkMetadata meta;
    meta.offset = offset;
    meta.k_size = k_size;
    meta.v_size = v_size;
    meta.total_size = total_size;
    offset_map_[key_hash] = meta;

    // Update LRU
    lru_list_.push_front(key_hash);
    lru_index_[key_hash] = lru_list_.begin();

    fprintf(stderr, "[MetalBufferPool] Stored: hash=%016llx, offset=%zu, size=%zu, utilization=%.1f%%\n",
            (unsigned long long)key_hash, offset, total_size, get_utilization() * 100.0);

    return offset;
}

// ============================================================================
// Get
// ============================================================================

size_t MetalBufferPool::get_offset(uint64_t key_hash) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = offset_map_.find(key_hash);
    if (it == offset_map_.end()) {
        return (size_t)-1;
    }

    // Update LRU (move to front)
    update_lru(key_hash);

    return it->second.offset;
}

bool MetalBufferPool::has(uint64_t key_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return offset_map_.find(key_hash) != offset_map_.end();
}

bool MetalBufferPool::blit_to_buffer(uint64_t key_hash,
                                      void* dst_buffer,
                                      size_t dst_offset,
                                      size_t chunk_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pool_buffer_ == nullptr || dst_buffer == nullptr) {
        fprintf(stderr, "[MetalBufferPool] ERROR: Invalid buffer in blit_to_buffer\n");
        return false;
    }

    // Check if chunk exists
    auto it = offset_map_.find(key_hash);
    if (it == offset_map_.end()) {
        return false;  // Chunk not in L1
    }

    ChunkMetadata & meta = it->second;
    size_t src_offset = meta.offset;

    // Verify size matches
    if (chunk_size > meta.k_size) {
        fprintf(stderr, "[MetalBufferPool] ERROR: Requested size %zu > cached k_size %zu\n",
                chunk_size, meta.k_size);
        return false;
    }

    @autoreleasepool {
        id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)pool_buffer_;
        // Perform __bridge cast here (only works in .mm files)
        id<MTLBuffer> dst_mtl_buffer = (__bridge id<MTLBuffer>)dst_buffer;

        // Get device and create command queue
        id<MTLDevice> device = [src_buffer device];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "[MetalBufferPool] ERROR: Failed to create command queue\n");
            return false;
        }

        // Create command buffer
        id<MTLCommandBuffer> cmd_buffer = [queue commandBuffer];
        if (!cmd_buffer) {
            fprintf(stderr, "[MetalBufferPool] ERROR: Failed to create command buffer\n");
            return false;
        }

        // Create blit command encoder
        id<MTLBlitCommandEncoder> blit_encoder = [cmd_buffer blitCommandEncoder];
        if (!blit_encoder) {
            fprintf(stderr, "[MetalBufferPool] ERROR: Failed to create blit encoder\n");
            return false;
        }

        // Execute GPU→GPU copy (Metal Blit)
        [blit_encoder copyFromBuffer:src_buffer
                        sourceOffset:src_offset
                            toBuffer:dst_mtl_buffer
                   destinationOffset:dst_offset
                                size:chunk_size];

        [blit_encoder endEncoding];

        // Commit and wait
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];

        // Update LRU (this chunk was accessed)
        update_lru(key_hash);

        fprintf(stderr, "[MetalBufferPool] Blit: hash=%016llx, src_offset=%zu, dst_offset=%zu, size=%zu\n",
                (unsigned long long)key_hash, src_offset, dst_offset, chunk_size);

        return true;
    }
}

// ============================================================================
// Allocation
// ============================================================================

size_t MetalBufferPool::allocate(size_t size) {
    // Try to reuse free slots first
    for (auto it = free_offsets_.begin(); it != free_offsets_.end(); ++it) {
        // Simple first-fit strategy
        // (In production, could use best-fit or buddy allocator)
        size_t offset = *it;
        free_offsets_.erase(it);
        used_bytes_ += size;
        return offset;
    }

    // Ensure we have space (evict if necessary)
    while (used_bytes_ + size > pool_size_bytes_) {
        if (!evict_lru()) {
            fprintf(stderr, "[MetalBufferPool] ERROR: Eviction failed, pool is full\n");
            return (size_t)-1;
        }
    }

    // Allocate from end
    size_t offset = next_offset_;
    next_offset_ += size;
    used_bytes_ += size;

    return offset;
}

// ============================================================================
// LRU Eviction
// ============================================================================

bool MetalBufferPool::evict_lru() {
    if (lru_list_.empty()) {
        fprintf(stderr, "[MetalBufferPool] ERROR: Cannot evict, pool is empty\n");
        return false;
    }

    // Get least recently used chunk (back of list)
    uint64_t victim_hash = lru_list_.back();
    lru_list_.pop_back();

    auto lru_it = lru_index_.find(victim_hash);
    if (lru_it != lru_index_.end()) {
        lru_index_.erase(lru_it);
    }

    // Get metadata
    auto it = offset_map_.find(victim_hash);
    if (it == offset_map_.end()) {
        fprintf(stderr, "[MetalBufferPool] ERROR: LRU inconsistency, chunk not found\n");
        return false;
    }

    ChunkMetadata meta = it->second;

    // Mark space as free
    free_offsets_.push_back(meta.offset);
    used_bytes_ -= meta.total_size;

    // Remove from map
    offset_map_.erase(it);

    fprintf(stderr, "[MetalBufferPool] Evicted: hash=%016llx, offset=%zu, freed=%zu bytes\n",
            (unsigned long long)victim_hash, meta.offset, meta.total_size);

    return true;
}

void MetalBufferPool::update_lru(uint64_t key_hash) {
    auto lru_it = lru_index_.find(key_hash);
    if (lru_it != lru_index_.end()) {
        // Move to front
        lru_list_.erase(lru_it->second);
    }

    lru_list_.push_front(key_hash);
    lru_index_[key_hash] = lru_list_.begin();
}

// ============================================================================
// C++ Factory Function
// ============================================================================

MetalBufferPool * create_metal_buffer_pool(size_t pool_size_bytes) {
    @autoreleasepool {
        // Get default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "[MetalBufferPool] ERROR: No Metal device available on this system\n");
            return nullptr;
        }

        fprintf(stderr, "[MetalBufferPool] Found Metal device: %s\n",
                [[device name] UTF8String]);

        // Create and initialize pool
        MetalBufferPool * pool = new MetalBufferPool();
        if (!pool->init((__bridge id)device, pool_size_bytes)) {
            delete pool;
            return nullptr;
        }

        return pool;
    }
}
