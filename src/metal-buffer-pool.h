#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <list>
#include <mutex>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Forward declarations for C++ (avoid importing Objective-C headers)
typedef struct objc_object * id;
#endif

/**
 * MetalBufferPool - GPU-side LMCache storage using Metal unified memory
 *
 * Purpose: Eliminate CPU→GPU transfer bottleneck for LMCache prefill skip
 *
 * Architecture:
 *   - Pre-allocate 10GB Metal buffer (MTLResourceStorageModeShared)
 *   - Store KV cache chunks in GPU memory
 *   - LRU eviction when pool is full
 *
 * Performance:
 *   - L1 hit: ~2ms (Metal blit, GPU-side copy)
 *   - L1 miss: fallback to L2/L3 (CPU/Disk)
 *   - Target speedup: 8x (16ms → 2ms)
 *
 * M4 Pro Unified Memory:
 *   - CPU and GPU share physical memory
 *   - Zero-copy access (CPU write, GPU read)
 *   - MTLResourceStorageModeShared enables both access
 */
class MetalBufferPool {
public:
    /**
     * Initialize the pool
     *
     * @param device Metal device
     * @param pool_size_bytes Total pool size (default: 10GB)
     * @return true if initialization succeeded
     */
    bool init(id device, size_t pool_size_bytes = 10ULL * 1024 * 1024 * 1024);

    /**
     * Store a KV cache chunk in GPU pool
     *
     * @param key_hash Hash of the chunk key
     * @param k_data Pointer to K tensor data
     * @param k_size Size of K tensor in bytes
     * @param v_data Pointer to V tensor data
     * @param v_size Size of V tensor in bytes
     * @return Offset in pool buffer, or (size_t)-1 if failed
     *
     * Note: If pool is full, will evict LRU chunk first
     */
    size_t store(uint64_t key_hash,
                 const void* k_data, size_t k_size,
                 const void* v_data, size_t v_size);

    /**
     * Get offset of a chunk in the pool
     *
     * @param key_hash Hash of the chunk key
     * @return Offset in pool buffer, or (size_t)-1 if not found
     *
     * Side effect: Updates LRU (moves to front)
     */
    size_t get_offset(uint64_t key_hash);

    /**
     * Check if a chunk exists in the pool
     *
     * @param key_hash Hash of the chunk key
     * @return true if chunk exists
     */
    bool has(uint64_t key_hash) const;

    /**
     * Get the Metal buffer (for Metal blit operations)
     *
     * @return Metal buffer object
     */
    id get_buffer() const { return pool_buffer_; }

    /**
     * Get pool statistics
     */
    size_t get_used_bytes() const { return used_bytes_; }
    size_t get_total_bytes() const { return pool_size_bytes_; }
    size_t get_chunk_count() const { return offset_map_.size(); }
    double get_utilization() const {
        return pool_size_bytes_ > 0 ? (double)used_bytes_ / pool_size_bytes_ : 0.0;
    }

    /**
     * Cleanup
     */
    void cleanup();

private:
    /**
     * Evict least recently used chunk
     *
     * @return true if evicted successfully
     *
     * Note: Evicted chunk is NOT copied back to L2/L3
     *       (caller should handle promotion/demotion)
     */
    bool evict_lru();

    /**
     * Update LRU (move to front)
     *
     * @param key_hash Hash of the chunk key
     */
    void update_lru(uint64_t key_hash);

    /**
     * Allocate space in the pool
     *
     * @param size Size to allocate
     * @return Offset, or (size_t)-1 if failed
     */
    size_t allocate(size_t size);

    // Metal buffer
    id pool_buffer_ = nullptr;
    size_t pool_size_bytes_ = 0;

    // Allocation tracking
    size_t next_offset_ = 0;
    size_t used_bytes_ = 0;
    std::vector<size_t> free_offsets_;  // Free slots from eviction

    // Chunk metadata
    struct ChunkMetadata {
        size_t offset;      // Offset in pool buffer
        size_t k_size;      // Size of K tensor
        size_t v_size;      // Size of V tensor
        size_t total_size;  // k_size + v_size
    };
    std::map<uint64_t, ChunkMetadata> offset_map_;  // key_hash -> metadata

    // LRU tracking
    std::list<uint64_t> lru_list_;  // key_hash, front = most recent
    std::map<uint64_t, std::list<uint64_t>::iterator> lru_index_;  // key_hash -> iterator

    // Thread safety
    mutable std::mutex mutex_;
};

/**
 * C++ factory function for creating MetalBufferPool
 *
 * This function provides a C++-compatible way to create and initialize
 * a MetalBufferPool without requiring Objective-C++ in the caller.
 *
 * @param pool_size_bytes Total pool size (default: 10GB)
 * @return Pointer to initialized MetalBufferPool, or nullptr if failed
 *
 * Note: Caller is responsible for calling cleanup() and deleting the pool
 */
MetalBufferPool * create_metal_buffer_pool(size_t pool_size_bytes = 10ULL * 1024 * 1024 * 1024);
