#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string>
#include "thunder-lmcache.h"
#include "thunder-lmcache-hash.h"

#ifdef __cplusplus

#include <unordered_map>
#include <list>
#include <mutex>

/**
 * @brief Multi-tier storage manager for KV cache chunks.
 *
 * Implements a two-tier caching system:
 * - L2 (CPU Heap): Fast in-memory storage with configurable limit (default 8GB)
 * - L3 (Disk mmap): Persistent storage with configurable limit (default 32GB)
 *
 * Features:
 * - LRU eviction policy for both tiers
 * - Automatic promotion of hot chunks from L3 to L2
 * - Thread-safe concurrent access
 * - Persistent storage across process restarts
 * - Hit rate tracking for cache efficiency monitoring
 *
 * Eviction strategy:
 * 1. When L2 is full: evict LRU chunk to L3
 * 2. When L3 is full: delete LRU chunk permanently
 * 3. On L3 hit: optionally promote to L2 (if space available)
 *
 * Thread safety: All public methods are protected by internal mutex.
 * Multiple threads can safely call put/get/evict concurrently.
 */
class ThunderChunkStorage {
public:
    /**
     * @brief Construct a new ThunderChunkStorage.
     *
     * Initializes the storage manager with configurable limits for CPU and disk tiers.
     * Creates the disk cache file if it doesn't exist, or loads existing cache.
     *
     * @param cpu_limit_bytes   Maximum bytes for L2 CPU heap storage (default 8GB).
     * @param disk_limit_bytes  Maximum bytes for L3 disk mmap storage (default 32GB).
     * @param disk_path         Path to disk cache file (default ~/.cache/thunderllama/kv_cache.bin).
     *                          Tilde (~) is expanded to user's home directory.
     *
     * @throws std::runtime_error if disk_path cannot be created or opened.
     * @throws std::bad_alloc if initial memory allocation fails.
     *
     * @note The disk file is created with read-write permissions (0644).
     * @note Existing disk cache is loaded on startup.
     */
    ThunderChunkStorage(
        size_t cpu_limit_bytes = 8ULL * 1024 * 1024 * 1024,   // 8GB
        size_t disk_limit_bytes = 32ULL * 1024 * 1024 * 1024, // 32GB
        const std::string & disk_path = "~/.cache/thunderllama/kv_cache.bin"
    );

    /**
     * @brief Destroy the ThunderChunkStorage.
     *
     * Unmaps the disk cache file and releases all resources.
     * L2 chunks are discarded, L3 chunks persist on disk.
     */
    ~ThunderChunkStorage();

    // Non-copyable, non-movable (due to mmap resource management)
    ThunderChunkStorage(const ThunderChunkStorage &) = delete;
    ThunderChunkStorage & operator=(const ThunderChunkStorage &) = delete;
    ThunderChunkStorage(ThunderChunkStorage &&) = delete;
    ThunderChunkStorage & operator=(ThunderChunkStorage &&) = delete;

    /**
     * @brief Store a KV chunk in the cache.
     *
     * Attempts to store the chunk in L2 (CPU heap).
     * If L2 is full, evicts LRU chunk to L3, then stores the new chunk.
     * If L3 is also full, evicts LRU chunk from L3 permanently.
     *
     * The chunk's k_data and v_data are deep-copied into the cache.
     * The original chunk's pointers can be freed after this call.
     *
     * @param chunk  The KV chunk to store. Must have valid k_data and v_data pointers.
     *
     * @return       true if successfully stored (in L2 or L3), false if storage failed.
     *
     * @note Thread-safe: can be called concurrently from multiple threads.
     * @note If a chunk with the same key already exists, it is overwritten.
     * @note The chunk's last_access_ns is updated to current time.
     */
    bool put(const thunder_kv_chunk & chunk);

    /**
     * @brief Retrieve a KV chunk from the cache.
     *
     * Searches for the chunk in L2 first, then L3.
     * - L2 hit: updates access time and returns pointer.
     * - L3 hit: optionally promotes to L2 (if space available), updates access time, returns pointer.
     * - Miss: returns nullptr.
     *
     * @param key  The unique key identifying the chunk.
     *
     * @return     Pointer to cached chunk if found (L2 or L3), nullptr if not found.
     *             The returned pointer remains valid until the chunk is evicted.
     *             The caller must NOT free the returned pointer.
     *
     * @note Thread-safe: can be called concurrently from multiple threads.
     * @note The returned pointer may be invalidated by concurrent eviction.
     *       Caller should use short critical sections.
     */
    thunder_kv_chunk * get(const thunder_kv_chunk_key & key);

    /**
     * @brief Evict chunks using LRU policy to free the specified amount of space.
     *
     * Evicts chunks from L2 to L3 (or from L3 permanently) until target_free_bytes is available.
     * Uses LRU policy: chunks with oldest last_access_ns are evicted first.
     *
     * @param target_free_bytes  Minimum bytes to free in L2.
     *
     * @note Thread-safe: can be called concurrently from multiple threads.
     * @note If target_free_bytes > current L2 usage, evicts all L2 chunks.
     * @note Evicted L2 chunks are moved to L3 if space available, otherwise deleted.
     */
    void evict_lru(size_t target_free_bytes);

    /**
     * @brief Get current L2 (CPU heap) usage in bytes.
     *
     * @return  Sum of k_size + v_size for all chunks in L2.
     *
     * @note Thread-safe.
     */
    size_t get_cpu_usage_bytes() const;

    /**
     * @brief Get current L3 (disk mmap) usage in bytes.
     *
     * @return  Sum of k_size + v_size for all chunks in L3.
     *
     * @note Thread-safe.
     */
    size_t get_disk_usage_bytes() const;

    /**
     * @brief Get cache hit rate (0.0 to 1.0).
     *
     * @return  hits / (hits + misses), or 0.0 if no accesses yet.
     *
     * @note Thread-safe.
     * @note Hit rate is cumulative since storage creation.
     */
    double get_hit_rate() const;

private:
    // ========================================================================
    // L2 (CPU Heap) Storage
    // ========================================================================

    // Map: key hash -> chunk
    std::unordered_map<uint64_t, thunder_kv_chunk> l2_cache_;

    // LRU queue for L2: most recently accessed at front
    std::list<thunder_kv_chunk_key> l2_lru_;

    // Map: key hash -> iterator into l2_lru_ (for O(1) LRU updates)
    std::unordered_map<uint64_t, std::list<thunder_kv_chunk_key>::iterator> l2_lru_index_;

    // Current L2 usage in bytes
    size_t l2_usage_bytes_ = 0;

    // L2 limit in bytes
    size_t l2_limit_bytes_;

    // ========================================================================
    // L3 (Disk mmap) Storage
    // ========================================================================

    // Map: key hash -> file offset
    std::unordered_map<uint64_t, size_t> l3_offsets_;

    // LRU queue for L3: most recently accessed at front
    std::list<thunder_kv_chunk_key> l3_lru_;

    // Map: key hash -> iterator into l3_lru_
    std::unordered_map<uint64_t, std::list<thunder_kv_chunk_key>::iterator> l3_lru_index_;

    // Current L3 usage in bytes
    size_t l3_usage_bytes_ = 0;

    // L3 limit in bytes
    size_t l3_limit_bytes_;

    // Disk file descriptor
    int disk_fd_ = -1;

    // Disk mmap pointer
    void * disk_mmap_ = nullptr;

    // Disk mmap size
    size_t disk_mmap_size_ = 0;

    // Next available offset in disk file
    size_t disk_next_offset_ = 0;

    // Disk file path
    std::string disk_path_;

    // ========================================================================
    // Statistics
    // ========================================================================

    // Total cache hits (L2 + L3)
    mutable uint64_t total_hits_ = 0;

    // Total cache misses
    mutable uint64_t total_misses_ = 0;

    // ========================================================================
    // Thread Safety
    // ========================================================================

    // Mutex protecting all data structures
    mutable std::mutex mutex_;

    // ========================================================================
    // Private Methods
    // ========================================================================

    /**
     * @brief Compute hash of a chunk key.
     *
     * @param key  The chunk key.
     * @return     Hash value (combines content_hash, layer_idx, chunk_start).
     */
    static uint64_t hash_key(const thunder_kv_chunk_key & key);

    /**
     * @brief Compute size of a chunk in bytes.
     *
     * @param chunk  The chunk.
     * @return       k_size + v_size.
     */
    static size_t chunk_size(const thunder_kv_chunk & chunk);

    /**
     * @brief Update LRU queue on access.
     *
     * Moves the chunk to the front of the LRU queue.
     * Assumes mutex_ is held.
     *
     * @param key        The chunk key.
     * @param is_l2      true for L2 queue, false for L3 queue.
     */
    void update_lru(const thunder_kv_chunk_key & key, bool is_l2);

    /**
     * @brief Evict one chunk from L2 to L3.
     *
     * Evicts the LRU chunk from L2, writes it to L3.
     * If L3 is full, evicts LRU chunk from L3 first.
     * Assumes mutex_ is held.
     *
     * @return  Number of bytes freed in L2.
     */
    size_t evict_one_l2_to_l3();

    /**
     * @brief Evict one chunk from L3 permanently.
     *
     * Evicts the LRU chunk from L3, frees disk space.
     * Assumes mutex_ is held.
     *
     * @return  Number of bytes freed in L3.
     */
    size_t evict_one_l3();

    /**
     * @brief Write a chunk to L3 disk.
     *
     * Allocates space in disk file, writes chunk data.
     * Assumes mutex_ is held.
     *
     * @param chunk  The chunk to write.
     * @return       File offset where chunk was written, or -1 on failure.
     */
    size_t write_to_disk(const thunder_kv_chunk & chunk);

    /**
     * @brief Read a chunk from L3 disk.
     *
     * Reads chunk data from disk at given offset.
     * Assumes mutex_ is held.
     *
     * @param offset  File offset to read from.
     * @param chunk   Output chunk (k_data and v_data are allocated and filled).
     * @return        true on success, false on failure.
     */
    bool read_from_disk(size_t offset, thunder_kv_chunk & chunk);

    /**
     * @brief Expand disk mmap if needed.
     *
     * Ensures disk_mmap_size_ >= required_size.
     * Assumes mutex_ is held.
     *
     * @param required_size  Minimum required size.
     * @return               true on success, false on failure.
     */
    bool ensure_disk_space(size_t required_size);

    /**
     * @brief Get current timestamp in nanoseconds.
     *
     * @return  Nanoseconds since epoch.
     */
    static uint64_t current_time_ns();

    /**
     * @brief Expand tilde (~) in path to home directory.
     *
     * @param path  Path potentially starting with ~.
     * @return      Expanded path.
     */
    static std::string expand_tilde(const std::string & path);
};

#endif // __cplusplus
