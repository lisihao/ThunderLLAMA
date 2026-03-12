#include "thunder-lmcache-storage.h"

#include <cstring>
#include <chrono>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pwd.h>
#include <algorithm>
#include <zlib.h>
#include <thread>
#include <future>
#include <vector>

// XXH64 for checksum (same as used in thunder-lmcache-hash.cpp)
#include "../examples/gguf-hash/deps/xxhash/xxhash.h"

// ============================================================================
// Constructor / Destructor
// ============================================================================

ThunderChunkStorage::ThunderChunkStorage(
    size_t cpu_limit_bytes,
    size_t disk_limit_bytes,
    const std::string & disk_path
)
    : l2_limit_bytes_(cpu_limit_bytes),
      l3_limit_bytes_(disk_limit_bytes),
      disk_path_([&disk_path]() {
          const char * env_path = getenv("THUNDER_LMCACHE_DISK_PATH");
          return expand_tilde(env_path ? env_path : disk_path);
      }())
{
    fprintf(stderr, "[ThunderChunkStorage] Using disk path: %s\n", disk_path_.c_str());

    // Attempt to initialize L3 disk storage
    // If this fails (disk unavailable/damaged), gracefully degrade to memory-only (L2) mode
    bool disk_init_success = false;

    do {
        // Create parent directory if not exists
        size_t last_slash = disk_path_.rfind('/');
        if (last_slash != std::string::npos) {
            std::string dir = disk_path_.substr(0, last_slash);
            // Use mkdir -p to create parent directories
            std::string mkdir_cmd = "mkdir -p \"" + dir + "\"";
            if (system(mkdir_cmd.c_str()) != 0) {
                fprintf(stderr, "[ThunderChunkStorage] WARNING: Failed to create cache directory: %s\n", dir.c_str());
                break;
            }
        }

        // Open or create disk file
        disk_fd_ = open(disk_path_.c_str(), O_RDWR | O_CREAT, 0644);
        if (disk_fd_ < 0) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Failed to open disk cache file: %s (errno=%d)\n",
                    disk_path_.c_str(), errno);
            break;
        }

        // Get current file size
        struct stat st;
        if (fstat(disk_fd_, &st) < 0) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Failed to stat disk cache file\n");
            close(disk_fd_);
            disk_fd_ = -1;
            break;
        }

        // If file is new or small, expand to initial size (1MB)
        size_t initial_size = 1024 * 1024; // 1MB
        if (static_cast<size_t>(st.st_size) < initial_size) {
            if (ftruncate(disk_fd_, initial_size) < 0) {
                fprintf(stderr, "[ThunderChunkStorage] WARNING: Failed to resize disk cache file\n");
                close(disk_fd_);
                disk_fd_ = -1;
                break;
            }
            disk_mmap_size_ = initial_size;
        } else {
            disk_mmap_size_ = st.st_size;
        }

        // mmap the file
        disk_mmap_ = mmap(nullptr, disk_mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, disk_fd_, 0);
        if (disk_mmap_ == MAP_FAILED) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Failed to mmap disk cache file\n");
            close(disk_fd_);
            disk_fd_ = -1;
            disk_mmap_ = nullptr;
            break;
        }

        // Test write to verify disk is healthy
        if (!check_disk_health()) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Disk health check failed\n");
            munmap(disk_mmap_, disk_mmap_size_);
            close(disk_fd_);
            disk_fd_ = -1;
            disk_mmap_ = nullptr;
            break;
        }

        // Load existing cache from disk
        load_cache_from_disk();

        disk_init_success = true;
    } while (false);

    if (!disk_init_success) {
        fprintf(stderr, "[ThunderChunkStorage] ⚠️  L3 disk cache DISABLED - running in memory-only mode\n");
        fprintf(stderr, "[ThunderChunkStorage] ℹ️  L2 cache (8GB memory) will continue to work\n");
        l3_enabled_ = false;
    } else {
        fprintf(stderr, "[ThunderChunkStorage] ✓ L3 disk cache initialized successfully\n");

        // Prefetch hot chunks from L3 to L2 (smart prefetch based on access frequency)
        // Note: prefetch_hot_chunks needs mutex, but we're already in constructor's scope
        // We'll call it without locking since constructor is single-threaded
        if (!access_freq_.empty()) {
            // Temporarily release and re-acquire to call prefetch safely
            std::lock_guard<std::mutex> lock(mutex_);
            prefetch_hot_chunks(100);
        }
    }
}

ThunderChunkStorage::~ThunderChunkStorage() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Persist all L2 chunks to L3 before shutdown
    fprintf(stderr, "[ThunderChunkStorage] Persisting %zu L2 chunks to disk...\n", l2_cache_.size());
    int persisted = 0;
    for (auto & [hash, chunk] : l2_cache_) {
        // Write to L3 (disk)
        size_t offset = write_to_disk(chunk);
        if (offset != static_cast<size_t>(-1)) {
            // Successfully written to L3
            thunder_kv_chunk_key key = chunk.key;
            l3_offsets_[hash] = offset;
            l3_usage_bytes_ += chunk_size(chunk);

            // Add to L3 LRU
            l3_lru_.push_front(key);
            l3_lru_index_[hash] = l3_lru_.begin();

            persisted++;
        }

        // Free L2 data
        if (chunk.k_data) {
            free(chunk.k_data);
            chunk.k_data = nullptr;
        }
        if (chunk.v_data) {
            free(chunk.v_data);
            chunk.v_data = nullptr;
        }
    }
    fprintf(stderr, "[ThunderChunkStorage] Persisted %d chunks to disk\n", persisted);

    // Sync all pending writes to disk before unmapping
    if (disk_mmap_ != nullptr && disk_mmap_ != MAP_FAILED) {
        fprintf(stderr, "[ThunderChunkStorage] Syncing cache to disk before shutdown...\n");
        if (msync(disk_mmap_, disk_mmap_size_, MS_SYNC) < 0) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Final msync failed\n");
        } else {
            fprintf(stderr, "[ThunderChunkStorage] Cache synced successfully\n");
        }

        munmap(disk_mmap_, disk_mmap_size_);
        disk_mmap_ = nullptr;
    }

    // Close disk file
    if (disk_fd_ >= 0) {
        close(disk_fd_);
        disk_fd_ = -1;
    }
}

// ============================================================================
// Public API
// ============================================================================

bool ThunderChunkStorage::put(const thunder_kv_chunk & chunk) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t key_hash = hash_key(chunk.key);
    size_t size = chunk_size(chunk);
    fprintf(stderr, "[Storage PUT] key_hash=%016llx, size=%zu\n", key_hash, size);

    // Check if chunk already exists in L2
    auto l2_it = l2_cache_.find(key_hash);
    if (l2_it != l2_cache_.end()) {
        // Update existing L2 chunk
        thunder_kv_chunk & existing = l2_it->second;

        // Free old data
        if (existing.k_data) free(existing.k_data);
        if (existing.v_data) free(existing.v_data);

        // Allocate and copy new data
        existing.k_size = chunk.k_size;
        existing.v_size = chunk.v_size;
        existing.k_data = chunk.k_size > 0 ? malloc(chunk.k_size) : nullptr;
        existing.v_data = chunk.v_size > 0 ? malloc(chunk.v_size) : nullptr;
        if ((chunk.k_size > 0 && !existing.k_data) || (chunk.v_size > 0 && !existing.v_data)) {
            if (existing.k_data) free(existing.k_data);
            if (existing.v_data) free(existing.v_data);
            return false; // Allocation failed
        }
        if (chunk.k_size > 0) memcpy(existing.k_data, chunk.k_data, chunk.k_size);
        if (chunk.v_size > 0) memcpy(existing.v_data, chunk.v_data, chunk.v_size);

        // Update access time
        existing.last_access_ns = current_time_ns();
        existing.access_count++;

        // Update LRU (move to front)
        update_lru(chunk.key, true);

        return true;
    }

    // Check if chunk exists in L3
    auto l3_it = l3_offsets_.find(key_hash);
    if (l3_it != l3_offsets_.end()) {
        // Remove from L3 (we'll re-add to L2)
        l3_offsets_.erase(l3_it);

        // Remove from L3 LRU
        auto lru_it = l3_lru_index_.find(key_hash);
        if (lru_it != l3_lru_index_.end()) {
            l3_lru_.erase(lru_it->second);
            l3_lru_index_.erase(lru_it);
        }

        l3_usage_bytes_ -= size;
    }

    // Ensure L2 has space
    while (l2_usage_bytes_ + size > l2_limit_bytes_ && !l2_cache_.empty()) {
        evict_one_l2_to_l3();
    }

    // If still no space (chunk too large), fail
    if (l2_usage_bytes_ + size > l2_limit_bytes_) {
        return false;
    }

    // Allocate and copy data
    thunder_kv_chunk new_chunk = chunk;
    new_chunk.k_data = chunk.k_size > 0 ? malloc(chunk.k_size) : nullptr;
    new_chunk.v_data = chunk.v_size > 0 ? malloc(chunk.v_size) : nullptr;
    if ((chunk.k_size > 0 && !new_chunk.k_data) || (chunk.v_size > 0 && !new_chunk.v_data)) {
        if (new_chunk.k_data) free(new_chunk.k_data);
        if (new_chunk.v_data) free(new_chunk.v_data);
        return false;
    }
    if (chunk.k_size > 0) memcpy(new_chunk.k_data, chunk.k_data, chunk.k_size);
    if (chunk.v_size > 0) memcpy(new_chunk.v_data, chunk.v_data, chunk.v_size);

    // Update access time
    new_chunk.last_access_ns = current_time_ns();
    new_chunk.access_count++;

    // Insert into L2
    l2_cache_[key_hash] = new_chunk;
    l2_usage_bytes_ += size;

    // Update LRU
    l2_lru_.push_front(chunk.key);
    l2_lru_index_[key_hash] = l2_lru_.begin();

    fprintf(stderr, "[Storage PUT] SUCCESS: l2_cache size=%zu\n", l2_cache_.size());
    return true;
}

thunder_kv_chunk * ThunderChunkStorage::get(const thunder_kv_chunk_key & key) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t key_hash = hash_key(key);
    fprintf(stderr, "[Storage GET] key_hash=%016llx, l2_cache size=%zu\n", key_hash, l2_cache_.size());

    // Check L2
    auto l2_it = l2_cache_.find(key_hash);
    if (l2_it != l2_cache_.end()) {
        // L2 hit
        total_hits_++;

        thunder_kv_chunk & chunk = l2_it->second;
        chunk.last_access_ns = current_time_ns();
        chunk.access_count++;

        // Track access frequency for prefetch
        access_freq_[key_hash]++;
        total_accesses_++;

        // Update LRU
        update_lru(key, true);

        return &chunk;
    }

    // Check L3
    auto l3_it = l3_offsets_.find(key_hash);
    if (l3_it != l3_offsets_.end()) {
        // L3 hit
        total_hits_++;

        // Track access frequency for prefetch
        access_freq_[key_hash]++;
        total_accesses_++;

        // Read from disk
        thunder_kv_chunk chunk;
        if (!read_from_disk(l3_it->second, chunk)) {
            total_misses_++;
            total_hits_--; // Revert hit count
            return nullptr;
        }

        chunk.last_access_ns = current_time_ns();
        chunk.access_count++;

        // Update L3 LRU
        update_lru(key, false);

        // Promote to L2 if space available
        size_t size = chunk_size(chunk);
        if (l2_usage_bytes_ + size <= l2_limit_bytes_) {
            // Move from L3 to L2
            l3_offsets_.erase(l3_it);

            auto lru_it = l3_lru_index_.find(key_hash);
            if (lru_it != l3_lru_index_.end()) {
                l3_lru_.erase(lru_it->second);
                l3_lru_index_.erase(lru_it);
            }

            l3_usage_bytes_ -= size;

            // Insert into L2
            l2_cache_[key_hash] = chunk;
            l2_usage_bytes_ += size;

            l2_lru_.push_front(key);
            l2_lru_index_[key_hash] = l2_lru_.begin();

            return &l2_cache_[key_hash];
        } else {
            // Keep in L3, but cache the read chunk in L2 if possible
            // For now, we just return nullptr and let caller handle L3 reads
            // (In a real implementation, we'd cache the read chunk temporarily)

            // Note: We cannot return a pointer to local variable 'chunk'
            // For Phase 1, we don't support promoting L3 to L2 on read
            // Instead, we return nullptr to force caller to re-put if needed

            // Free allocated data
            if (chunk.k_data) free(chunk.k_data);
            if (chunk.v_data) free(chunk.v_data);

            // For now, treat L3 hits as misses in terms of API
            // (Proper implementation would maintain a read buffer)
            total_misses_++;
            total_hits_--; // Revert hit count
            return nullptr;
        }
    }

    // Miss
    total_misses_++;
    return nullptr;
}

void ThunderChunkStorage::evict_lru(size_t target_free_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t freed = 0;
    while (freed < target_free_bytes && !l2_cache_.empty()) {
        freed += evict_one_l2_to_l3();
    }
}

size_t ThunderChunkStorage::get_cpu_usage_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return l2_usage_bytes_;
}

size_t ThunderChunkStorage::get_disk_usage_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return l3_usage_bytes_;
}

double ThunderChunkStorage::get_hit_rate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = total_hits_ + total_misses_;
    return total > 0 ? static_cast<double>(total_hits_) / total : 0.0;
}

// ============================================================================
// Private Methods
// ============================================================================

uint64_t ThunderChunkStorage::hash_key(const thunder_kv_chunk_key & key) {
    // Simple hash combining content_hash, layer_idx, chunk_start
    uint64_t h = key.content_hash;
    h ^= static_cast<uint64_t>(key.layer_idx) << 32;
    h ^= static_cast<uint64_t>(key.chunk_start);
    return h;
}

size_t ThunderChunkStorage::chunk_size(const thunder_kv_chunk & chunk) {
    return chunk.k_size + chunk.v_size;
}

void ThunderChunkStorage::update_lru(const thunder_kv_chunk_key & key, bool is_l2) {
    uint64_t key_hash = hash_key(key);

    if (is_l2) {
        // Update L2 LRU
        auto it = l2_lru_index_.find(key_hash);
        if (it != l2_lru_index_.end()) {
            // Move to front
            l2_lru_.erase(it->second);
            l2_lru_.push_front(key);
            l2_lru_index_[key_hash] = l2_lru_.begin();
        }
    } else {
        // Update L3 LRU
        auto it = l3_lru_index_.find(key_hash);
        if (it != l3_lru_index_.end()) {
            // Move to front
            l3_lru_.erase(it->second);
            l3_lru_.push_front(key);
            l3_lru_index_[key_hash] = l3_lru_.begin();
        }
    }
}

size_t ThunderChunkStorage::evict_one_l2_to_l3() {
    if (l2_lru_.empty()) {
        return 0;
    }

    // Get LRU chunk (back of list)
    thunder_kv_chunk_key key = l2_lru_.back();
    uint64_t key_hash = hash_key(key);

    auto it = l2_cache_.find(key_hash);
    if (it == l2_cache_.end()) {
        // Inconsistent state, remove from LRU
        l2_lru_.pop_back();
        l2_lru_index_.erase(key_hash);
        return 0;
    }

    thunder_kv_chunk & chunk = it->second;
    size_t size = chunk_size(chunk);

    // Write to L3
    size_t offset = write_to_disk(chunk);
    if (offset != static_cast<size_t>(-1)) {
        // Successfully written to L3
        l3_offsets_[key_hash] = offset;
        l3_usage_bytes_ += size;

        // Add to L3 LRU
        l3_lru_.push_front(key);
        l3_lru_index_[key_hash] = l3_lru_.begin();
    }

    // Free L2 data
    if (chunk.k_data) free(chunk.k_data);
    if (chunk.v_data) free(chunk.v_data);

    // Remove from L2
    l2_cache_.erase(it);
    l2_usage_bytes_ -= size;

    // Remove from L2 LRU
    l2_lru_.pop_back();
    l2_lru_index_.erase(key_hash);

    return size;
}

size_t ThunderChunkStorage::evict_one_l3() {
    if (l3_lru_.empty()) {
        return 0;
    }

    // Get LRU chunk (back of list)
    thunder_kv_chunk_key key = l3_lru_.back();
    uint64_t key_hash = hash_key(key);

    auto it = l3_offsets_.find(key_hash);
    if (it == l3_offsets_.end()) {
        // Inconsistent state, remove from LRU
        l3_lru_.pop_back();
        l3_lru_index_.erase(key_hash);
        return 0;
    }

    // Read chunk to get size
    thunder_kv_chunk chunk;
    size_t size = 0;
    if (read_from_disk(it->second, chunk)) {
        size = chunk_size(chunk);

        // Free allocated data
        if (chunk.k_data) free(chunk.k_data);
        if (chunk.v_data) free(chunk.v_data);
    }

    // Remove from L3
    l3_offsets_.erase(it);
    l3_usage_bytes_ -= size;

    // Remove from L3 LRU
    l3_lru_.pop_back();
    l3_lru_index_.erase(key_hash);

    // Note: Disk space is not reclaimed (would require compaction)
    // For Phase 1, we just mark the space as unused

    return size;
}

size_t ThunderChunkStorage::write_to_disk(const thunder_kv_chunk & chunk) {
    // Check if L3 is enabled (may be disabled due to disk failure or unmount)
    if (!l3_enabled_) {
        return static_cast<size_t>(-1);
    }

    size_t size = chunk_size(chunk);

    // Ensure L3 has space
    while (l3_usage_bytes_ + size > l3_limit_bytes_ && !l3_offsets_.empty()) {
        evict_one_l3();
    }

    // If still no space, fail
    if (l3_usage_bytes_ + size > l3_limit_bytes_) {
        return static_cast<size_t>(-1);
    }

    // Auto-compact if disk usage > 90% of limit
    // (Reclaim space from evicted chunks to prevent disk file from growing indefinitely)
    if (disk_next_offset_ > l3_limit_bytes_ * 9 / 10 && !l3_offsets_.empty()) {
        fprintf(stderr, "[ThunderChunkStorage] Disk usage %.1f%%, triggering compaction...\n",
                100.0 * disk_next_offset_ / l3_limit_bytes_);
        if (!compact_disk()) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Compaction failed\n");
        }
    }

    // Chunk on-disk format (v3 with checksum + compression):
    // [8 bytes: content_hash] [4 bytes: layer_idx] [4 bytes: chunk_start]
    // [8 bytes: k_size (original)] [8 bytes: v_size (original)]
    // [8 bytes: last_access_ns] [4 bytes: access_count] [4 bytes: padding]
    // [8 bytes: data_checksum (XXH64 of original k_data + v_data)]
    // [8 bytes: compressed_k_size] [8 bytes: compressed_v_size]
    // [compressed_k_size bytes: compressed_k_data]
    // [compressed_v_size bytes: compressed_v_data]

    // Compress k_data
    uLongf compressed_k_size = compressBound(chunk.k_size);
    uint8_t * compressed_k_data = (uint8_t *)malloc(compressed_k_size);
    if (!compressed_k_data) {
        return static_cast<size_t>(-1);
    }
    int ret_k = compress(compressed_k_data, &compressed_k_size,
                         (const Bytef *)chunk.k_data, chunk.k_size);
    if (ret_k != Z_OK) {
        free(compressed_k_data);
        return static_cast<size_t>(-1);
    }

    // Compress v_data
    uLongf compressed_v_size = compressBound(chunk.v_size);
    uint8_t * compressed_v_data = (uint8_t *)malloc(compressed_v_size);
    if (!compressed_v_data) {
        free(compressed_k_data);
        return static_cast<size_t>(-1);
    }
    int ret_v = compress(compressed_v_data, &compressed_v_size,
                         (const Bytef *)chunk.v_data, chunk.v_size);
    if (ret_v != Z_OK) {
        free(compressed_k_data);
        free(compressed_v_data);
        return static_cast<size_t>(-1);
    }

    size_t header_size = 8 + 4 + 4 + 8 + 8 + 8 + 4 + 4 + 8 + 8 + 8; // 72 bytes
    size_t total_size = header_size + compressed_k_size + compressed_v_size;

    // Ensure disk space
    if (!ensure_disk_space(disk_next_offset_ + total_size)) {
        free(compressed_k_data);
        free(compressed_v_data);
        return static_cast<size_t>(-1);
    }

    // Calculate checksum of ORIGINAL data (before compression)
    uint64_t checksum_k = XXH64(chunk.k_data, chunk.k_size, 0);
    uint64_t checksum_v = XXH64(chunk.v_data, chunk.v_size, checksum_k);
    uint64_t data_checksum = checksum_v;

    // Write header
    uint8_t * ptr = static_cast<uint8_t *>(disk_mmap_) + disk_next_offset_;
    memcpy(ptr, &chunk.key.content_hash, 8); ptr += 8;
    memcpy(ptr, &chunk.key.layer_idx, 4); ptr += 4;
    memcpy(ptr, &chunk.key.chunk_start, 4); ptr += 4;
    memcpy(ptr, &chunk.k_size, 8); ptr += 8;  // Original size
    memcpy(ptr, &chunk.v_size, 8); ptr += 8;  // Original size
    memcpy(ptr, &chunk.last_access_ns, 8); ptr += 8;
    memcpy(ptr, &chunk.access_count, 4); ptr += 4;
    uint32_t padding = 0;
    memcpy(ptr, &padding, 4); ptr += 4;
    memcpy(ptr, &data_checksum, 8); ptr += 8; // Checksum of original data
    memcpy(ptr, &compressed_k_size, 8); ptr += 8; // NEW: compressed size
    memcpy(ptr, &compressed_v_size, 8); ptr += 8; // NEW: compressed size

    // Write compressed data
    memcpy(ptr, compressed_k_data, compressed_k_size); ptr += compressed_k_size;
    memcpy(ptr, compressed_v_data, compressed_v_size); ptr += compressed_v_size;

    // Free compressed buffers
    free(compressed_k_data);
    free(compressed_v_data);

    // No msync here for performance - rely on OS page cache and final msync in destructor
    // This avoids blocking on each chunk write (432 chunks = 432 msync calls)
    // Data safety: destructor calls msync(MS_SYNC) once to flush all pages
    // Trade-off: abnormal termination may lose cache, but cache is for optimization only

    size_t offset = disk_next_offset_;
    disk_next_offset_ += total_size;

    return offset;
}

bool ThunderChunkStorage::read_from_disk(size_t offset, thunder_kv_chunk & chunk) {
    if (offset >= disk_mmap_size_) {
        return false;
    }

    // Read header
    uint8_t * ptr = static_cast<uint8_t *>(disk_mmap_) + offset;
    memcpy(&chunk.key.content_hash, ptr, 8); ptr += 8;
    memcpy(&chunk.key.layer_idx, ptr, 4); ptr += 4;
    memcpy(&chunk.key.chunk_start, ptr, 4); ptr += 4;
    memcpy(&chunk.k_size, ptr, 8); ptr += 8;  // Original size
    memcpy(&chunk.v_size, ptr, 8); ptr += 8;  // Original size
    memcpy(&chunk.last_access_ns, ptr, 8); ptr += 8;
    memcpy(&chunk.access_count, ptr, 4); ptr += 4;
    ptr += 4; // skip padding

    // Read stored checksum
    uint64_t stored_checksum;
    memcpy(&stored_checksum, ptr, 8); ptr += 8;

    // Read compressed sizes
    size_t compressed_k_size, compressed_v_size;
    memcpy(&compressed_k_size, ptr, 8); ptr += 8;
    memcpy(&compressed_v_size, ptr, 8); ptr += 8;

    // Allocate buffers for decompression
    chunk.k_data = malloc(chunk.k_size);
    chunk.v_data = malloc(chunk.v_size);
    if (!chunk.k_data || !chunk.v_data) {
        if (chunk.k_data) free(chunk.k_data);
        if (chunk.v_data) free(chunk.v_data);
        return false;
    }

    // Decompress k_data
    uLongf decompressed_k_size = chunk.k_size;
    int ret_k = uncompress((Bytef *)chunk.k_data, &decompressed_k_size,
                           (const Bytef *)ptr, compressed_k_size);
    ptr += compressed_k_size;

    if (ret_k != Z_OK || decompressed_k_size != chunk.k_size) {
        fprintf(stderr, "[ThunderChunkStorage] ⚠️  Decompression FAILED for k_data at offset %zu\n", offset);
        free(chunk.k_data);
        free(chunk.v_data);
        chunk.k_data = nullptr;
        chunk.v_data = nullptr;
        return false;
    }

    // Decompress v_data
    uLongf decompressed_v_size = chunk.v_size;
    int ret_v = uncompress((Bytef *)chunk.v_data, &decompressed_v_size,
                           (const Bytef *)ptr, compressed_v_size);
    ptr += compressed_v_size;

    if (ret_v != Z_OK || decompressed_v_size != chunk.v_size) {
        fprintf(stderr, "[ThunderChunkStorage] ⚠️  Decompression FAILED for v_data at offset %zu\n", offset);
        free(chunk.k_data);
        free(chunk.v_data);
        chunk.k_data = nullptr;
        chunk.v_data = nullptr;
        return false;
    }

    // Verify checksum (on decompressed data)
    uint64_t checksum_k = XXH64(chunk.k_data, chunk.k_size, 0);
    uint64_t checksum_v = XXH64(chunk.v_data, chunk.v_size, checksum_k);
    uint64_t computed_checksum = checksum_v;

    if (computed_checksum != stored_checksum) {
        fprintf(stderr, "[ThunderChunkStorage] ⚠️  CHECKSUM MISMATCH at offset %zu: stored=0x%llx, computed=0x%llx\n",
                offset, (unsigned long long)stored_checksum, (unsigned long long)computed_checksum);
        fprintf(stderr, "[ThunderChunkStorage] ⚠️  Data corruption detected - chunk discarded\n");
        free(chunk.k_data);
        free(chunk.v_data);
        chunk.k_data = nullptr;
        chunk.v_data = nullptr;
        return false;
    }

    return true;
}

bool ThunderChunkStorage::ensure_disk_space(size_t required_size) {
    if (required_size <= disk_mmap_size_) {
        return true;
    }

    // Expand file to double size or required_size, whichever is larger
    size_t new_size = std::max(disk_mmap_size_ * 2, required_size);

    // Limit to L3 limit
    new_size = std::min(new_size, l3_limit_bytes_);

    if (new_size <= disk_mmap_size_) {
        return false; // Cannot expand further
    }

    // Unmap old mapping
    if (munmap(disk_mmap_, disk_mmap_size_) < 0) {
        return false;
    }

    // Resize file
    if (ftruncate(disk_fd_, new_size) < 0) {
        // Try to remap old size
        disk_mmap_ = mmap(nullptr, disk_mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, disk_fd_, 0);
        return false;
    }

    // Remap with new size
    disk_mmap_ = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, disk_fd_, 0);
    if (disk_mmap_ == MAP_FAILED) {
        // Critical error
        disk_mmap_ = nullptr;
        disk_mmap_size_ = 0;
        return false;
    }

    disk_mmap_size_ = new_size;
    return true;
}

uint64_t ThunderChunkStorage::current_time_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

std::string ThunderChunkStorage::expand_tilde(const std::string & path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char * home = getenv("HOME");
    if (!home) {
        struct passwd * pw = getpwuid(getuid());
        if (pw) {
            home = pw->pw_dir;
        }
    }

    if (!home) {
        return path; // Cannot expand, return as-is
    }

    return std::string(home) + path.substr(1);
}

void ThunderChunkStorage::load_cache_from_disk() {
    // Scan disk file and rebuild L3 index
    size_t offset = 0;
    size_t header_size = 72; // Fixed header size (v3 format with checksum + compression)
    int chunks_loaded = 0;

    fprintf(stderr, "[ThunderChunkStorage] Loading cache from disk...\n");

    while (offset + header_size <= disk_mmap_size_) {
        uint8_t * ptr = static_cast<uint8_t *>(disk_mmap_) + offset;

        // Read header to determine chunk size
        thunder_kv_chunk_key key;
        size_t k_size, v_size;
        size_t compressed_k_size, compressed_v_size;

        memcpy(&key.content_hash, ptr, 8); ptr += 8;
        memcpy(&key.layer_idx, ptr, 4); ptr += 4;
        memcpy(&key.chunk_start, ptr, 4); ptr += 4;
        memcpy(&k_size, ptr, 8); ptr += 8;
        memcpy(&v_size, ptr, 8); ptr += 8;
        // Skip last_access_ns, access_count, padding, checksum (28 bytes)
        ptr += 28;
        memcpy(&compressed_k_size, ptr, 8); ptr += 8;
        memcpy(&compressed_v_size, ptr, 8); ptr += 8;

        size_t chunk_size = k_size + v_size; // Original size for usage tracking
        size_t total_size = header_size + compressed_k_size + compressed_v_size; // Actual disk size

        // Check if this is a valid chunk (non-zero size)
        if (chunk_size == 0 || offset + total_size > disk_mmap_size_) {
            // Reached end of valid data
            break;
        }

        // Add to L3 index
        uint64_t key_hash = hash_key(key);
        l3_offsets_[key_hash] = offset;
        l3_usage_bytes_ += chunk_size;

        // Add to L3 LRU (at back, since we don't know access order)
        l3_lru_.push_back(key);
        l3_lru_index_[key_hash] = std::prev(l3_lru_.end());

        chunks_loaded++;
        offset += total_size;
    }

    // Update next available offset
    disk_next_offset_ = offset;

    fprintf(stderr, "[ThunderChunkStorage] Loaded %d chunks from disk (offset=%zu, usage=%zu bytes)\n",
            chunks_loaded, disk_next_offset_, l3_usage_bytes_);
}

bool ThunderChunkStorage::compact_disk() {
    // Compaction: Create new file with only valid chunks, reclaim space from evicted chunks

    auto start = std::chrono::steady_clock::now();

    std::string temp_path = disk_path_ + ".compact";

    // 1. Open temporary file
    int temp_fd = open(temp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (temp_fd < 0) {
        fprintf(stderr, "[ThunderChunkStorage] ERROR: Failed to create temp file: %s\n", temp_path.c_str());
        return false;
    }

    // 2. Collect all valid chunks with their offsets
    std::vector<std::pair<uint64_t, size_t>> valid_chunks; // (key_hash, old_offset)
    valid_chunks.reserve(l3_offsets_.size());
    for (const auto & [key_hash, offset] : l3_offsets_) {
        valid_chunks.push_back({key_hash, offset});
    }

    fprintf(stderr, "[ThunderChunkStorage] Compacting %zu chunks...\n", valid_chunks.size());

    // 3. Write all valid chunks to new file
    size_t new_offset = 0;
    std::unordered_map<uint64_t, size_t> new_offsets;

    for (const auto & [key_hash, old_offset] : valid_chunks) {
        // Read chunk from old file
        thunder_kv_chunk chunk;
        if (!read_from_disk(old_offset, chunk)) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Failed to read chunk at offset %zu during compaction\n", old_offset);
            continue;
        }

        // Calculate sizes
        size_t header_size = 56; // v2 format with checksum
        size_t total_size = header_size + chunk.k_size + chunk.v_size;

        // Calculate checksum
        uint64_t checksum_k = XXH64(chunk.k_data, chunk.k_size, 0);
        uint64_t checksum_v = XXH64(chunk.v_data, chunk.v_size, checksum_k);
        uint64_t data_checksum = checksum_v;

        // Ensure temp file has space
        if (ftruncate(temp_fd, new_offset + total_size) < 0) {
            fprintf(stderr, "[ThunderChunkStorage] ERROR: ftruncate failed during compaction\n");
            close(temp_fd);
            unlink(temp_path.c_str());
            if (chunk.k_data) free(chunk.k_data);
            if (chunk.v_data) free(chunk.v_data);
            return false;
        }

        // Map temp file region
        void * temp_map = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, temp_fd, new_offset);
        if (temp_map == MAP_FAILED) {
            fprintf(stderr, "[ThunderChunkStorage] ERROR: mmap failed during compaction\n");
            close(temp_fd);
            unlink(temp_path.c_str());
            if (chunk.k_data) free(chunk.k_data);
            if (chunk.v_data) free(chunk.v_data);
            return false;
        }

        // Write header
        uint8_t * ptr = static_cast<uint8_t *>(temp_map);
        memcpy(ptr, &chunk.key.content_hash, 8); ptr += 8;
        memcpy(ptr, &chunk.key.layer_idx, 4); ptr += 4;
        memcpy(ptr, &chunk.key.chunk_start, 4); ptr += 4;
        memcpy(ptr, &chunk.k_size, 8); ptr += 8;
        memcpy(ptr, &chunk.v_size, 8); ptr += 8;
        memcpy(ptr, &chunk.last_access_ns, 8); ptr += 8;
        memcpy(ptr, &chunk.access_count, 4); ptr += 4;
        uint32_t padding = 0;
        memcpy(ptr, &padding, 4); ptr += 4;
        memcpy(ptr, &data_checksum, 8); ptr += 8; // NEW: checksum

        // Write data
        memcpy(ptr, chunk.k_data, chunk.k_size); ptr += chunk.k_size;
        memcpy(ptr, chunk.v_data, chunk.v_size);

        // Sync this chunk
        msync(temp_map, total_size, MS_SYNC);
        munmap(temp_map, total_size);

        // Record new offset
        new_offsets[key_hash] = new_offset;
        new_offset += total_size;

        // Free chunk data
        if (chunk.k_data) free(chunk.k_data);
        if (chunk.v_data) free(chunk.v_data);
    }

    close(temp_fd);

    // 4. Unmap old file
    if (disk_mmap_ != nullptr && disk_mmap_ != MAP_FAILED) {
        munmap(disk_mmap_, disk_mmap_size_);
        disk_mmap_ = nullptr;
        disk_mmap_size_ = 0;
    }

    if (disk_fd_ >= 0) {
        close(disk_fd_);
        disk_fd_ = -1;
    }

    // 5. Replace old file with new file
    if (rename(temp_path.c_str(), disk_path_.c_str()) < 0) {
        fprintf(stderr, "[ThunderChunkStorage] ERROR: Failed to rename temp file\n");
        return false;
    }

    // 6. Reopen new file
    disk_fd_ = open(disk_path_.c_str(), O_RDWR, 0644);
    if (disk_fd_ < 0) {
        fprintf(stderr, "[ThunderChunkStorage] ERROR: Failed to reopen compacted file\n");
        return false;
    }

    // 7. Remap file
    struct stat st;
    if (fstat(disk_fd_, &st) < 0) {
        close(disk_fd_);
        disk_fd_ = -1;
        return false;
    }

    if (st.st_size > 0) {
        disk_mmap_size_ = st.st_size;
        disk_mmap_ = mmap(nullptr, disk_mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, disk_fd_, 0);
        if (disk_mmap_ == MAP_FAILED) {
            close(disk_fd_);
            disk_fd_ = -1;
            return false;
        }
    }

    // 8. Update offsets index
    l3_offsets_ = std::move(new_offsets);
    disk_next_offset_ = new_offset;

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double reclaim_pct = st.st_size > 0
        ? 100.0 * (st.st_size - static_cast<off_t>(disk_next_offset_)) / st.st_size
        : 0.0;

    fprintf(stderr, "[ThunderChunkStorage] Compaction complete: %zu chunks, %lld -> %zu bytes (%.1f%% reclaimed) in %ld ms\n",
            l3_offsets_.size(),
            static_cast<long long>(st.st_size), disk_next_offset_,
            reclaim_pct,
            duration);

    return true;
}

bool ThunderChunkStorage::check_disk_health() {
    // Assumes mutex_ is held

    if (disk_fd_ < 0 || disk_mmap_ == nullptr) {
        return false;
    }

    // Test 1: Check if disk is still mounted (stat the file)
    struct stat st;
    if (fstat(disk_fd_, &st) < 0) {
        fprintf(stderr, "[ThunderChunkStorage] Disk health check FAILED: fstat error (errno=%d)\n", errno);
        return false;
    }

    // Test 2: Try a small test write to verify disk is writable
    // Write a magic number at offset 0 (header area, won't corrupt data)
    if (disk_mmap_size_ >= 8) {
        // Save original value
        uint64_t original_value;
        memcpy(&original_value, disk_mmap_, 8);

        // Try to write test pattern
        uint64_t test_value = 0xDEADBEEFCAFEBABE;
        memcpy(disk_mmap_, &test_value, 8);

        // Verify write
        uint64_t verify_value;
        memcpy(&verify_value, disk_mmap_, 8);

        // Restore original value
        memcpy(disk_mmap_, &original_value, 8);

        if (verify_value != test_value) {
            fprintf(stderr, "[ThunderChunkStorage] Disk health check FAILED: test write verification failed\n");
            return false;
        }
    }

    return true;
}

void ThunderChunkStorage::disable_l3() {
    // Assumes mutex_ is held

    fprintf(stderr, "[ThunderChunkStorage] Disabling L3 disk cache...\n");

    l3_enabled_ = false;

    // Clear L3 index (data will be lost, but L2 continues working)
    l3_offsets_.clear();
    l3_lru_.clear();
    l3_lru_index_.clear();
    l3_usage_bytes_ = 0;

    // Unmap and close disk file
    if (disk_mmap_ != nullptr && disk_mmap_ != MAP_FAILED) {
        munmap(disk_mmap_, disk_mmap_size_);
        disk_mmap_ = nullptr;
        disk_mmap_size_ = 0;
    }

    if (disk_fd_ >= 0) {
        close(disk_fd_);
        disk_fd_ = -1;
    }

    fprintf(stderr, "[ThunderChunkStorage] L3 disabled, running in memory-only mode (L2 only)\n");
}

void ThunderChunkStorage::safe_unmount() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!l3_enabled_) {
        fprintf(stderr, "[ThunderChunkStorage] L3 already disabled, nothing to unmount\n");
        return;
    }

    fprintf(stderr, "[ThunderChunkStorage] ⚠️  Received unmount request - preparing for external storage removal...\n");

    // Step 1: Stop accepting new writes
    l3_enabled_ = false;
    fprintf(stderr, "[ThunderChunkStorage] ✓ Stopped accepting new writes to L3\n");

    // Step 2: Sync all pending writes to disk
    if (disk_mmap_ != nullptr && disk_mmap_ != MAP_FAILED) {
        fprintf(stderr, "[ThunderChunkStorage] Syncing all pending data to disk...\n");
        if (msync(disk_mmap_, disk_mmap_size_, MS_SYNC) < 0) {
            fprintf(stderr, "[ThunderChunkStorage] WARNING: Final sync failed (errno=%d)\n", errno);
        } else {
            fprintf(stderr, "[ThunderChunkStorage] ✓ All data synced to disk\n");
        }

        // Step 3: Unmap the file
        munmap(disk_mmap_, disk_mmap_size_);
        disk_mmap_ = nullptr;
        disk_mmap_size_ = 0;
        fprintf(stderr, "[ThunderChunkStorage] ✓ Disk file unmapped\n");
    }

    // Step 4: Close the file
    if (disk_fd_ >= 0) {
        close(disk_fd_);
        disk_fd_ = -1;
        fprintf(stderr, "[ThunderChunkStorage] ✓ Disk file closed\n");
    }

    fprintf(stderr, "[ThunderChunkStorage] ✅ Safe unmount complete - external storage can be safely removed\n");
    fprintf(stderr, "[ThunderChunkStorage] ℹ️  L2 cache (8GB memory) will continue to work\n");
}

bool ThunderChunkStorage::is_l3_enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return l3_enabled_;
}

void ThunderChunkStorage::prefetch_hot_chunks(size_t top_n) {
    if (!l3_enabled_ || l3_offsets_.empty()) {
        return; // Nothing to prefetch
    }

    fprintf(stderr, "[ThunderChunkStorage] 🔥 Prefetching top-%zu hot chunks from L3 to L2 (parallel I/O)...\n", top_n);

    // Build list of (key_hash, access_count, offset) tuples
    std::vector<std::tuple<uint64_t, uint32_t, size_t>> prefetch_list;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto & [key_hash, count] : access_freq_) {
            // Only prefetch chunks that are in L3 (not already in L2)
            auto l3_it = l3_offsets_.find(key_hash);
            if (l3_it != l3_offsets_.end() && l2_cache_.find(key_hash) == l2_cache_.end()) {
                prefetch_list.push_back({key_hash, count, l3_it->second});
            }
        }
    }

    // Sort by access count (descending)
    std::sort(prefetch_list.begin(), prefetch_list.end(),
              [](const auto & a, const auto & b) { return std::get<1>(a) > std::get<1>(b); });

    // Limit to top-N
    if (prefetch_list.size() > top_n) {
        prefetch_list.resize(top_n);
    }

    if (prefetch_list.empty()) {
        fprintf(stderr, "[ThunderChunkStorage] No chunks to prefetch\n");
        return;
    }

    // Parallel read phase: read chunks from disk in parallel
    const size_t num_threads = std::min(size_t(4), prefetch_list.size()); // Use 4 threads max
    std::vector<std::future<std::pair<uint64_t, thunder_kv_chunk>>> futures;

    auto read_chunk_async = [this](uint64_t key_hash, size_t offset) -> std::pair<uint64_t, thunder_kv_chunk> {
        thunder_kv_chunk chunk;
        std::lock_guard<std::mutex> lock(mutex_); // Protect mmap access
        if (read_from_disk(offset, chunk)) {
            return {key_hash, chunk};
        } else {
            // Return invalid chunk on failure
            chunk.k_data = nullptr;
            chunk.v_data = nullptr;
            return {key_hash, chunk};
        }
    };

    // Launch parallel reads
    for (const auto & [key_hash, count, offset] : prefetch_list) {
        futures.push_back(std::async(std::launch::async, read_chunk_async, key_hash, offset));
    }

    // Collect results and insert into L2
    size_t prefetched = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto & future : futures) {
            auto [key_hash, chunk] = future.get();

            if (chunk.k_data == nullptr || chunk.v_data == nullptr) {
                continue; // Failed to read
            }

            size_t size = chunk_size(chunk);

            // Check if L2 has space
            if (l2_usage_bytes_ + size > l2_limit_bytes_) {
                // No more space in L2
                free(chunk.k_data);
                free(chunk.v_data);
                break;
            }

            // Move from L3 to L2
            auto l3_it = l3_offsets_.find(key_hash);
            if (l3_it != l3_offsets_.end()) {
                l3_offsets_.erase(l3_it);
                l3_usage_bytes_ -= size;

                // Remove from L3 LRU
                auto lru_it = l3_lru_index_.find(key_hash);
                if (lru_it != l3_lru_index_.end()) {
                    l3_lru_.erase(lru_it->second);
                    l3_lru_index_.erase(lru_it);
                }
            }

            // Insert into L2
            l2_cache_[key_hash] = chunk;
            l2_usage_bytes_ += size;

            l2_lru_.push_front(chunk.key);
            l2_lru_index_[key_hash] = l2_lru_.begin();

            prefetched++;
        }
    }

    fprintf(stderr, "[ThunderChunkStorage] ✓ Prefetched %zu chunks using %zu threads (L2 usage: %.1f MB)\n",
            prefetched, num_threads, l2_usage_bytes_ / (1024.0 * 1024.0));
}
