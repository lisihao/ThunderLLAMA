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
    // Create parent directory if not exists
    size_t last_slash = disk_path_.rfind('/');
    if (last_slash != std::string::npos) {
        std::string dir = disk_path_.substr(0, last_slash);
        // Use mkdir -p to create parent directories
        std::string mkdir_cmd = "mkdir -p \"" + dir + "\"";
        if (system(mkdir_cmd.c_str()) != 0) {
            throw std::runtime_error("Failed to create cache directory: " + dir);
        }
    }

    // Open or create disk file
    disk_fd_ = open(disk_path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (disk_fd_ < 0) {
        throw std::runtime_error("Failed to open disk cache file: " + disk_path_);
    }

    // Get current file size
    struct stat st;
    if (fstat(disk_fd_, &st) < 0) {
        close(disk_fd_);
        throw std::runtime_error("Failed to stat disk cache file: " + disk_path_);
    }

    // If file is new or small, expand to initial size (1MB)
    size_t initial_size = 1024 * 1024; // 1MB
    if (static_cast<size_t>(st.st_size) < initial_size) {
        if (ftruncate(disk_fd_, initial_size) < 0) {
            close(disk_fd_);
            throw std::runtime_error("Failed to resize disk cache file: " + disk_path_);
        }
        disk_mmap_size_ = initial_size;
    } else {
        disk_mmap_size_ = st.st_size;
    }

    // mmap the file
    disk_mmap_ = mmap(nullptr, disk_mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, disk_fd_, 0);
    if (disk_mmap_ == MAP_FAILED) {
        close(disk_fd_);
        throw std::runtime_error("Failed to mmap disk cache file: " + disk_path_);
    }

    // Note: In a production implementation, we would load existing cache metadata
    // from disk header here. For Phase 1, we start with empty cache.
}

ThunderChunkStorage::~ThunderChunkStorage() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free all L2 chunks' allocated data
    for (auto & [hash, chunk] : l2_cache_) {
        if (chunk.k_data) {
            free(chunk.k_data);
            chunk.k_data = nullptr;
        }
        if (chunk.v_data) {
            free(chunk.v_data);
            chunk.v_data = nullptr;
        }
    }

    // Unmap disk file
    if (disk_mmap_ != nullptr && disk_mmap_ != MAP_FAILED) {
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
        existing.k_data = malloc(chunk.k_size);
        existing.v_data = malloc(chunk.v_size);
        if (!existing.k_data || !existing.v_data) {
            return false; // Allocation failed
        }
        memcpy(existing.k_data, chunk.k_data, chunk.k_size);
        memcpy(existing.v_data, chunk.v_data, chunk.v_size);

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
    new_chunk.k_data = malloc(chunk.k_size);
    new_chunk.v_data = malloc(chunk.v_size);
    if (!new_chunk.k_data || !new_chunk.v_data) {
        if (new_chunk.k_data) free(new_chunk.k_data);
        if (new_chunk.v_data) free(new_chunk.v_data);
        return false;
    }
    memcpy(new_chunk.k_data, chunk.k_data, chunk.k_size);
    memcpy(new_chunk.v_data, chunk.v_data, chunk.v_size);

    // Update access time
    new_chunk.last_access_ns = current_time_ns();
    new_chunk.access_count++;

    // Insert into L2
    l2_cache_[key_hash] = new_chunk;
    l2_usage_bytes_ += size;

    // Update LRU
    l2_lru_.push_front(chunk.key);
    l2_lru_index_[key_hash] = l2_lru_.begin();

    return true;
}

thunder_kv_chunk * ThunderChunkStorage::get(const thunder_kv_chunk_key & key) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t key_hash = hash_key(key);

    // Check L2
    auto l2_it = l2_cache_.find(key_hash);
    if (l2_it != l2_cache_.end()) {
        // L2 hit
        total_hits_++;

        thunder_kv_chunk & chunk = l2_it->second;
        chunk.last_access_ns = current_time_ns();
        chunk.access_count++;

        // Update LRU
        update_lru(key, true);

        return &chunk;
    }

    // Check L3
    auto l3_it = l3_offsets_.find(key_hash);
    if (l3_it != l3_offsets_.end()) {
        // L3 hit
        total_hits_++;

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
    size_t size = chunk_size(chunk);

    // Ensure L3 has space
    while (l3_usage_bytes_ + size > l3_limit_bytes_ && !l3_offsets_.empty()) {
        evict_one_l3();
    }

    // If still no space, fail
    if (l3_usage_bytes_ + size > l3_limit_bytes_) {
        return static_cast<size_t>(-1);
    }

    // Chunk on-disk format:
    // [8 bytes: content_hash] [4 bytes: layer_idx] [4 bytes: chunk_start]
    // [8 bytes: k_size] [8 bytes: v_size]
    // [8 bytes: last_access_ns] [4 bytes: access_count] [4 bytes: padding]
    // [k_size bytes: k_data]
    // [v_size bytes: v_data]

    size_t header_size = 8 + 4 + 4 + 8 + 8 + 8 + 4 + 4; // 48 bytes
    size_t total_size = header_size + chunk.k_size + chunk.v_size;

    // Ensure disk space
    if (!ensure_disk_space(disk_next_offset_ + total_size)) {
        return static_cast<size_t>(-1);
    }

    // Write header
    uint8_t * ptr = static_cast<uint8_t *>(disk_mmap_) + disk_next_offset_;
    memcpy(ptr, &chunk.key.content_hash, 8); ptr += 8;
    memcpy(ptr, &chunk.key.layer_idx, 4); ptr += 4;
    memcpy(ptr, &chunk.key.chunk_start, 4); ptr += 4;
    memcpy(ptr, &chunk.k_size, 8); ptr += 8;
    memcpy(ptr, &chunk.v_size, 8); ptr += 8;
    memcpy(ptr, &chunk.last_access_ns, 8); ptr += 8;
    memcpy(ptr, &chunk.access_count, 4); ptr += 4;
    uint32_t padding = 0;
    memcpy(ptr, &padding, 4); ptr += 4;

    // Write data
    memcpy(ptr, chunk.k_data, chunk.k_size); ptr += chunk.k_size;
    memcpy(ptr, chunk.v_data, chunk.v_size); ptr += chunk.v_size;

    // Sync to disk (optional, for durability)
    msync(static_cast<uint8_t *>(disk_mmap_) + disk_next_offset_, total_size, MS_ASYNC);

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
    memcpy(&chunk.k_size, ptr, 8); ptr += 8;
    memcpy(&chunk.v_size, ptr, 8); ptr += 8;
    memcpy(&chunk.last_access_ns, ptr, 8); ptr += 8;
    memcpy(&chunk.access_count, ptr, 4); ptr += 4;
    ptr += 4; // skip padding

    // Allocate and read data
    chunk.k_data = malloc(chunk.k_size);
    chunk.v_data = malloc(chunk.v_size);
    if (!chunk.k_data || !chunk.v_data) {
        if (chunk.k_data) free(chunk.k_data);
        if (chunk.v_data) free(chunk.v_data);
        return false;
    }

    memcpy(chunk.k_data, ptr, chunk.k_size); ptr += chunk.k_size;
    memcpy(chunk.v_data, ptr, chunk.v_size); ptr += chunk.v_size;

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
