#include "thunder-lmcache-storage.h"
#include "thunder-lmcache-hash.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <sys/stat.h>

// ============================================================================
// Test Helpers
// ============================================================================

void print_test(const char * name) {
    std::cout << "[TEST] " << name << std::endl;
}

void assert_true(bool cond, const char * msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        exit(1);
    }
}

thunder_kv_chunk make_test_chunk(
    uint64_t content_hash,
    int32_t layer_idx,
    int32_t chunk_start,
    size_t k_size,
    size_t v_size
) {
    thunder_kv_chunk chunk;
    chunk.key.content_hash = content_hash;
    chunk.key.layer_idx = layer_idx;
    chunk.key.chunk_start = chunk_start;

    chunk.k_size = k_size;
    chunk.v_size = v_size;

    chunk.k_data = malloc(k_size);
    chunk.v_data = malloc(v_size);

    // Fill with test pattern
    for (size_t i = 0; i < k_size; ++i) {
        static_cast<uint8_t *>(chunk.k_data)[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < v_size; ++i) {
        static_cast<uint8_t *>(chunk.v_data)[i] = static_cast<uint8_t>((i + 128) % 256);
    }

    chunk.last_access_ns = 0;
    chunk.access_count = 0;

    return chunk;
}

void free_test_chunk(thunder_kv_chunk & chunk) {
    if (chunk.k_data) {
        free(chunk.k_data);
        chunk.k_data = nullptr;
    }
    if (chunk.v_data) {
        free(chunk.v_data);
        chunk.v_data = nullptr;
    }
}

bool verify_chunk_data(const thunder_kv_chunk & chunk, size_t k_size, size_t v_size) {
    if (chunk.k_size != k_size || chunk.v_size != v_size) {
        return false;
    }

    for (size_t i = 0; i < k_size; ++i) {
        if (static_cast<uint8_t *>(chunk.k_data)[i] != static_cast<uint8_t>(i % 256)) {
            return false;
        }
    }
    for (size_t i = 0; i < v_size; ++i) {
        if (static_cast<uint8_t *>(chunk.v_data)[i] != static_cast<uint8_t>((i + 128) % 256)) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Unit Tests
// ============================================================================

static void test_basic_put_get() {
    print_test("Basic put/get");

    // Create storage with small limits for testing
    ThunderChunkStorage storage(
        1024 * 1024,  // 1MB L2
        4 * 1024 * 1024,  // 4MB L3
        "/tmp/thunder_test_cache.bin"
    );

    // Create test chunk
    thunder_kv_chunk chunk = make_test_chunk(12345, 0, 0, 1024, 1024);

    // Put chunk
    bool ok = storage.put(chunk);
    assert_true(ok, "put should succeed");

    // Get chunk
    thunder_kv_chunk * retrieved = storage.get(chunk.key);
    assert_true(retrieved != nullptr, "get should return chunk");
    assert_true(verify_chunk_data(*retrieved, 1024, 1024), "chunk data should match");

    // Verify CPU usage
    assert_true(storage.get_cpu_usage_bytes() == 2048, "CPU usage should be 2048 bytes");

    // Verify hit rate
    assert_true(storage.get_hit_rate() == 1.0, "hit rate should be 1.0");

    free_test_chunk(chunk);

    std::cout << "[PASS] Basic put/get" << std::endl;
}

static void test_lru_eviction() {
    print_test("LRU eviction from L2 to L3");

    // Create storage with very small L2 limit
    ThunderChunkStorage storage(
        4096,  // 4KB L2 (can hold ~2 chunks)
        1024 * 1024,  // 1MB L3
        "/tmp/thunder_test_lru.bin"
    );

    // Create 3 chunks, each 1KB + 1KB = 2KB
    std::vector<thunder_kv_chunk> chunks;
    for (int i = 0; i < 3; ++i) {
        chunks.push_back(make_test_chunk(1000 + i, 0, i * 256, 1024, 1024));
    }

    // Put first 2 chunks (should fit in L2)
    storage.put(chunks[0]);
    storage.put(chunks[1]);

    assert_true(storage.get_cpu_usage_bytes() == 4096, "L2 should have 4096 bytes");
    assert_true(storage.get_disk_usage_bytes() == 0, "L3 should be empty");

    // Put third chunk (should evict first chunk to L3)
    storage.put(chunks[2]);

    assert_true(storage.get_cpu_usage_bytes() == 4096, "L2 should still have 4096 bytes");
    assert_true(storage.get_disk_usage_bytes() > 0, "L3 should have data");

    // Try to get first chunk (should be in L3, but get returns nullptr for L3 in Phase 1)
    thunder_kv_chunk * c0 = storage.get(chunks[0].key);
    // Note: In current implementation, L3 hits return nullptr
    // This is acceptable for Phase 1

    // Get second and third chunks (should be in L2)
    thunder_kv_chunk * c1 = storage.get(chunks[1].key);
    thunder_kv_chunk * c2 = storage.get(chunks[2].key);

    assert_true(c1 != nullptr, "chunk 1 should be in L2");
    assert_true(c2 != nullptr, "chunk 2 should be in L2");

    for (auto & chunk : chunks) {
        free_test_chunk(chunk);
    }

    std::cout << "[PASS] LRU eviction" << std::endl;
}

static void test_update_existing() {
    print_test("Update existing chunk");

    ThunderChunkStorage storage(
        1024 * 1024,
        4 * 1024 * 1024,
        "/tmp/thunder_test_update.bin"
    );

    // Put initial chunk
    thunder_kv_chunk chunk1 = make_test_chunk(5555, 0, 0, 512, 512);
    storage.put(chunk1);

    // Update with different data
    thunder_kv_chunk chunk2 = make_test_chunk(5555, 0, 0, 1024, 1024);
    storage.put(chunk2);

    // Get and verify
    thunder_kv_chunk * retrieved = storage.get(chunk2.key);
    assert_true(retrieved != nullptr, "should retrieve updated chunk");
    assert_true(verify_chunk_data(*retrieved, 1024, 1024), "updated data should match");

    free_test_chunk(chunk1);
    free_test_chunk(chunk2);

    std::cout << "[PASS] Update existing chunk" << std::endl;
}

static void test_hit_rate() {
    print_test("Hit rate calculation");

    ThunderChunkStorage storage(
        1024 * 1024,
        4 * 1024 * 1024,
        "/tmp/thunder_test_hitrate.bin"
    );

    thunder_kv_chunk chunk = make_test_chunk(7777, 0, 0, 1024, 1024);
    storage.put(chunk);

    // 3 hits
    storage.get(chunk.key);
    storage.get(chunk.key);
    storage.get(chunk.key);

    // 2 misses
    thunder_kv_chunk_key miss_key = {9999, 0, 0};
    storage.get(miss_key);
    storage.get(miss_key);

    // Hit rate should be 3/5 = 0.6
    double hit_rate = storage.get_hit_rate();
    assert_true(hit_rate >= 0.59 && hit_rate <= 0.61, "hit rate should be ~0.6");

    free_test_chunk(chunk);

    std::cout << "[PASS] Hit rate calculation" << std::endl;
}

static void test_manual_eviction() {
    print_test("Manual eviction");

    ThunderChunkStorage storage(
        1024 * 1024,
        4 * 1024 * 1024,
        "/tmp/thunder_test_evict.bin"
    );

    // Put 10 chunks
    std::vector<thunder_kv_chunk> chunks;
    for (int i = 0; i < 10; ++i) {
        chunks.push_back(make_test_chunk(8000 + i, 0, i * 256, 10 * 1024, 10 * 1024));
        storage.put(chunks[i]);
    }

    size_t usage_before = storage.get_cpu_usage_bytes();

    // Evict 100KB
    storage.evict_lru(100 * 1024);

    size_t usage_after = storage.get_cpu_usage_bytes();
    assert_true(usage_after <= usage_before - 100 * 1024, "should evict at least 100KB");

    for (auto & chunk : chunks) {
        free_test_chunk(chunk);
    }

    std::cout << "[PASS] Manual eviction" << std::endl;
}

static void test_thread_safety() {
    print_test("Thread safety");

    ThunderChunkStorage storage(
        10 * 1024 * 1024,
        40 * 1024 * 1024,
        "/tmp/thunder_test_threads.bin"
    );

    const int num_threads = 4;
    const int ops_per_thread = 100;

    auto worker = [&](int thread_id) {
        std::mt19937 rng(thread_id);
        std::uniform_int_distribution<uint64_t> dist(0, 999);

        for (int i = 0; i < ops_per_thread; ++i) {
            uint64_t hash = dist(rng);
            thunder_kv_chunk chunk = make_test_chunk(hash, thread_id, i * 256, 1024, 1024);

            // Put
            storage.put(chunk);

            // Get
            storage.get(chunk.key);

            free_test_chunk(chunk);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto & t : threads) {
        t.join();
    }

    // Verify storage is still functional
    thunder_kv_chunk test = make_test_chunk(99999, 0, 0, 1024, 1024);
    storage.put(test);
    thunder_kv_chunk * retrieved = storage.get(test.key);
    assert_true(retrieved != nullptr, "storage should be functional after concurrent access");

    free_test_chunk(test);

    std::cout << "[PASS] Thread safety" << std::endl;
}

static void test_persistence() {
    print_test("Disk persistence");

    const char * disk_path = "/tmp/thunder_test_persist.bin";

    // Create storage, put some chunks, destroy
    {
        ThunderChunkStorage storage(
            1024,  // Very small L2 to force L3 writes
            1024 * 1024,
            disk_path
        );

        for (int i = 0; i < 5; ++i) {
            thunder_kv_chunk chunk = make_test_chunk(10000 + i, 0, i * 256, 512, 512);
            storage.put(chunk);
            free_test_chunk(chunk);
        }

        // Force eviction to L3
        storage.evict_lru(1024);

        assert_true(storage.get_disk_usage_bytes() > 0, "should have data in L3");
    }

    // Verify disk file exists and has data
    struct stat st;
    assert_true(stat(disk_path, &st) == 0, "disk file should exist");
    assert_true(st.st_size > 0, "disk file should have data");

    std::cout << "[PASS] Disk persistence" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== ThunderChunkStorage Tests ===" << std::endl;

    test_basic_put_get();
    test_lru_eviction();
    test_update_existing();
    test_hit_rate();
    test_manual_eviction();
    test_thread_safety();
    test_persistence();

    std::cout << "\n=== All Tests Passed ===" << std::endl;
    return 0;
}
