/**
 * @file test_disk_io_async.cpp
 * @brief Performance test for Task #24: Disk I/O Asynchronization
 *
 * Tests the performance improvement of batch_get() vs sequential get() calls
 * for L3 cache hits.
 *
 * Expected improvement: 89ms → 40-50ms (45%+ improvement)
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "thunder-lmcache-storage.h"

using namespace std::chrono;

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Disk I/O Asynchronization - Performance Test                ║\n");
    printf("║  Task #24: madvise + batch_get() optimization                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Create storage with sufficient L2 to hold all test chunks
    size_t cpu_limit = 8ULL * 1024 * 1024;  // 8MB L2 (enough for 8 chunks)
    size_t disk_limit = 256ULL * 1024 * 1024;  // 256MB L3
    std::string disk_path = "/tmp/test_disk_io_async.bin";

    printf("Creating ThunderChunkStorage...\n");
    printf("  L2 (CPU): 8 MB\n");
    printf("  L3 (Disk): 256 MB\n");
    printf("  Disk path: %s\n\n", disk_path.c_str());

    ThunderChunkStorage storage(cpu_limit, disk_limit, disk_path);

    // Create test chunks
    const size_t num_chunks = 8;  // Test with 8 chunks
    const size_t chunk_size_k = 128 * 1024;  // 128KB per K
    const size_t chunk_size_v = 128 * 1024;  // 128KB per V
    const size_t total_chunk_size = chunk_size_k + chunk_size_v;  // 256KB per chunk

    printf("Preparing %zu test chunks (%zu KB each)...\n", num_chunks, total_chunk_size / 1024);

    std::vector<thunder_kv_chunk> chunks;
    std::vector<thunder_kv_chunk_key> keys;

    for (size_t i = 0; i < num_chunks; i++) {
        thunder_kv_chunk chunk;
        chunk.key.content_hash = 0x1000 + i;
        chunk.key.layer_idx = 0;
        chunk.key.chunk_start = i * 256;  // THUNDER_CHUNK_SIZE = 256
        chunk.k_size = chunk_size_k;
        chunk.v_size = chunk_size_v;
        chunk.last_access_ns = 0;
        chunk.access_count = 0;

        // Allocate and fill with random data (compressible)
        chunk.k_data = malloc(chunk_size_k);
        chunk.v_data = malloc(chunk_size_v);

        // Fill with semi-random data (50% compressibility)
        std::mt19937 rng(i);
        for (size_t j = 0; j < chunk_size_k / 8; j++) {
            uint64_t val = rng() % 256;  // Low entropy for compression
            reinterpret_cast<uint64_t*>(chunk.k_data)[j] = val;
        }
        for (size_t j = 0; j < chunk_size_v / 8; j++) {
            uint64_t val = rng() % 256;
            reinterpret_cast<uint64_t*>(chunk.v_data)[j] = val;
        }

        chunks.push_back(chunk);
        keys.push_back(chunk.key);
    }

    // Step 1: Put all chunks into storage (will go to L3 because L2 is small)
    printf("Storing %zu chunks to L3...\n", num_chunks);
    for (auto & chunk : chunks) {
        if (!storage.put(chunk)) {
            fprintf(stderr, "⚠️  Failed to store chunk\n");
            return 1;
        }
    }

    printf("  L2 usage: %.1f MB\n", storage.get_cpu_usage_bytes() / (1024.0 * 1024.0));
    printf("  L3 usage: %.1f MB\n", storage.get_disk_usage_bytes() / (1024.0 * 1024.0));
    printf("  Total chunks: %zu\n\n", storage.get_total_chunks());

    // Step 2: Clear L2 to force L3 hits
    printf("Clearing L2 cache to force L3 hits...\n");
    storage.evict_lru(cpu_limit);  // Evict everything from L2
    printf("  L2 usage after clear: %.1f MB\n\n", storage.get_cpu_usage_bytes() / (1024.0 * 1024.0));

    // ========================================================================
    // Test 1: Sequential get() (OLD approach)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Test 1: Sequential get() (OLD approach)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Clear L2 again
    storage.evict_lru(cpu_limit);

    auto start_sequential = high_resolution_clock::now();

    size_t sequential_hits = 0;
    for (const auto & key : keys) {
        thunder_kv_chunk * result = storage.get(key);
        if (result != nullptr) {
            sequential_hits++;
        }

        // Clear L2 after each get to simulate pure L3 reads
        storage.evict_lru(cpu_limit);
    }

    auto end_sequential = high_resolution_clock::now();
    auto duration_sequential_us = duration_cast<microseconds>(end_sequential - start_sequential);

    printf("Results:\n");
    printf("  Chunks retrieved: %zu / %zu\n", sequential_hits, num_chunks);
    printf("  Total time: %lld us (%.1f ms)\n", duration_sequential_us.count(), duration_sequential_us.count() / 1000.0);
    printf("  Avg per chunk: %.1f us (%.2f ms)\n\n",
           duration_sequential_us.count() / (double)num_chunks,
           duration_sequential_us.count() / (double)num_chunks / 1000.0);

    // ========================================================================
    // Test 2: Parallel batch_get() (NEW approach)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Test 2: Parallel batch_get() (NEW approach)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Clear L2 again
    storage.evict_lru(cpu_limit);

    auto start_batch = high_resolution_clock::now();

    std::vector<thunder_kv_chunk *> results = storage.batch_get(keys);

    auto end_batch = high_resolution_clock::now();
    auto duration_batch_us = duration_cast<microseconds>(end_batch - start_batch);

    size_t batch_hits = 0;
    for (auto * result : results) {
        if (result != nullptr) {
            batch_hits++;
        }
    }

    printf("Results:\n");
    printf("  Chunks retrieved: %zu / %zu\n", batch_hits, num_chunks);
    printf("  Total time: %lld us (%.1f ms)\n", duration_batch_us.count(), duration_batch_us.count() / 1000.0);
    printf("  Avg per chunk: %.1f us (%.2f ms)\n\n",
           duration_batch_us.count() / (double)num_chunks,
           duration_batch_us.count() / (double)num_chunks / 1000.0);

    // ========================================================================
    // Summary
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Performance Summary\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    printf("┌───────────────────┬──────────────┬──────────────┬─────────────┐\n");
    printf("│ Metric            │ Sequential   │ Batch        │ Improvement │\n");
    printf("├───────────────────┼──────────────┼──────────────┼─────────────┤\n");
    printf("│ Total Time (us)   │   %8lld │   %8lld │   %7.1fx │\n",
           duration_sequential_us.count(), duration_batch_us.count(),
           duration_sequential_us.count() / (double)duration_batch_us.count());
    printf("│ Total Time (ms)   │   %8.1f │   %8.1f │   %7.1fx │\n",
           duration_sequential_us.count() / 1000.0, duration_batch_us.count() / 1000.0,
           duration_sequential_us.count() / (double)duration_batch_us.count());
    printf("│ Avg per chunk (us)│   %8.1f │   %8.1f │   %7.1fx │\n",
           duration_sequential_us.count() / (double)num_chunks,
           duration_batch_us.count() / (double)num_chunks,
           (duration_sequential_us.count() / (double)num_chunks) / (duration_batch_us.count() / (double)num_chunks));
    printf("└───────────────────┴──────────────┴──────────────┴─────────────┘\n\n");

    double improvement = 1.0 - (duration_batch_us.count() / (double)duration_sequential_us.count());
    printf("Total Improvement: %.1f%%\n\n", improvement * 100);

    if (improvement >= 0.45) {
        printf("✅ SUCCESS: Achieved target improvement (>45%%)!\n");
    } else if (improvement >= 0.30) {
        printf("⚠️  PARTIAL: Good improvement, but below 45%% target\n");
    } else {
        printf("❌ BELOW TARGET: Improvement < 30%%\n");
    }

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Test completed!\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Cleanup
    for (auto & chunk : chunks) {
        free(chunk.k_data);
        free(chunk.v_data);
    }

    return 0;
}
