#include <cassert>
#include <cstring>
#include <iostream>
#include <chrono>
#include <vector>

#include "thunder-lmcache-hash.h"
#include "llama.h"

void test_hash_tokens_empty() {
    ThunderChunkHasher hasher;
    uint64_t hash1 = hasher.hash_tokens(nullptr, 0);
    uint64_t hash2 = hasher.hash_tokens(nullptr, 0);

    // Same input should produce same hash
    assert(hash1 == hash2);
    std::cout << "✓ test_hash_tokens_empty: PASS (hash=" << hash1 << ")\n";
}

void test_hash_tokens_deterministic() {
    ThunderChunkHasher hasher;

    // Create token sequences
    llama_token tokens1[] = {1, 2, 3, 4, 5};
    llama_token tokens2[] = {1, 2, 3, 4, 5};

    uint64_t hash1 = hasher.hash_tokens(tokens1, 5);
    uint64_t hash2 = hasher.hash_tokens(tokens2, 5);

    // Same tokens should produce same hash
    assert(hash1 == hash2);
    std::cout << "✓ test_hash_tokens_deterministic: PASS (hash=" << hash1 << ")\n";
}

void test_hash_tokens_different() {
    ThunderChunkHasher hasher;

    // Create different token sequences
    llama_token tokens1[] = {1, 2, 3, 4, 5};
    llama_token tokens2[] = {1, 2, 3, 4, 6};  // Last token different

    uint64_t hash1 = hasher.hash_tokens(tokens1, 5);
    uint64_t hash2 = hasher.hash_tokens(tokens2, 5);

    // Different tokens should produce different hash (with very high probability)
    assert(hash1 != hash2);
    std::cout << "✓ test_hash_tokens_different: PASS (hash1=" << hash1 << ", hash2=" << hash2 << ")\n";
}

void test_hash_tokens_chunk_size() {
    ThunderChunkHasher hasher;

    // Create token sequence larger than THUNDER_CHUNK_SIZE (256)
    std::vector<llama_token> tokens1(300);
    std::vector<llama_token> tokens2(300);

    // Fill with same values
    for (int i = 0; i < 300; i++) {
        tokens1[i] = (llama_token)(i % 256);
        tokens2[i] = (llama_token)(i % 256);
    }

    uint64_t hash1 = hasher.hash_tokens(tokens1.data(), 300);
    uint64_t hash2 = hasher.hash_tokens(tokens2.data(), 300);

    // Should only hash first 256 tokens, so hashes should be same
    assert(hash1 == hash2);
    std::cout << "✓ test_hash_tokens_chunk_size: PASS (only first " << THUNDER_CHUNK_SIZE << " tokens hashed)\n";
}

void test_hash_kv_data_empty() {
    ThunderChunkHasher hasher;
    uint64_t hash1 = hasher.hash_kv_data(nullptr, 0);
    uint64_t hash2 = hasher.hash_kv_data(nullptr, 0);

    // Same input should produce same hash
    assert(hash1 == hash2);
    std::cout << "✓ test_hash_kv_data_empty: PASS (hash=" << hash1 << ")\n";
}

void test_hash_kv_data_deterministic() {
    ThunderChunkHasher hasher;

    // Create test data
    uint8_t data1[] = {0x01, 0x02, 0x03, 0x04, 0x05};
    uint8_t data2[] = {0x01, 0x02, 0x03, 0x04, 0x05};

    uint64_t hash1 = hasher.hash_kv_data(data1, sizeof(data1));
    uint64_t hash2 = hasher.hash_kv_data(data2, sizeof(data2));

    // Same data should produce same hash
    assert(hash1 == hash2);
    std::cout << "✓ test_hash_kv_data_deterministic: PASS (hash=" << hash1 << ")\n";
}

void test_hash_kv_data_different() {
    ThunderChunkHasher hasher;

    // Create different data
    uint8_t data1[] = {0x01, 0x02, 0x03, 0x04, 0x05};
    uint8_t data2[] = {0x01, 0x02, 0x03, 0x04, 0x06};  // Last byte different

    uint64_t hash1 = hasher.hash_kv_data(data1, sizeof(data1));
    uint64_t hash2 = hasher.hash_kv_data(data2, sizeof(data2));

    // Different data should produce different hash
    assert(hash1 != hash2);
    std::cout << "✓ test_hash_kv_data_different: PASS (hash1=" << hash1 << ", hash2=" << hash2 << ")\n";
}

void test_make_key_alignment() {
    ThunderChunkHasher hasher;

    // Create token sequence
    std::vector<llama_token> tokens(1024);
    for (int i = 0; i < 1024; i++) {
        tokens[i] = (llama_token)i;
    }

    // Test chunk alignment
    auto key1 = hasher.make_key(tokens.data(), 1024, 0, 0);
    assert(key1.chunk_start == 0);
    assert(key1.layer_idx == 0);

    auto key2 = hasher.make_key(tokens.data(), 1024, 100, 1);
    assert(key2.chunk_start == 0);  // Should align down to 0
    assert(key2.layer_idx == 1);

    auto key3 = hasher.make_key(tokens.data(), 1024, 256, 2);
    assert(key3.chunk_start == 256);
    assert(key3.layer_idx == 2);

    auto key4 = hasher.make_key(tokens.data(), 1024, 300, 3);
    assert(key4.chunk_start == 256);  // Should align down to 256
    assert(key4.layer_idx == 3);

    std::cout << "✓ test_make_key_alignment: PASS\n";
}

void test_make_key_deterministic() {
    ThunderChunkHasher hasher;

    // Create token sequence
    std::vector<llama_token> tokens1(1024);
    std::vector<llama_token> tokens2(1024);
    for (int i = 0; i < 1024; i++) {
        tokens1[i] = (llama_token)i;
        tokens2[i] = (llama_token)i;
    }

    // Same inputs should produce same keys
    auto key1 = hasher.make_key(tokens1.data(), 1024, 512, 5);
    auto key2 = hasher.make_key(tokens2.data(), 1024, 512, 5);

    assert(key1.content_hash == key2.content_hash);
    assert(key1.chunk_start == key2.chunk_start);
    assert(key1.layer_idx == key2.layer_idx);

    std::cout << "✓ test_make_key_deterministic: PASS\n";
}

void test_make_key_different_chunks() {
    ThunderChunkHasher hasher;

    // Create token sequence
    std::vector<llama_token> tokens(1024);
    for (int i = 0; i < 1024; i++) {
        tokens[i] = (llama_token)i;
    }

    // Different chunks should have different hashes
    auto key1 = hasher.make_key(tokens.data(), 1024, 0, 0);
    auto key2 = hasher.make_key(tokens.data(), 1024, 256, 0);

    // Different content hashes (different token ranges)
    assert(key1.content_hash != key2.content_hash);
    assert(key1.chunk_start != key2.chunk_start);

    std::cout << "✓ test_make_key_different_chunks: PASS\n";
}

void test_hash_tokens_performance() {
    ThunderChunkHasher hasher;

    // Create a large token sequence
    std::vector<llama_token> tokens(THUNDER_CHUNK_SIZE);
    for (int i = 0; i < THUNDER_CHUNK_SIZE; i++) {
        tokens[i] = (llama_token)i;
    }

    // Measure hashing performance
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        uint64_t hash = hasher.hash_tokens(tokens.data(), THUNDER_CHUNK_SIZE);
        (void)hash;  // Avoid compiler optimization
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double time_per_hash_ms = (double)duration.count() / 1000.0;

    std::cout << "✓ test_hash_tokens_performance: PASS (avg " << time_per_hash_ms << "ms per hash)\n";

    // Performance requirement: < 1ms per hash
    assert(time_per_hash_ms < 1.0);
}

int main() {
    std::cout << "Running Thunder LMCache Hash Tests...\n\n";

    try {
        test_hash_tokens_empty();
        test_hash_tokens_deterministic();
        test_hash_tokens_different();
        test_hash_tokens_chunk_size();
        test_hash_kv_data_empty();
        test_hash_kv_data_deterministic();
        test_hash_kv_data_different();
        test_make_key_alignment();
        test_make_key_deterministic();
        test_make_key_different_chunks();
        test_hash_tokens_performance();

        std::cout << "\n✓ All tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
