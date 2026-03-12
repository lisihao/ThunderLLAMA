#include "thunder-lmcache-hash.h"
#include <xxhash.h>

#include <algorithm>
#include <cstring>

// ============================================================================
// ThunderChunkHasher Implementation
// ============================================================================

uint64_t ThunderChunkHasher::hash_tokens(const llama_token *tokens, size_t n_tokens) {
    // Handle empty sequence
    if (tokens == nullptr || n_tokens == 0) {
        // Return a deterministic hash for empty sequence
        return XXH64(nullptr, 0, 0);
    }

    // Determine how many tokens to hash (min of actual tokens and chunk size)
    size_t tokens_to_hash = std::min(n_tokens, (size_t)THUNDER_CHUNK_SIZE);

    // Hash tokens as raw bytes
    // Each llama_token is an int32_t, so multiply by sizeof(llama_token)
    size_t bytes_to_hash = tokens_to_hash * sizeof(llama_token);

    // Use xxHash64 with seed 0 for deterministic results
    uint64_t hash = XXH64(tokens, bytes_to_hash, 0);

    return hash;
}

uint64_t ThunderChunkHasher::hash_kv_data(const void *data, size_t size) {
    // Handle empty data
    if (data == nullptr || size == 0) {
        // Return a deterministic hash for empty data
        return XXH64(nullptr, 0, 0);
    }

    // Hash raw bytes using xxHash64
    uint64_t hash = XXH64(data, size, 0);

    return hash;
}

thunder_kv_chunk_key ThunderChunkHasher::make_key(
    const llama_token *tokens,
    size_t total_len,
    size_t chunk_start,
    int32_t layer_idx
) {
    thunder_kv_chunk_key key = {};

    // Align chunk_start to the nearest THUNDER_CHUNK_SIZE boundary (round down)
    size_t aligned_start = align_chunk_start(chunk_start);

    // Determine chunk end (may be less than THUNDER_CHUNK_SIZE for last chunk)
    size_t chunk_end = std::min(aligned_start + THUNDER_CHUNK_SIZE, total_len);

    // Compute number of tokens in this chunk
    size_t chunk_tokens = (chunk_end > aligned_start) ? (chunk_end - aligned_start) : 0;

    // Hash the tokens in this specific chunk
    if (tokens != nullptr && chunk_tokens > 0) {
        // Hash only the tokens from aligned_start to chunk_end
        const llama_token *chunk_tokens_ptr = tokens + aligned_start;
        key.content_hash = XXH64(
            chunk_tokens_ptr,
            chunk_tokens * sizeof(llama_token),
            0  // seed
        );
    } else {
        // Empty chunk or invalid input
        key.content_hash = XXH64(nullptr, 0, 0);
    }

    // Set layer index and chunk start position
    key.layer_idx = layer_idx;
    key.chunk_start = aligned_start;

    return key;
}
