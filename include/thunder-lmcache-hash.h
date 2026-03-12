#pragma once

#include <stddef.h>
#include <stdint.h>
#include "llama.h"
#include "thunder-lmcache.h"

#ifdef __cplusplus

/**
 * @brief Hashing utility for KV chunks.
 *
 * Handles content-based hashing of prompt tokens using xxHash64.
 * Ensures consistent chunk identification across different inference paths.
 *
 * Features:
 * - Fast xxHash64 for token sequences
 * - KV data integrity verification
 * - Automatic chunk boundary alignment
 * - Deterministic chunk key generation
 *
 * Thread safety: Each thread should have its own ThunderChunkHasher instance,
 * or use external synchronization for concurrent hash operations.
 */
class ThunderChunkHasher {
public:
    /**
     * @brief Construct a new ThunderChunkHasher.
     *
     * Initializes the hasher with no internal state.
     * Thread-safe: No shared state between instances.
     */
    ThunderChunkHasher() = default;

    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~ThunderChunkHasher() = default;

    /**
     * @brief Hash a sequence of tokens using xxHash64.
     *
     * Computes the xxHash64 value for a sequence of llama tokens.
     * Only hashes the first THUNDER_CHUNK_SIZE tokens (or fewer if n_tokens < THUNDER_CHUNK_SIZE).
     *
     * @param tokens     Pointer to token array. Must not be NULL if n_tokens > 0.
     * @param n_tokens   Number of tokens in the sequence. May be 0.
     *
     * @return           xxHash64 hash value. Deterministic: same input yields same output.
     *
     * @note The hash is computed over the token IDs only, not the token type or attributes.
     * @note Empty sequence (n_tokens == 0) returns a well-defined hash (not all zeros).
     * @note If n_tokens > THUNDER_CHUNK_SIZE, only the first 256 tokens are hashed.
     */
    uint64_t hash_tokens(const llama_token *tokens, size_t n_tokens);

    /**
     * @brief Hash raw KV tensor data for integrity verification.
     *
     * Computes the xxHash64 value for raw tensor data (K or V matrices).
     * Used to validate that cached KV data has not been corrupted or modified.
     *
     * @param data       Pointer to tensor data. Must not be NULL if size > 0.
     * @param size       Size of tensor data in bytes. May be 0.
     *
     * @return           xxHash64 hash value. Deterministic: same data yields same output.
     *
     * @note This is a byte-level hash, sensitive to any changes in the raw data.
     * @note Empty data (size == 0) returns a well-defined hash.
     */
    uint64_t hash_kv_data(const void *data, size_t size);

    /**
     * @brief Generate a complete and unique KV chunk key.
     *
     * Creates a thunder_kv_chunk_key that uniquely identifies a chunk of KV cache.
     * The chunk_start position is automatically aligned to THUNDER_CHUNK_SIZE boundary.
     *
     * @param tokens      Pointer to complete token sequence. Must not be NULL if total_len > 0.
     * @param total_len   Total number of tokens in the sequence.
     * @param chunk_start Desired starting position of this chunk (will be aligned).
     * @param layer_idx   Transformer layer index (0 to n_layers-1).
     *
     * @return            A thunder_kv_chunk_key with:
     *                    - content_hash: hash of tokens [chunk_start_aligned, chunk_start_aligned + THUNDER_CHUNK_SIZE)
     *                    - layer_idx: copied from parameter
     *                    - chunk_start: aligned chunk_start (rounded down to nearest 256-token boundary)
     *
     * @note The content_hash is computed only from the tokens within this specific chunk,
     *       not the entire sequence. This allows chunks to be cached independently.
     * @note If the final chunk has fewer than THUNDER_CHUNK_SIZE tokens, it is still hashed correctly.
     * @note chunk_start is always rounded down to the nearest multiple of THUNDER_CHUNK_SIZE.
     *
     * @example
     *   // Create key for chunk starting at token 512 of layer 5
     *   auto key = hasher.make_key(tokens, 1024, 512, 5);
     *   // key.chunk_start == 512 (already aligned)
     *   // key.layer_idx == 5
     *   // key.content_hash = xxHash64(tokens[512...767])
     */
    thunder_kv_chunk_key make_key(
        const llama_token *tokens,
        size_t total_len,
        size_t chunk_start,
        int32_t layer_idx
    );

private:
    /**
     * @brief Helper: align a position down to the nearest chunk boundary.
     *
     * @param pos        Position to align.
     * @return           pos rounded down to nearest multiple of THUNDER_CHUNK_SIZE.
     */
    static inline size_t align_chunk_start(size_t pos) {
        return (pos / THUNDER_CHUNK_SIZE) * THUNDER_CHUNK_SIZE;
    }

    /**
     * @brief Helper: get the end position of a chunk.
     *
     * @param chunk_start The aligned start position of the chunk.
     * @return            chunk_start + THUNDER_CHUNK_SIZE (or less if it's the last chunk).
     */
    static inline size_t get_chunk_end(size_t chunk_start) {
        return chunk_start + THUNDER_CHUNK_SIZE;
    }
};

#endif // __cplusplus
