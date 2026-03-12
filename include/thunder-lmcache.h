#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

    //
    // Thunder LMCache-Lite API (Phase 1)
    //
    // Lightweight KV cache chunk management inspired by LMCache.
    // Allows efficient reuse of KV cache across multiple sequences by chunking
    // and content-based hashing.
    //

    // ============================================================================
    // Constants
    // ============================================================================

    // Number of tokens per KV cache chunk
    // Chunks are 256-token granules that can be independently cached
    #define THUNDER_CHUNK_SIZE 256

    // ============================================================================
    // Data Structures
    // ============================================================================

    // Unique key for a KV cache chunk.
    // Identifies a chunk of KV cache by its content hash, layer, and position.
    //
    // - content_hash: xxHash64 of the prompt tokens in this chunk
    //                 Used for deduplication across sequences
    // - layer_idx:    Index of the transformer layer (0 to n_layers-1)
    // - chunk_start:  Starting token position of the chunk (always aligned to 256)
    //
    typedef struct {
        uint64_t content_hash;  // xxHash64 of prompt tokens
        int32_t  layer_idx;     // Layer index (0-N)
        int32_t  chunk_start;   // Chunk start position (256-token aligned)
    } thunder_kv_chunk_key;

    // Metadata and data for a cached KV chunk.
    // Stores the actual K and V tensor data along with cache metadata.
    //
    // - key:           Unique identifier for this chunk
    // - k_size:        Size in bytes of K tensor
    // - v_size:        Size in bytes of V tensor
    // - k_data:        Pointer to K tensor data (must remain valid while cached)
    // - v_data:        Pointer to V tensor data (must remain valid while cached)
    // - last_access_ns: Timestamp (nanoseconds since epoch) of last access
    //                   Used for LRU eviction policies
    // - access_count:  Number of times this chunk has been accessed
    //                  Used for popularity-based caching decisions
    //
    typedef struct {
        thunder_kv_chunk_key key;

        size_t   k_size;        // K tensor size in bytes
        size_t   v_size;        // V tensor size in bytes
        void *   k_data;        // Pointer to K tensor
        void *   v_data;        // Pointer to V tensor

        uint64_t last_access_ns;  // Last access timestamp (LRU)
        uint32_t access_count;     // Access count (popularity)
    } thunder_kv_chunk;

    // ============================================================================
    // C API Functions (implemented in thunder-lmcache.cpp)
    // ============================================================================

    // NOTE: C API functions are declared here for future integration.
    // Currently, the C++ class-based API is recommended for full functionality.

#ifdef __cplusplus
}  // extern "C"
#endif

// ============================================================================
// Forward Declarations (C++ classes)
// ============================================================================

#ifdef __cplusplus

    // Hashing utility for KV chunks.
    // Handles content-based hashing of prompt tokens using xxHash64.
    // Ensures consistent chunk identification across different inference paths.
    class ThunderChunkHasher;

    // Core storage and retrieval manager for KV chunks.
    // Implements the cache storage, LRU eviction, and chunk lookup logic.
    // Provides thread-safe access to cached chunks.
    class ThunderChunkStorage;

#endif // __cplusplus
