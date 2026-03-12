# Thunder LMCache K/V Tensor Copy Implementation Status

**Date**: 2026-03-12
**Status**: Skeleton Implemented, Core Logic TODO

## ✅ Completed (Phase 1 & 2)

### 1. Code Investigation
- [x] Understood K/V cache structure in `llama_kv_cache`
- [x] Found tensor access API: `ggml_backend_tensor_get()` / `ggml_backend_tensor_set()`
- [x] Located K/V tensors in `layers[ikv].k` and `layers[ikv].v`
- [x] Understood tensor layout: `[n_embd_k_gqa, kv_size, n_stream]`

### 2. Infrastructure
- [x] Added `#include "llama-kv-cache.h"` to `llama-context.cpp`
- [x] Added public getters in `llama_kv_cache`:
  - `ggml_tensor * get_layer_k(int32_t il) const`
  - `ggml_tensor * get_layer_v(int32_t il) const`
- [x] Implemented basic skeleton for cache hit/miss hooks
- [x] Compilation successful

### 3. Hook Structure
```cpp
// Before processing (line ~1195)
if (lmcache_enabled && ubatch.n_tokens >= THUNDER_CHUNK_SIZE) {
    auto * kv_cache = dynamic_cast<llama_kv_cache *>(memory.get());
    if (kv_cache) {
        // For each layer, for each chunk:
        //   1. Generate chunk key
        //   2. Check cache
        //   3. If hit: restore K/V (TODO)
    }
}

// After processing (line ~1235)
if (lmcache_enabled && ubatch.n_tokens >= THUNDER_CHUNK_SIZE) {
    auto * kv_cache = dynamic_cast<llama_kv_cache *>(memory.get());
    if (kv_cache) {
        // For each layer, for each chunk:
        //   1. Generate chunk key
        //   2. If not cached: extract K/V and store (TODO)
    }
}
```

## ❌ TODO (Phase 3 - Core Logic)

### Critical Missing Pieces

#### 1. Calculate Tensor Offset and Size
**Challenge**: Need to calculate the correct byte offset for each 256-token chunk.

**Tensor Layout**:
- K tensor: `[n_embd_k_gqa, kv_size, n_stream]`
- V tensor: similar

**Required Calculation**:
```cpp
// Get dimensions
size_t n_embd_k_gqa = k_tensor->ne[0];
size_t element_size = ggml_element_size(k_tensor);

// Calculate chunk offset and size
size_t chunk_offset = chunk_start * n_embd_k_gqa * element_size;
size_t chunk_size = THUNDER_CHUNK_SIZE * n_embd_k_gqa * element_size;
```

**Issue**: Need to verify this calculation is correct for the tensor layout.

#### 2. Restore K/V (Cache Hit)
**Location**: `src/llama-context.cpp:~1210`

**Required Steps**:
1. Calculate offset and size (see above)
2. Call `ggml_backend_tensor_set()`:
   ```cpp
   ggml_backend_tensor_set(k_tensor, cached->k_data, chunk_offset, cached->k_size);
   ggml_backend_tensor_set(v_tensor, cached->v_data, chunk_offset, cached->v_size);
   ```

**Issue**: Need to ensure backend is ready and tensor is allocated.

#### 3. Extract and Store K/V (Cache Miss)
**Location**: `src/llama-context.cpp:~1255`

**Required Steps**:
1. Calculate offset and size
2. Allocate memory:
   ```cpp
   thunder_kv_chunk chunk;
   chunk.key = key;
   chunk.k_size = chunk_size;
   chunk.v_size = chunk_size;  // May differ for V if transposed
   chunk.k_data = malloc(chunk.k_size);
   chunk.v_data = malloc(chunk.v_size);
   ```
3. Extract data:
   ```cpp
   ggml_backend_tensor_get(k_tensor, chunk.k_data, chunk_offset, chunk.k_size);
   ggml_backend_tensor_get(v_tensor, chunk.v_data, chunk_offset, chunk.v_size);
   ```
4. Store:
   ```cpp
   lmcache_storage->put(chunk);
   ```

**Issue**: Need to handle memory ownership (who frees `k_data`/`v_data`?).

### Complexity Issues

#### A. V Tensor Transposition
The V cache may be transposed (`v_trans` flag in `llama_kv_cache`).
- Need to check `kv_cache->v_trans`
- If transposed, offset calculation differs

#### B. Multi-Stream Support
The cache supports multiple streams (`n_stream`).
- Need to determine which stream to use for the current ubatch
- Offset calculation needs to account for stream

#### C. Backend Synchronization
Tensors are on GPU backends (Metal/CUDA).
- `ggml_backend_tensor_get/set` may be async
- May need `ggml_backend_synchronize()`

#### D. Partial Chunks
If `ubatch.n_tokens` is not a multiple of 256:
- Last chunk may be incomplete
- Should we cache partial chunks?

### Example Implementation (Simplified)

```cpp
// Cache Hit: Restore K/V
if (cached && cached->k_data && cached->v_data) {
    ggml_tensor * k_tensor = kv_cache->get_layer_k(il);
    ggml_tensor * v_tensor = kv_cache->get_layer_v(il);

    // Calculate offset
    size_t n_embd_k = k_tensor->ne[0];
    size_t element_size = ggml_element_size(k_tensor);
    size_t offset = chunk_start * n_embd_k * element_size;

    // Restore
    ggml_backend_tensor_set(k_tensor, cached->k_data, offset, cached->k_size);
    ggml_backend_tensor_set(v_tensor, cached->v_data, offset, cached->v_size);

    LLAMA_LOG_DEBUG("LMCache restored: layer=%d, chunk_start=%zu\n", il, chunk_start);
}
```

## Testing Plan (Phase 3)

1. **Unit Test**: Single-layer, single-chunk copy
   - Disable all layers except layer 0
   - Use a 256-token prompt
   - Verify K/V data matches

2. **Integration Test**: Multi-layer, multi-chunk
   - Use a 512-token prompt (2 chunks)
   - Verify cache hit on second run

3. **Stress Test**: Large prompt (2048 tokens)
   - Check memory usage
   - Verify no crashes

4. **Quality Test**: Compare output with/without LMCache
   - Must be identical

## Estimated Complexity

- **Easy (2-4 hours)**: If offset calculation is straightforward
- **Medium (4-8 hours)**: If V transposition requires special handling
- **Hard (8+ hours)**: If backend synchronization causes issues

## Next Steps

1. **Implement offset/size calculation** (priority 1)
2. **Implement cache hit restore** (priority 2)
3. **Implement cache miss extract** (priority 3)
4. **Add error handling** (priority 4)
5. **Test and validate** (priority 5)

## Current Code Locations

- **Hook (before)**: `src/llama-context.cpp:1195-1226`
- **Hook (after)**: `src/llama-context.cpp:1235-1269`
- **Getters**: `src/llama-kv-cache.cpp:1259-1267`
- **Declarations**: `src/llama-kv-cache.h:161-162`

---

**Conclusion**: The framework is ready. The core tensor copy logic is the remaining challenge.
This requires understanding:
1. Tensor memory layout
2. Offset calculation for chunked access
3. Backend API for async tensor operations

**Recommendation**: Start with a simple single-chunk test case to validate the approach.
