# ThunderLLAMA

**Apple Silicon Paged Attention for llama.cpp**

> This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with enhanced Paged Attention support for Apple Silicon (M1/M2/M3/M4) GPUs.

## What's New

### Paged Attention for Apple Silicon

This fork enables efficient Paged Attention on Apple Metal GPUs, addressing the original implementation issue where paged attention was effectively disabled even when `LLAMA_PAGED_ATTENTION=1` was set.

#### Key Changes

| File | Changes |
|------|---------|
| `src/llama-block-pool.cpp/h` | **NEW** - Block pool implementation for paged KV cache |
| `src/llama-kv-cache.cpp/h` | Added block pool integration, fixed k_pool/v_pool vector access |
| `src/llama-graph.cpp` | Pass actual block_table tensor with `use_paged=1` |
| `src/llama-context.cpp` | Context-level paged attention support |

#### The Bug We Fixed

Previously, `ggml_flash_attn_ext_set_paged()` was called with:
- `block_table = nullptr`
- `use_paged = 0`

This caused **17x performance degradation** (4 t/s vs 70 t/s on 30B MoE models).

## Performance Results

### 30B MoE Model on Apple M4

| Mode | Generation Speed |
|------|------------------|
| Contiguous + Flash Attn | 72.62 t/s |
| **Paged + Flash Attn** | **77.79 t/s (+7%)** |

Paged attention is now **faster** than contiguous mode!

## Build Instructions

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools
- CMake

### Build with Paged Attention

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/ThunderLLAMA.git
cd ThunderLLAMA

# Build with Metal support (default on macOS)
cmake -B build
cmake --build build --config Release -j
```

Paged attention is **enabled by default** when building on macOS with Metal support.

## Usage

### Basic Usage

```bash
# Run with Paged Attention (default when available)
./build/bin/llama-cli -m /path/to/model.gguf -p "Hello, world!"

# Run benchmark
./build/bin/llama-bench -m /path/to/model.gguf
```

### Enable Flash Attention (Recommended)

Flash Attention is **required** for optimal paged attention performance:

```bash
./build/bin/llama-cli -m /path/to/model.gguf -fa 1 -p "Hello, world!"
```

### Environment Variables

```bash
# Force enable paged attention (enabled by default on Metal)
LLAMA_PAGED_ATTENTION=1 ./build/bin/llama-cli -m model.gguf -fa 1
```

## Technical Details

### Block Pool Architecture

Paged attention uses a block pool for efficient KV cache management:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Block Pool Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   block_table_gpu: Maps logical blocks to physical GPU memory  │
│   k_pool/v_pool:   Per-layer KV tensor pools (one per layer)   │
│   block_size:      Number of tokens per block                  │
│                                                                 │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│   │ Block 0 │     │ Block 1 │     │ Block N │                  │
│   │ [tok...]│     │ [tok...]│     │ [tok...]│                  │
│   └────┬────┘     └────┬────┘     └────┬────┘                  │
│        │               │               │                        │
│        └───────────────┴───────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│              block_table_gpu                                    │
│              [0, 1, ..., N]                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Flash Attention Integration

The Metal Flash Attention kernel supports paged mode through:

```cpp
ggml_flash_attn_ext_set_paged(
    cur,              // Flash attention node
    block_table,      // Block table tensor (was nullptr before fix)
    1,                // use_paged = 1 (was 0 before fix)
    block_size,       // Tokens per block
    block_stride_k,   // K block stride
    block_stride_v,   // V block stride
    token_stride_k,   // K token stride
    token_stride_v    // V token stride
);
```

## Comparison with Upstream

| Feature | Upstream llama.cpp | ThunderLLAMA |
|---------|-------------------|--------------|
| Paged Attention on Metal | ❌ Not working | ✅ Working |
| Block Pool | ❌ Not available | ✅ Implemented |
| k_pool/v_pool per-layer | ❌ Single tensor | ✅ Vector per layer |
| Flash Attention + Paged | ❌ Incompatible | ✅ Working together |

## Contributing

Contributions are welcome! Please open issues or pull requests on GitHub.

## License

Same as llama.cpp (MIT License)

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - The upstream project
- [vLLM Paged Attention](https://arxiv.org/abs/2309.06180) - Original paper

---

*ThunderLLAMA - Making Paged Attention roar on Apple Silicon* 🍎⚡
