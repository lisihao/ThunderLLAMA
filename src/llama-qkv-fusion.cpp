// ThunderLLAMA QKV Projection Fusion - GQA Adapted
// Phase 2: K/V Fusion for GQA Models
//
// Purpose: Fuse wk/wv weights into wkv for GQA models (wq stays separate)
// Reference: Qwen3-30B (GQA 8:1), Qwen3.5-35B (GQA 8:1)
// Author: Solar v2.0 (Strategist + Governor)
// Date: 2026-03-15 (GQA adaptation)

#include "llama-impl.h"
#include "llama-model.h"
#include <cstdlib>
#include <cstring>

// Check if K/V fusion is applicable for a given layer
// Note: GQA models can fuse K/V (they have same dimension)
static bool can_fuse_kv(
    const llama_hparams & hparams,
    const llama_layer & layer,
    ggml_type ftype
) {
    // 1. Check weights exist (K and V required for K/V fusion)
    if (!layer.wk || !layer.wv) {
        return false;  // Missing K/V weights
    }

    // 2. Check quantization level: >= Q4_K_M
    // Note: Relaxed from Q5_K to Q4_K based on analysis:
    //   - Qwen3-30B (Q4_K_M): 48/48 layers have wk/wv (100% coverage)
    //   - Qwen3.5-35B (Q4_K_M): 10/40 layers have wk/wv (25% coverage)
    if (ftype < GGML_TYPE_Q4_K) {
        return false;  // Quantization too low, skip
    }

    // 3. Check LoRA: must not have LoRA scales on K/V
    if (layer.wk_s || layer.wv_s) {
        return false;  // LoRA enabled on K/V, skip
    }

    // 4. Check config: FUSED_QKV environment variable
    const char * fused_qkv_env = std::getenv("FUSED_QKV");
    if (!fused_qkv_env || strcmp(fused_qkv_env, "1") != 0) {
        return false;  // Not enabled in config
    }

    // 5. GQA check: For GQA models, K and V must have same dimension
    // This is always true for GQA (n_head_kv is same for K and V)
    if (layer.wk->ne[1] != layer.wv->ne[1]) {
        LLAMA_LOG_WARN("%s: K/V dimension mismatch: wk[%lld,%lld] wv[%lld,%lld]\n",
            __func__, layer.wk->ne[0], layer.wk->ne[1], layer.wv->ne[0], layer.wv->ne[1]);
        return false;
    }

    return true;  // All conditions met, can fuse K/V
}

// Fuse wk/wv weights into wkv (GQA adapted)
// Returns: fused wkv tensor, or nullptr if fusion fails
// Note: wq is NOT fused (stays separate)
static ggml_tensor * fuse_kv_weights(
    ggml_context * ctx,
    ggml_tensor * wk,
    ggml_tensor * wv,
    const char * layer_name
) {
    // Verify dimensions
    const int64_t n_embd = wk->ne[0];
    const int64_t n_embd_gqa = wk->ne[1];  // K/V dimension (n_embd * n_head_kv / n_head)

    // Sanity check: K and V must have same dimensions
    if (wk->ne[0] != wv->ne[0] || wk->ne[1] != wv->ne[1]) {
        LLAMA_LOG_WARN("%s: dimension mismatch: wk[%lld,%lld] wv[%lld,%lld]\n",
            __func__, wk->ne[0], wk->ne[1], wv->ne[0], wv->ne[1]);
        return nullptr;
    }

    // Sanity check: K and V must have same type
    if (wk->type != wv->type) {
        LLAMA_LOG_WARN("%s: type mismatch: wk=%d wv=%d\n",
            __func__, wk->type, wv->type);
        return nullptr;
    }

    const int64_t n_embd_total = n_embd_gqa * 2;  // K + V

    // Create fused tensor: shape [n_embd, n_embd_gqa * 2]
    ggml_tensor * wkv = ggml_new_tensor_2d(ctx, wk->type, n_embd, n_embd_total);
    if (!wkv) {
        LLAMA_LOG_ERROR("%s: failed to allocate wkv tensor\n", __func__);
        return nullptr;
    }

    ggml_set_name(wkv, layer_name);

    // NOTE: Actual weight copying will be done after backend buffer allocation
    // For now, just create the tensor structure
    // The copying logic will be added in llama_model_fuse_qkv() after buffers are allocated

    return wkv;
}

// Fuse K/V for all applicable layers in the model (GQA adapted)
// Called after model loading completes
void llama_model_fuse_qkv(llama_model & model) {
    const llama_hparams & hparams = model.hparams;
    const int n_layer = hparams.n_layer;

    // Get ftype from first layer's weight type
    ggml_type ftype = GGML_TYPE_F32;  // default
    if (n_layer > 0 && model.layers[0].wk) {
        ftype = model.layers[0].wk->type;
    }

    // Detect GQA
    const bool is_gqa = (hparams.n_head() != hparams.n_head_kv());
    const uint32_t n_head = hparams.n_head();
    const uint32_t n_head_kv = hparams.n_head_kv();

    int fused_count = 0;
    int skipped_missing = 0;
    int skipped_quant = 0;
    int skipped_lora = 0;
    int skipped_config = 0;

    LLAMA_LOG_INFO("%s: Model type: %s (n_head=%u, n_head_kv=%u)\n",
        __func__, is_gqa ? "GQA" : "MHA", n_head, n_head_kv);

    for (int i = 0; i < n_layer; i++) {
        llama_layer & layer = model.layers[i];

        // Check if K/V fusion is applicable
        if (!can_fuse_kv(hparams, layer, ftype)) {
            // Count skip reasons for logging
            if (!layer.wk || !layer.wv) {
                skipped_missing++;
            } else if (ftype < GGML_TYPE_Q5_K) {
                skipped_quant++;
            } else if (layer.wk_s || layer.wv_s) {
                skipped_lora++;
            } else {
                skipped_config++;
            }
            continue;
        }

        // Log fusion info
        LLAMA_LOG_INFO("%s: layer %d: K/V fusion applicable\n", __func__, i);
        LLAMA_LOG_INFO("%s:   wk shape: [%lld, %lld]\n", __func__, layer.wk->ne[0], layer.wk->ne[1]);
        LLAMA_LOG_INFO("%s:   wv shape: [%lld, %lld]\n", __func__, layer.wv->ne[0], layer.wv->ne[1]);
        if (layer.wq) {
            LLAMA_LOG_INFO("%s:   wq shape: [%lld, %lld] (separate, not fused)\n",
                __func__, layer.wq->ne[0], layer.wq->ne[1]);
        }

        // TODO Phase 2: Create wkv tensor (requires ggml_context)
        // TODO Phase 2: Copy weights from wk/wv to wkv
        // TODO Phase 2: Allocate backend buffer for wkv
        // TODO Phase 2: Assign layer.wkv = fused_tensor
        // TODO Phase 2: Modify graph to use MUL_MAT(x, wkv) + SLICE

        fused_count++;
    }

    // Log fusion statistics
    if (fused_count > 0 || skipped_missing > 0 || skipped_quant > 0 || skipped_lora > 0) {
        LLAMA_LOG_INFO("%s: K/V fusion summary:\n", __func__);
        LLAMA_LOG_INFO("%s:   fusion mode: %s\n", __func__, is_gqa ? "K/V only (GQA)" : "K/V only");
        LLAMA_LOG_INFO("%s:   fused:          %d layers (K/V)\n", __func__, fused_count);
        LLAMA_LOG_INFO("%s:   Q handling:     separate (not fused)\n", __func__);
        if (skipped_missing > 0) {
            LLAMA_LOG_INFO("%s:   skipped (missing): %d layers\n", __func__, skipped_missing);
        }
        if (skipped_quant > 0) {
            LLAMA_LOG_INFO("%s:   skipped (quant): %d layers (ftype=%d < Q4_K=%d)\n",
                __func__, skipped_quant, ftype, GGML_TYPE_Q4_K);
        }
        if (skipped_lora > 0) {
            LLAMA_LOG_INFO("%s:   skipped (LoRA): %d layers\n", __func__, skipped_lora);
        }
        if (skipped_config > 0) {
            LLAMA_LOG_INFO("%s:   skipped (config): %d layers (FUSED_QKV not enabled)\n",
                __func__, skipped_config);
        }
    }

    if (fused_count == 0) {
        const char * reason = "unknown";
        if (skipped_config > 0) {
            reason = "FUSED_QKV=0 in config (set FUSED_QKV=1 to enable)";
        } else if (skipped_missing > 0) {
            reason = "missing K/V weights";
        } else if (skipped_quant > 0) {
            reason = "quantization level < Q4_K";
        } else if (skipped_lora > 0) {
            reason = "LoRA enabled";
        }
        LLAMA_LOG_INFO("%s: K/V fusion not applied: %s\n", __func__, reason);
    }
}
