// ThunderLLAMA QKV Projection Fusion - GQA Adapted
// Phase 2: K/V Fusion for GQA Models - Header
//
// Purpose: Fuse wk/wv weights into wkv for GQA models (wq stays separate)
// Reference: Qwen3-30B (GQA 8:1), Qwen3.5-35B (GQA 8:1)
// Author: Solar v2.0 (Strategist + Governor)
// Date: 2026-03-15 (GQA adaptation)

#pragma once

#include "llama-model.h"

// Fuse K/V for all applicable layers in the model (GQA adapted)
// Called after model loading completes
//
// Fusion mode:
// - GQA models: Fuse K/V only, Q stays separate
// - MHA models: Fuse K/V only (Q/K/V full fusion is future work)
//
// Conditions for K/V fusion:
// 1. Weights exist: wk && wv
// 2. Quantization >= Q5_K
// 3. LoRA not enabled on K/V: !wk_s && !wv_s
// 4. Config enabled: FUSED_QKV=1 in thunderllama.conf
//
// Phase 2 behavior:
// - Logs which layers would be fused
// - Reports K/V dimensions
// - Does not create wkv tensor yet (requires context access)
// - Does not modify forward pass yet
//
// Phase 2 TODO:
// - Create wkv tensor and copy weights
// - Allocate backend buffer
// - Modify graph: 2×MUL_MAT(wk/wv) → 1×MUL_MAT(wkv) + 2×SLICE
void llama_model_fuse_qkv(llama_model & model);
