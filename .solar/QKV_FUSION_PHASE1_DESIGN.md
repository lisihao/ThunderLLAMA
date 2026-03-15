# QKV Fusion Phase 1: 模型加载器设计

> **设计日期**: 2026-03-15
> **负责人**: Solar (战略家 + 治理官双签)
> **实施者**: 建设者 (glm-5) - 待委派

---

## 一、代码分析结果

### 1.1 现有架构发现

**关键发现**：
- ✅ **已有融合实现**：PHI2, PHI3, QWEN2MOE 等模型原生使用 `layer.wqkv` tensor
- ✅ **融合模式参考**：`phi2.cpp` (line 37-46) 展示了标准融合用法
- ✅ **分离模式**：LLAMA, Qwen3.5 等使用分离的 `wq/wk/wv`

**代码位置**：
1. **模型加载**: `src/llama-model.cpp` (line 2736-3220)
2. **Forward 实现**: `src/models/phi2.cpp`, `src/models/llama.cpp`
3. **Tensor 名称定义**: `src/llama-arch.cpp` (line 372)

### 1.2 融合逻辑分析（PHI2 参考）

**PHI2 使用 wqkv 的方式** (`phi2.cpp:37-46`):

```cpp
// 1. 融合的矩阵乘法
cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
cb(cur, "wqkv", il);

// 2. 通过 ggml_view_3d 切片提取 Q/K/V
Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens,
                    n_embd_head*sizeof(float), cur->nb[1],
                    0*sizeof(float)*(n_embd));  // offset=0

Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens,
                    n_embd_head*sizeof(float), cur->nb[1],
                    1*sizeof(float)*(n_embd));  // offset=n_embd

Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens,
                    n_embd_head*sizeof(float), cur->nb[1],
                    1*sizeof(float)*(n_embd + n_embd_gqa));  // offset=n_embd+n_embd_gqa
```

**关键点**：
- `ggml_view_3d` 是零开销操作（只修改 tensor 的视图，不复制数据）
- offset 参数指向 wqkv 输出中的不同位置
- Q, K, V 顺序拼接在输出中

---

## 二、Phase 1 设计方案

### 2.1 目标

在模型加载后，动态融合已有的 wq/wk/wv tensor 为 wqkv（仅限支持的模型）。

### 2.2 实现位置

**文件**: `src/llama-model.cpp`

**函数**: 新增 `llama_model_fuse_qkv()` 函数，在模型加载后调用

### 2.3 融合条件检查

**函数签名**:
```cpp
bool can_fuse_qkv(const llama_hparams & hparams, const llama_layer & layer, ggml_type ftype);
```

**检查项**:
1. **非 GQA**: `hparams.n_head == hparams.n_head_kv`
   - GQA 模型的 K/V 维度与 Q 不同，无法简单融合

2. **权重存在**: `layer.wq && layer.wk && layer.wv`
   - 确保三个权重 tensor 都已加载

3. **量化等级**: `ftype >= LLAMA_FTYPE_MOSTLY_Q5_K_M`
   - 只对 Q5_K 及以上启用（低量化收益不明显）

4. **未启用 LoRA**: `!layer.wq_s && !layer.wk_s && !layer.wv_s`
   - LoRA 需要单独的权重，无法融合

5. **配置启用**: `thunderllama.conf` 中 `FUSED_QKV=1`

**伪代码**:
```cpp
bool can_fuse_qkv(const llama_hparams & hparams, const llama_layer & layer, ggml_type ftype) {
    // 1. 检查 GQA
    if (hparams.n_head != hparams.n_head_kv) {
        return false;  // GQA 模型，降级
    }

    // 2. 检查权重存在
    if (!layer.wq || !layer.wk || !layer.wv) {
        return false;  // 权重缺失
    }

    // 3. 检查量化等级
    if (ftype < LLAMA_FTYPE_MOSTLY_Q5_K_M) {
        return false;  // 低量化，降级
    }

    // 4. 检查 LoRA
    if (layer.wq_s || layer.wk_s || layer.wv_s) {
        return false;  // LoRA 启用，降级
    }

    // 5. 检查配置
    const char * fused_qkv_env = getenv("FUSED_QKV");
    if (!fused_qkv_env || strcmp(fused_qkv_env, "1") != 0) {
        return false;  // 配置未启用
    }

    return true;  // 可以融合
}
```

### 2.4 权重融合逻辑

**函数签名**:
```cpp
ggml_tensor * fuse_qkv_weights(
    ggml_context * ctx,
    ggml_tensor * wq,
    ggml_tensor * wk,
    ggml_tensor * wv,
    const char * layer_name
);
```

**实现步骤**:

1. **验证维度**:
   ```cpp
   assert(wq->ne[0] == wk->ne[0] && wk->ne[0] == wv->ne[0]);  // n_embd 相同
   assert(wq->ne[1] == wk->ne[1] && wk->ne[1] == wv->ne[1]);  // n_embd 相同（非 GQA）
   ```

2. **创建融合 tensor**:
   ```cpp
   int64_t n_embd = wq->ne[0];
   int64_t n_embd_total = n_embd * 3;  // Q + K + V

   ggml_tensor * wqkv = ggml_new_tensor_2d(ctx, wq->type, n_embd, n_embd_total);
   ggml_set_name(wqkv, layer_name);  // e.g. "blk.0.attn_qkv.weight"
   ```

3. **复制权重数据**（按行拼接）:
   ```cpp
   // GGML tensor shape: [n_embd, n_embd_total]
   // Layout: [Q_rows | K_rows | V_rows]
   // 每个部分 n_embd 行

   // 复制 Q (offset=0)
   size_t row_bytes = ggml_row_size(wq->type, n_embd);
   for (int64_t row = 0; row < n_embd; row++) {
       void * dst = (char*)wqkv->data + (row + 0*n_embd) * wqkv->nb[1];
       void * src = (char*)wq->data + row * wq->nb[1];
       memcpy(dst, src, row_bytes);
   }

   // 复制 K (offset=n_embd)
   for (int64_t row = 0; row < n_embd; row++) {
       void * dst = (char*)wqkv->data + (row + 1*n_embd) * wqkv->nb[1];
       void * src = (char*)wk->data + row * wk->nb[1];
       memcpy(dst, src, row_bytes);
   }

   // 复制 V (offset=2*n_embd)
   for (int64_t row = 0; row < n_embd; row++) {
       void * dst = (char*)wqkv->data + (row + 2*n_embd) * wqkv->nb[1];
       void * src = (char*)wv->data + row * wv->nb[1];
       memcpy(dst, src, row_bytes);
   }
   ```

4. **设置 tensor 属性**:
   ```cpp
   wqkv->backend = wq->backend;  // 保持后端一致
   wqkv->buffer = nullptr;       // 稍后分配
   ```

### 2.5 模型修改流程

**主函数**: `llama_model_fuse_qkv(llama_model & model)`

**调用时机**: 在 `llama_model_load()` 完成后，所有权重加载完毕时

**伪代码**:
```cpp
void llama_model_fuse_qkv(llama_model & model) {
    const llama_hparams & hparams = model.hparams;
    ggml_type ftype = model.ftype;

    int fused_count = 0;

    for (int i = 0; i < hparams.n_layer; i++) {
        llama_layer & layer = model.layers[i];

        // 检查是否可以融合
        if (!can_fuse_qkv(hparams, layer, ftype)) {
            continue;  // 跳过此层
        }

        // 创建临时 context 用于融合
        struct ggml_init_params params = {
            .mem_size   = ggml_tensor_overhead() * 3,  // wqkv
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * ctx_fuse = ggml_init(params);

        // 融合权重
        char name[128];
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
        ggml_tensor * wqkv = fuse_qkv_weights(ctx_fuse, layer.wq, layer.wk, layer.wv, name);

        // 分配 backend buffer
        // (此处需要调用 ggml_backend 的分配 API，具体实现待建设者补充)

        // 保存融合后的 tensor
        layer.wqkv = wqkv;

        // 保留原始 tensor（向后兼容）
        // layer.wq/wk/wv 仍然可用，fallback 时使用

        fused_count++;
    }

    if (fused_count > 0) {
        fprintf(stderr, "%s: fused QKV for %d layers\n", __func__, fused_count);
    }
}
```

---

## 三、配置文件集成

### 3.1 thunderllama.conf

**新增配置项**:
```bash
# QKV Projection Fusion (experimental)
# 0 = disabled (default)
# 1 = enabled (fuse wq/wk/wv to wqkv for non-GQA models)
FUSED_QKV=0
```

### 3.2 config-parser.h

**修改点** (`common/config-parser.h`):

```cpp
// 新增环境变量映射
if (strcmp(key, "FUSED_QKV") == 0) {
    setenv("FUSED_QKV", value, 1);
}
```

**无需修改 `common_params`**：
- 使用环境变量传递（与 METAL_FUSION 一致）
- 不破坏现有 API

---

## 四、风险点与边界情况

### 4.1 GQA 检测失败

**风险**: 误将 GQA 模型当作非 GQA 融合

**缓解**:
- 在 `can_fuse_qkv` 中严格检查 `n_head == n_head_kv`
- 添加维度 assertion (`wq->ne[1] == wk->ne[1]`)

### 4.2 量化等级误判

**风险**: 对 Q4_0 模型启用融合，收益不明显

**缓解**:
- `ftype >= LLAMA_FTYPE_MOSTLY_Q5_K_M` 检查
- 默认 `FUSED_QKV=0`，需手动启用

### 4.3 内存分配失败

**风险**: 融合后的 wqkv tensor 内存分配失败

**缓解**:
- 检查 `ggml_new_tensor_2d` 返回值
- 失败时回退到原始 wq/wk/wv（保留原 tensor）

### 4.4 LoRA 兼容性

**风险**: 启用 LoRA 后融合失败

**缓解**:
- `can_fuse_qkv` 中检查 `wq_s/wk_s/wv_s`
- LoRA 自动降级到分离模式

### 4.5 正确性验证

**风险**: 融合后计算结果与原始不一致

**缓解方案**:
- **Phase 1 只创建 wqkv tensor，不修改 forward pass**
- Forward 仍使用 `wq/wk/wv`（fallback 模式）
- Phase 2 再替换计算图（独立验证）

---

## 五、验证方案

### 5.1 编译验证

```bash
cd /Users/lisihao/ThunderLLAMA
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

**预期**: 编译通过，无警告

### 5.2 维度检查

**测试**:
```bash
FUSED_QKV=1 ./build/bin/llama-server -m ~/models/qwen3.5-30b.Q5_K_M.gguf -c 4096 -ngl 99 --verbose
```

**检查日志**:
```
llama_model_fuse_qkv: checking layer 0
llama_model_fuse_qkv: n_head=40, n_head_kv=40 (non-GQA) ✓
llama_model_fuse_qkv: wq shape: [3584, 3584]
llama_model_fuse_qkv: wk shape: [3584, 3584]
llama_model_fuse_qkv: wv shape: [3584, 3584]
llama_model_fuse_qkv: wqkv shape: [3584, 10752] (3x3584) ✓
llama_model_fuse_qkv: fused QKV for 48 layers
```

### 5.3 降级测试

**测试 1: GQA 模型（DeepSeek-V3）**:
```bash
FUSED_QKV=1 ./build/bin/llama-server -m ~/models/deepseek-v3.Q5_K_M.gguf -c 4096 -ngl 99 --verbose
```

**预期日志**:
```
llama_model_fuse_qkv: layer 0: n_head=128, n_head_kv=16 (GQA detected, skipping)
llama_model_fuse_qkv: fused QKV for 0 layers (all skipped due to GQA)
```

**测试 2: 低量化（Q4_0）**:
```bash
FUSED_QKV=1 ./build/bin/llama-server -m ~/models/qwen3.5-30b.Q4_0.gguf -c 4096 -ngl 99 --verbose
```

**预期日志**:
```
llama_model_fuse_qkv: layer 0: ftype=Q4_0 < Q5_K_M (skipping)
llama_model_fuse_qkv: fused QKV for 0 layers (all skipped due to low quantization)
```

### 5.4 内存检查

**使用 Instruments (macOS)**:
```bash
instruments -t Leaks -D /tmp/llama-leaks.trace \
  ./build/bin/llama-server -m ~/models/qwen3.5-30b.Q5_K_M.gguf -c 4096 -ngl 99
```

**检查点**:
- 无内存泄漏
- wqkv tensor 内存正确分配

---

## 六、与现有代码的兼容性

### 6.1 向后兼容

- ✅ **默认关闭**: `FUSED_QKV=0` 默认，不影响现有行为
- ✅ **保留原 tensor**: `wq/wk/wv` 仍然存在，fallback 可用
- ✅ **无 API 变更**: 不修改 `llama_model` 结构体的公开 API

### 6.2 与 Phase 2 的接口

**Phase 1 交付**:
- `layer.wqkv` tensor 已创建并填充数据
- `layer.wq/wk/wv` 仍然可用

**Phase 2 使用**:
- 在 forward graph 中检测 `layer.wqkv` 是否存在
- 存在 → 使用融合模式（`MUL_MAT(x, wqkv) + SLICE`）
- 不存在 → fallback 到分离模式（`3×MUL_MAT(x, wq/wk/wv)`）

---

## 七、Definition of Done

- [ ] `can_fuse_qkv()` 函数实现（5 项检查）
- [ ] `fuse_qkv_weights()` 函数实现（权重拼接）
- [ ] `llama_model_fuse_qkv()` 函数实现（遍历所有层）
- [ ] `thunderllama.conf` 添加 `FUSED_QKV` 配置项
- [ ] `config-parser.h` 添加环境变量映射
- [ ] 编译通过（无警告）
- [ ] 维度检查通过（Qwen3.5-30B Q5_K_M）
- [ ] GQA 降级验证（DeepSeek-V3）
- [ ] 低量化降级验证（Q4_0）
- [ ] 内存无泄漏（Instruments 验证）

---

## 八、委派计划

**实施者**: 建设者 (glm-5)

**输入**:
- 本设计文档
- 参考代码：`src/models/phi2.cpp`, `src/models/llama.cpp`
- 配置系统：`common/config-parser.h`

**输出**:
- 修改后的 `src/llama-model.cpp`（新增融合函数）
- 修改后的 `common/config-parser.h`（环境变量映射）
- 更新后的 `thunderllama.conf`（新配置项）

**验收标准**:
- 编译通过
- 3 个降级测试通过（GQA/低量化/LoRA）
- 无内存泄漏

---

## 九、审计签名

**战略家签名**: ✅ Solar (架构设计完成，逻辑清晰，可执行)

**治理官签名**: ⚠️ 等待建设者实现后验证

**审计要点**:
1. 融合逻辑是否与 PHI2 参考一致？
2. 降级条件是否覆盖所有边界情况？
3. 内存管理是否正确（无泄漏）？
4. 向后兼容性是否保持（默认关闭，fallback 可用）？

**最终 Go/No-Go**: 待实现验证
