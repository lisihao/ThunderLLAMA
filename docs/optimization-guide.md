# ThunderLLAMA 性能优化指南

**适用模型**: Qwen3-30B-A3B-128K-Q5_K_M
**硬件**: Apple M4 Pro

## 当前配置

```bash
THUNDER_LMCACHE=1 ./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 -ngl 99 --port 30000
```

## 优化清单

### ✅ 已启用的优化

| 优化 | 配置 | 效果 |
|------|------|------|
| **GPU Offload** | `-ngl 99` | 所有层卸载到 GPU |
| **LMCache** | `THUNDER_LMCACHE=1` | 内容哈希缓存，跳过重复计算 |
| **KV Cache 量化** | API 动态配置 | Q4 节省 75% KV 内存 |

### 🚀 推荐启用的优化

#### 1. **Flash Attention** ⭐⭐⭐

**效果**: 加速注意力计算，降低内存占用

```bash
-fa on  # 或 --flash-attn on
```

**预期提升**: 10-30% 吞吐量提升（取决于序列长度）

#### 2. **Paged Attention** ⭐⭐⭐

**效果**: 分页管理 KV cache，减少内存碎片

```bash
export LLAMA_PAGED_ATTENTION=1
```

**预期提升**:
- 更高的并发能力
- 更灵活的内存管理
- 减少 OOM 风险

#### 3. **Chunk Prefill** ⭐⭐

**效果**: 分块处理长 prompt，平滑首 token 延迟

```bash
export THUNDERLLAMA_CHUNK_PREFILL=1
```

**预期提升**:
- 长 prompt (> 1000 tokens) 首 token 延迟降低
- 更平滑的用户体验
- 不影响总吞吐量

#### 4. **并发槽位** ⭐⭐

**效果**: 支持多个并发请求

```bash
--parallel 4  # 支持 4 个并发请求
```

**预期提升**:
- 多用户场景吞吐量提升
- 单用户场景无明显影响

**注意**: 每个槽位占用独立的 KV cache 内存

#### 5. **Batch Size 优化** ⭐

**效果**: 批处理大小影响吞吐量和延迟

```bash
-b 2048    # 逻辑 batch size (默认值)
-ub 512    # 物理 batch size (默认值)
```

**调优建议**:
- 长文本生成：增大 batch size (如 4096)
- 低延迟场景：减小 batch size (如 512)

#### 6. **前缀匹配优化** ⭐

**效果**: LMCache 前缀匹配加速

```bash
export THUNDER_PREFIX_MATCHING=1
```

**预期提升**:
- RAG 场景性能提升
- 多轮对话缓存复用率提高

#### 7. **磁盘缓存** ⭐

**效果**: LMCache 溢出到磁盘，扩大缓存容量

```bash
export THUNDER_LMCACHE_DISK_PATH=/path/to/cache
```

**适用场景**:
- 内存受限
- 需要缓存大量历史对话

## 推荐配置组合

### 配置 1: 高性能 (推荐) ⭐⭐⭐

**适合**: 内存充足 (32+ GB)，追求最高吞吐量

```bash
export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1
export THUNDER_PREFIX_MATCHING=1

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 4 \
  -b 4096 \
  --port 30000
```

**KV Cache 策略** (通过 API):
```json
{"name": "fixed", "params": {"level": "q4_0"}}
```

**预期效果**:
- 吞吐量: 55-60 tok/s
- 并发: 4 个槽位
- 内存: ~24 GB (模型 + 4×Q4 KV cache)

### 配置 2: 内存优化 (16 GB RAM)

**适合**: 内存受限，单用户场景

```bash
export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 1 \
  -b 2048 \
  --port 30000
```

**KV Cache 策略**:
```json
{"name": "fixed", "params": {"level": "q4_0"}}
```

**预期效果**:
- 吞吐量: 54-56 tok/s
- 并发: 1 个槽位
- 内存: ~22.2 GB

### 配置 3: 低延迟

**适合**: 交互式对话，注重首 token 速度

```bash
export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 2 \
  -b 512 \
  -ub 256 \
  --port 30000
```

**KV Cache 策略**:
```json
{"name": "adaptive", "params": {...}}
```

**预期效果**:
- 首 token 延迟: 低
- 吞吐量: 50-55 tok/s
- 并发: 2 个槽位

## 优化效果对比

| 配置 | 吞吐量 | 内存 | 并发 | 适用场景 |
|------|--------|------|------|----------|
| **基础** (当前) | 54 tok/s | 22 GB | 1 | 测试 |
| **高性能** | 55-60 tok/s | 24 GB | 4 | 生产环境 |
| **内存优化** | 54-56 tok/s | 22 GB | 1 | 16 GB RAM |
| **低延迟** | 50-55 tok/s | 23 GB | 2 | 交互对话 |

## 测试验证

### 测试 Flash Attention

```bash
# 启用 Flash Attention
curl -s http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain quantum computing:","max_tokens":200}' \
  | jq '.timings.predicted_per_second'
```

### 测试并发

```bash
# 并发发送 4 个请求
for i in {1..4}; do
  curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Test concurrent request:","max_tokens":50}' &
done
wait
```

### 监控内存

```bash
# 查看进程内存
ps -p $(lsof -ti:30000) -o pid,rss,vsz | \
  awk 'NR==2 {printf "RSS: %.2f GB\n", $2/1048576}'
```

## 故障排查

### Flash Attention 不生效

**症状**: 加了 `-fa on` 但性能无变化

**原因**: 模型可能不支持或硬件不支持

**解决**: 查看日志确认是否启用

### Paged Attention OOM

**症状**: 启用 paged attention 后内存溢出

**原因**: 并发槽位过多

**解决**: 减少 `--parallel` 数量或降低 context size

### LMCache 命中率低

**症状**: `skip_rate` 接近 0

**原因**: 请求模式不重复

**解决**:
- RAG 场景启用 `THUNDER_PREFIX_MATCHING=1`
- 检查是否有重复的 system prompt

## 最终推荐配置

**对于 M4 Pro + 30B Q5 模型，推荐使用『高性能配置』**：

```bash
#!/bin/bash
export LLAMA_PAGED_ATTENTION=1
export THUNDER_LMCACHE=1
export THUNDERLLAMA_CHUNK_PREFILL=1
export THUNDER_PREFIX_MATCHING=1

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 4 \
  -b 4096 \
  --port 30000
```

**启动后配置 KV Cache**:

```bash
curl -s http://localhost:30000/thunder/kv-strategy \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"fixed","params":{"level":"q4_0"},"version":1}'
```

**预期效果**:
- ✅ 吞吐量 55-60 tok/s
- ✅ 内存占用 ~24 GB
- ✅ 支持 4 并发
- ✅ LMCache 加速重复内容

---

*优化指南版本: 1.0*
*测试日期: 2026-03-13*
*硬件: Apple M4 Pro*
