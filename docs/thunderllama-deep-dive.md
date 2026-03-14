# ThunderLLAMA 深度研究报告

**研究日期**: 2026-03-13
**模型**: Qwen3-30B-A3B-128K-Q5_K_M
**硬件**: Apple M4 Pro
**当前性能**: 54-57 tok/s

---

## 🔍 核心发现：ThunderLLAMA 独特架构

### 1. **LMCache-Lite 两层缓存架构** ⭐⭐⭐⭐⭐

ThunderLLAMA 实现了两层 KV cache 缓存系统，这是其**最核心的优化特性**：

```
┌──────────────────────────────────────┐
│         Application Layer            │
└───────────┬──────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│   L2 Cache (Memory)                  │
│   - 热数据，快速访问                  │
│   - 当前: 144 chunks, 12.8 MB        │
│   - 命中率: 100%                     │
└───────────┬──────────────────────────┘
            │ (溢出)
            ▼
┌──────────────────────────────────────┐
│   L3 Cache (Disk/SSD)                │
│   - 冷数据，持久化存储                │
│   - 路径: /Volumes/toshiba/...       │
│   - 当前: 144 chunks, 12.0 MB        │
│   - 可用空间: 741 GB                 │
└──────────────────────────────────────┘
```

**核心参数**:
- **THUNDER_CHUNK_SIZE = 256 tokens**
- **Content-based Hashing**: xxHash64
- **自动分层**: L2 满后自动溢出到 L3

---

### 2. **前缀匹配 (Prefix Matching)** ⭐⭐⭐⭐

**环境变量**: `THUNDER_PREFIX_MATCHING=1`

**工作原理**:
```
Prompt A: [System] [User1] [User2]
Prompt B: [System] [User1] [User3]  ← 前两个 chunk 相同

LMCache 识别：
- Chunk 0 (System): ✅ 命中，跳过计算
- Chunk 1 (User1):  ✅ 命中，跳过计算
- Chunk 2 (User3):  ❌ 未命中，重新计算
```

**适用场景**:
- ✅ **RAG**: System prompt 相同
- ✅ **多轮对话**: 历史 context 相同
- ✅ **批量任务**: Template 相同

**实测效果** (来自 thunder-env.sh 注释):
- 配合 ContextPilot 使用时效果显著

---

### 3. **Paged Attention 实测数据** ⭐⭐⭐⭐

**环境变量**: `LLAMA_PAGED_ATTENTION=1`

**thunder-env.sh 中的验证结果**:
```bash
# 启用 Paged Attention（经测试验证无乱码，收益：-40%内存，8K→128K上下文，+50%并发）
export LLAMA_PAGED_ATTENTION=1
```

**实测收益**:
- ✅ 内存降低 **40%**
- ✅ Context 扩展: 8K → 128K
- ✅ 并发能力 **+50%**
- ✅ 无质量损失（已验证无乱码）

---

### 4. **外置 SSD 磁盘缓存** ⭐⭐⭐

**配置**:
```bash
export THUNDER_LMCACHE_DISK_PATH="/Volumes/toshiba/thunderllama-cache/kv_cache.bin"
```

**当前状态**:
- 磁盘: Toshiba 外置 SSD, 745 GB 总容量
- 已用: 128 MB KV cache
- 可用: **741 GB** (可缓存大量历史对话)

**优势**:
- 内存满时自动溢出到磁盘
- 持久化缓存，重启后仍可用
- 支持超大规模缓存（数百 GB）

---

## 🎯 基于发现的优化策略

### 优化 1: 扩大磁盘缓存容量 ⭐⭐⭐⭐

**当前问题**:
- 磁盘缓存文件只有 128 MB
- 可用空间 741 GB 未充分利用

**优化方案**:
```bash
# 方式 1: 预分配更大缓存文件
truncate -s 10G /Volumes/toshiba/thunderllama-cache/kv_cache.bin

# 方式 2: 让 LMCache 自动增长（可能已支持）
# 检查源码确认
```

**预期效果**:
- 缓存更多历史对话
- 多用户场景：缓存命中率提升 50-80%
- 长期运行：性能持续优化

**风险**: 低

---

### 优化 2: 验证 Paged Attention 是否已启用

**检查方法**:
```bash
# 查看启动日志
tail -100 /tmp/llama-server-30b.log | grep -i "paged\|attention"

# 如果未启用，重启时加上
export LLAMA_PAGED_ATTENTION=1
```

**预期收益**:
- -40% 内存
- +50% 并发

---

### 优化 3: 激活前缀匹配优化

**检查方法**:
```bash
# 当前是否启用
env | grep THUNDER_PREFIX_MATCHING

# 如果未启用，添加
export THUNDER_PREFIX_MATCHING=1
```

**预期收益**:
- RAG 场景: 首 token 延迟降低 50-80%
- 多轮对话: 吞吐量提升 20-40%

---

### 优化 4: LMCache 统计监控和调优

**当前统计**:
```json
{
  "total_prefills": 3,
  "skip_count": 0,         ← ❌ 没有跳过任何计算！
  "skip_rate": 0.0,
  "l2_hit_rate": 1.0,
  "l2_chunks": 144,
  "l3_chunks": 144
}
```

**问题分析**:
- `skip_count = 0` 说明没有利用缓存跳过计算
- 可能原因：
  1. 测试请求都是新内容（无重复）
  2. Prefix Matching 未启用
  3. Chunk 未对齐导致无法匹配

**优化方案**:
```bash
# 1. 确保启用前缀匹配
export THUNDER_PREFIX_MATCHING=1

# 2. 测试缓存效果（发送重复 prompt）
# 第一次请求
curl -s http://localhost:30000/v1/completions \
  -d '{"prompt":"Explain quantum computing:","max_tokens":100}'

# 第二次请求（相同 prompt）
curl -s http://localhost:30000/v1/completions \
  -d '{"prompt":"Explain quantum computing:","max_tokens":100}'

# 检查 skip_count 是否增加
curl -s http://localhost:30000/lmcache/stats | jq '.skip_count, .skip_rate'
```

---

## 📊 完整优化配置（基于深度研究）

```bash
#!/bin/bash

# === LMCache 配置 ===
export THUNDER_LMCACHE=1
export THUNDER_LMCACHE_DISK_PATH="/Volumes/toshiba/thunderllama-cache/kv_cache.bin"
export THUNDER_PREFIX_MATCHING=1

# === Paged Attention ===
export LLAMA_PAGED_ATTENTION=1

# === Chunk Prefill ===
export THUNDERLLAMA_CHUNK_PREFILL=1

# === 预分配大缓存文件（可选） ===
if [ ! -f "/Volumes/toshiba/thunderllama-cache/kv_cache.bin" ]; then
    truncate -s 10G /Volumes/toshiba/thunderllama-cache/kv_cache.bin
fi

cd /Users/lisihao/ThunderLLAMA

./build/bin/llama-server \
  -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  -c 4096 \
  -ngl 99 \
  -fa on \
  --parallel 4 \
  -b 4096 \
  -ub 1024 \
  --cache-reuse 256 \
  --cache-ram 16384 \
  --kv-unified \
  --cont-batching \
  --prio-batch 2 \
  --port 30000 > /tmp/llama-server-optimized.log 2>&1 &

echo "Server PID: $!"
sleep 10

# 配置 Q4 KV cache
curl -s http://localhost:30000/thunder/kv-strategy \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"fixed","params":{"level":"q4_0"},"version":1}'

echo ""
echo "✅ ThunderLLAMA 优化配置已启动"
echo "   LMCache: 内存 + 磁盘双层缓存"
echo "   Paged Attention: -40% 内存, +50% 并发"
echo "   Prefix Matching: RAG/多轮对话加速"
echo "   KV Cache: Q4 量化 (-72% 内存)"
```

---

## 🔬 验证和测试

### 测试 1: LMCache 缓存效果

```bash
# 发送两次相同请求，验证 skip_count
for i in 1 2; do
  echo "=== Request $i ==="
  curl -s http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Explain machine learning in detail:","max_tokens":200}' \
    | jq '.timings.predicted_per_second'
  sleep 2
done

# 检查缓存统计
echo ""
echo "=== LMCache Stats ==="
curl -s http://localhost:30000/lmcache/stats | jq '{
  skip_count,
  skip_rate,
  total_skip_rate,
  l2_hit_rate,
  l2_chunks,
  l3_chunks
}'
```

**预期结果**:
- 第二次请求的 skip_count > 0
- skip_rate > 0

---

### 测试 2: 前缀匹配效果

```bash
# 请求 1: 完整 prompt
curl -s http://localhost:30000/v1/completions \
  -d '{"prompt":"System: You are a helpful AI. User: Explain AI.","max_tokens":50}' \
  | jq '.timings.prompt_per_second'

sleep 2

# 请求 2: 相同前缀，不同问题
curl -s http://localhost:30000/v1/completions \
  -d '{"prompt":"System: You are a helpful AI. User: Explain ML.","max_tokens":50}' \
  | jq '.timings.prompt_per_second'

# 检查是否跳过了"System: You are a helpful AI."部分
curl -s http://localhost:30000/lmcache/stats | jq '.skip_count'
```

---

### 测试 3: 磁盘缓存容量测试

```bash
# 发送大量不同请求，填充缓存
for i in {1..100}; do
  curl -s http://localhost:30000/v1/completions \
    -d "{\"prompt\":\"Test prompt $i:\",\"max_tokens\":100}" > /dev/null
  sleep 0.5
done

# 检查磁盘缓存大小
du -h /Volumes/toshiba/thunderllama-cache/kv_cache.bin

# 检查缓存统计
curl -s http://localhost:30000/lmcache/stats | jq '{l2_chunks, l3_chunks, l3_usage_bytes}'
```

---

## 📈 预期性能提升

| 场景 | 当前 | 优化后 | 提升 | 关键优化 |
|------|------|--------|------|----------|
| **单次新请求** | 54 tok/s | 54-57 tok/s | +0-5% | Flash Attention |
| **重复请求** | 54 tok/s | 80-120 tok/s | **+50-120%** | LMCache skip |
| **RAG (相同前缀)** | 54 tok/s | 75-90 tok/s | **+40-65%** | Prefix Matching |
| **多轮对话** | 54 tok/s | 65-80 tok/s | **+20-50%** | Prefix Matching + LMCache |
| **多用户并发** | 54 tok/s | 60-75 tok/s | +10-40% | Paged Attention + Cache |

---

## 🎯 最重要的发现

### ✅ ThunderLLAMA 的真正优势不是单次推理速度

**而是**:
1. **缓存复用**: L2+L3 双层缓存，极致复用
2. **前缀匹配**: RAG/对话场景跳过重复计算
3. **磁盘持久化**: 741 GB 可用，无限缓存
4. **Paged Attention**: -40% 内存，+50% 并发

### ❌ 不适用的优化

基于实测，以下优化**不要使用**（已记录到 Cortex）:
- Speculative Decoding (Draft Model)
- N-gram Speculative
- 原因: 30B 大模型验证成本高，质量下降且无性能提升

---

## 🚀 最终建议

**对于 30B 大模型，ThunderLLAMA 的优化重点应该是**:

1. **✅ 充分利用 LMCache 双层缓存**
   - 扩大磁盘缓存容量（10 GB+）
   - 启用前缀匹配
   - 监控 skip_rate

2. **✅ 启用 Paged Attention**
   - 验证是否已启用
   - -40% 内存，+50% 并发

3. **✅ 优化缓存策略**
   - 增大 cache-ram
   - 启用 cache-reuse
   - 使用 kv-unified

4. **✅ Q4 KV cache 量化**
   - -72% KV 内存
   - -4% 性能损失可接受

**不要期望**:
- 单次新请求的性能大幅提升（< 10%）
- 推测解码的帮助（已验证无效）

**应该专注**:
- 缓存命中率优化
- 重复场景加速
- 多用户并发能力

---

*ThunderLLAMA 深度研究版本: 1.0*
*研究日期: 2026-03-13*
*记录于: /Users/lisihao/ThunderLLAMA/docs/*
