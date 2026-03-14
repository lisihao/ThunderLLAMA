# ThunderLLAMA 配置系统快速参考

> **目的**: 防止参数丢失，所有配置集中管理

---

## 🚀 快速开始

```bash
cd /Users/lisihao/ThunderLLAMA

# 启动
./start-thunderllama.sh

# 状态
./status-thunderllama.sh

# 停止
./stop-thunderllama.sh

# 重启
./restart-thunderllama.sh
```

---

## 📁 关键文件

| 文件 | 作用 | 是否可编辑 |
|------|------|-----------|
| `thunderllama.conf` | **唯一配置文件** | ✅ 是 |
| `start-thunderllama.sh` | 启动脚本 | ❌ 否（除非修改逻辑） |
| `stop-thunderllama.sh` | 停止脚本 | ❌ 否 |
| `restart-thunderllama.sh` | 重启脚本 | ❌ 否 |
| `status-thunderllama.sh` | 状态检查脚本 | ❌ 否 |
| `.thunderllama.pid` | 运行时 PID（自动生成） | ❌ 否 |

---

## ⚙️ 修改配置

**修改流程**：

```bash
# 1. 编辑配置文件
vim thunderllama.conf

# 2. 重启服务器
./restart-thunderllama.sh
```

**常见修改**：

| 想要... | 修改参数 | 示例值 |
|---------|----------|--------|
| 切换模型 | `MODEL_PATH`, `MODEL_TYPE` | `~/models/qwen3-0.6b-gguf/...` |
| 调整 context | `CONTEXT_SIZE` | `8192` |
| 增大缓存 | `CACHE_RAM` | `32768` (32GB) |
| 切换 KV 量化 | `KV_CACHE_LEVEL` | `f16` / `q8_0` / `q4_0` |
| 改端口 | `SERVER_PORT` | `30001` |

---

## 🔍 监控命令

```bash
# 实时日志
tail -f /tmp/llama-server-30b.log

# LMCache 统计
curl -s http://localhost:30000/lmcache/stats | jq .

# 健康检查
curl -s http://localhost:30000/health

# KV Cache 策略
curl -s http://localhost:30000/thunder/kv-strategy | jq .
```

---

## 📊 当前配置（2026-03-13）

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3-30B-A3B-128K-Q5_K_M (21.7 GB) |
| Context | 4096 tokens |
| KV Cache | Q4 量化 (0.47 GB, -72% 内存) |
| 端口 | 30000 |
| 并发槽位 | 4 |

**已启用优化**：
- ✅ LMCache (L2 内存 + L3 磁盘)
- ✅ Prefix Matching (RAG/多轮对话加速)
- ✅ Paged Attention (-40% 内存, +50% 并发)
- ✅ Flash Attention
- ✅ Continuous Batching

**已禁用（已验证无效）**：
- ❌ Speculative Decoding (Draft Model)
- ❌ N-gram Speculative

---

## 📈 预期性能（基于实测）

| 场景 | 吞吐量 |
|------|--------|
| 单次新请求 | 54-57 tok/s |
| 重复请求（LMCache 命中） | 80-120 tok/s |
| RAG（前缀匹配） | 75-90 tok/s |
| 多轮对话 | 65-80 tok/s |

---

## 🐛 常见问题

### Q: 启动失败？

```bash
# 检查日志
tail -100 /tmp/llama-server-30b.log

# 检查端口
lsof -i:30000

# 检查模型文件
ls ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf
```

### Q: skip_count 始终为 0？

**原因**：提示词 < 256 tokens

**解决**：使用 > 256 tokens 的提示词测试，例如：
```bash
curl -X POST http://localhost:30000/v1/completions \
  -d @long_prompt.txt  # 确保文件内容 > 256 tokens
```

### Q: 性能未达到预期？

**检查环境变量是否启用**：
```bash
grep "THUNDER_\|LLAMA_PAGED" /tmp/llama-server-30b.log
```

---

## 📚 完整文档

详见 `docs/configuration-system.md`

---

*快速参考 v1.0 - 2026-03-13*
