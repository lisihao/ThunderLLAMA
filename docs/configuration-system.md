# ThunderLLAMA 统一配置系统

**版本**: 1.0
**创建于**: 2026-03-13
**目的**: 防止优化参数在会话丢失/上下文压缩后遗失

---

## 核心理念

> **配置即代码**：所有 ThunderLLAMA 的启动参数、优化选项、环境变量都集中管理在一个配置文件中，作为唯一真相源 (Single Source of Truth)。

**问题**：之前每次启动 ThunderLLAMA 都需要手动设置大量参数，容易遗忘，导致优化特性未启用。

**解决**：创建统一配置系统，自动读取、验证、应用所有参数。

---

## 文件结构

```
/Users/lisihao/ThunderLLAMA/
├── thunderllama.conf          # 唯一配置文件（所有参数）
├── start-thunderllama.sh      # 启动脚本（读取 conf）
├── stop-thunderllama.sh       # 停止脚本
├── status-thunderllama.sh     # 状态检查脚本
└── .thunderllama.pid          # 运行时 PID 文件（自动生成）
```

---

## 快速开始

```bash
cd /Users/lisihao/ThunderLLAMA

# 启动服务器
./start-thunderllama.sh

# 检查状态
./status-thunderllama.sh

# 停止服务器
./stop-thunderllama.sh
```

---

## 配置文件：thunderllama.conf

**位置**: `/Users/lisihao/ThunderLLAMA/thunderllama.conf`

**作用**: 唯一真相源 (Single Source of Truth)，所有启动参数从此文件读取。

### 配置分区

| 分区 | 说明 | 关键参数 |
|------|------|----------|
| **模型配置** | 模型路径和类型 | MODEL_PATH, MODEL_TYPE, CONTEXT_SIZE |
| **ThunderLLAMA 核心优化** | 关键优化特性 | THUNDER_LMCACHE, THUNDER_PREFIX_MATCHING, LLAMA_PAGED_ATTENTION |
| **KV Cache 策略** | 量化策略 | KV_CACHE_STRATEGY, KV_CACHE_LEVEL |
| **服务器基础配置** | 端口、GPU | SERVER_PORT, GPU_LAYERS, PARALLEL_SLOTS |
| **性能优化参数** | 批处理、缓存 | FLASH_ATTENTION, BATCH_SIZE, CACHE_REUSE |
| **已验证无效的优化** | 避免重复尝试 | SPECULATIVE_DECODING（已禁用） |

### 性能基准数据

配置文件记录了当前配置的预期性能（基于 2026-03-13 实测）：

- **单次新请求**: 54-57 tok/s
- **重复请求**: 80-120 tok/s (LMCache)
- **RAG 场景**: 75-90 tok/s (Prefix Matching)
- **多轮对话**: 65-80 tok/s (LMCache + Prefix)
- **内存占用**: 22.2 GB (模型 21.7GB + KV Cache Q4 0.47GB)
- **并发能力**: 4 slots, +50% (Paged Attention)

---

## 启动脚本详解

### 执行流程

```
读取 thunderllama.conf
    ↓
预检查（模型、缓存、端口、内存）
    ↓
设置环境变量（THUNDER_*, LLAMA_*）
    ↓
构建启动命令（基于配置动态生成）
    ↓
启动服务器（后台运行）
    ↓
健康检查（HTTP /health）
    ↓
配置 KV Cache 策略（HTTP API）
    ↓
显示状态和监控命令
```

### 关键功能

1. **自动预分配缓存文件**:
   ```bash
   if [ ! -f "$THUNDER_LMCACHE_DISK_PATH" ]; then
       truncate -s 10G "$THUNDER_LMCACHE_DISK_PATH"
   fi
   ```

2. **端口冲突检测**:
   ```bash
   if lsof -ti:$SERVER_PORT > /dev/null 2>&1; then
       echo "⚠️  端口已占用，是否停止现有服务？"
   fi
   ```

3. **内存检查**（macOS）:
   ```bash
   FREE_MEM_MB=$(vm_stat | grep "Pages free" | awk ...)
   if [ "$FREE_MEM_MB" -lt 10000 ]; then
       echo "⚠️  可用内存不足 10GB"
   fi
   ```

4. **动态命令构建**:
   ```bash
   CMD="./build/bin/llama-server"
   CMD="$CMD -m $MODEL_PATH"
   CMD="$CMD -c $CONTEXT_SIZE"
   # ... 基于配置文件动态添加参数
   ```

5. **KV Cache 策略配置**（通过 HTTP API）:
   ```bash
   curl -X POST http://localhost:$SERVER_PORT/thunder/kv-strategy \
     -d '{"name":"fixed","params":{"level":"q4_0"},"version":1}'
   ```

---

## 使用场景

### 场景 1: 日常启动（推荐）

```bash
cd /Users/lisihao/ThunderLLAMA
./start-thunderllama.sh
```

### 场景 2: 修改配置后重启

```bash
# 1. 编辑配置
vim thunderllama.conf

# 2. 重启服务器
./stop-thunderllama.sh && ./start-thunderllama.sh
```

### 场景 3: 检查运行状态

```bash
./status-thunderllama.sh
```

输出示例：
```
✅ 服务器运行中
   PID:    12345
   模型:   30B-Q5
   端口:   30000
   CPU:    92.3%
   内存:   22.4 GB

=== LMCache 统计 ===
{
  "total_prefills": 15,
  "skip_count": 3,
  "skip_rate": 0.2,
  "l2_chunks": 256,
  "l3_chunks": 512
}
```

### 场景 4: 查看实时日志

```bash
tail -f /tmp/llama-server-30b.log
```

### 场景 5: 获取 LMCache 统计

```bash
curl -s http://localhost:30000/lmcache/stats | jq .
```

---

## 配置修改指南

### 临时修改（单次生效）

不修改 `thunderllama.conf`，直接手动启动：

```bash
THUNDER_LMCACHE=1 ./build/bin/llama-server \
  -m ~/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q5_K_M.gguf \
  -c 2048 -ngl 99 --port 30000
```

### 永久修改（所有后续启动）

编辑 `thunderllama.conf`：

```bash
vim thunderllama.conf
```

**常见修改**：

1. **切换模型**：
   ```bash
   MODEL_PATH="$HOME/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q5_K_M.gguf"
   MODEL_TYPE="0.6B-Q5"
   ```

2. **调整 context size**：
   ```bash
   CONTEXT_SIZE=8192  # 从 4096 增大到 8192
   ```

3. **增大缓存**：
   ```bash
   CACHE_RAM=32768  # 从 16GB 增大到 32GB
   ```

4. **切换 KV Cache 量化**：
   ```bash
   KV_CACHE_LEVEL="q8_0"  # f16 | q8_0 | q4_0
   ```

5. **启用实验性功能**：
   ```bash
   NO_HOST=1  # 绕过 host buffer（可能不稳定）
   ```

修改后重启：
```bash
./stop-thunderllama.sh && ./start-thunderllama.sh
```

---

## 故障排查

### 问题 1: 启动失败

**症状**: `start-thunderllama.sh` 报错退出

**检查步骤**:
1. 查看日志：`tail -100 /tmp/llama-server-30b.log`
2. 检查模型路径：`ls ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf`
3. 检查端口占用：`lsof -i:30000`
4. 检查可用内存：`vm_stat`

**常见原因**:
- 模型文件不存在或路径错误
- 端口被其他进程占用
- 内存不足（需要 > 22GB）
- GPU 不可用（Metal 未启用）

### 问题 2: 健康检查超时

**症状**: "❌ 服务器健康检查超时"

**检查步骤**:
1. 检查进程是否运行：`ps aux | grep llama-server`
2. 检查端口监听：`lsof -i:30000`
3. 手动健康检查：`curl http://localhost:30000/health`
4. 查看日志错误：`tail -100 /tmp/llama-server-30b.log | grep -i error`

**可能原因**:
- 模型加载时间过长（30B 模型需要 30-60 秒）
- 端口配置错误
- 防火墙阻止本地连接

### 问题 3: KV Cache 策略配置失败

**症状**: "⚠️ KV Cache 策略配置可能失败"

**检查当前策略**:
```bash
curl -s http://localhost:30000/thunder/kv-strategy | jq .
```

**手动设置**:
```bash
curl -s http://localhost:30000/thunder/kv-strategy \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"fixed","params":{"level":"q4_0"},"version":1}' | jq .
```

### 问题 4: LMCache skip_count 始终为 0

**原因**:
- 提示词 < 256 tokens（无法形成完整 chunk）
- THUNDER_PREFIX_MATCHING 未启用
- 没有发送重复请求

**验证**:
```bash
# 发送两次相同 prompt（> 256 tokens）
curl -X POST http://localhost:30000/v1/completions \
  -d @long_prompt.txt  # 确保 prompt > 256 tokens

# 检查 skip_count
curl -s http://localhost:30000/lmcache/stats | jq '.skip_count'
```

### 问题 5: 性能未达到预期

**检查环境变量**:
```bash
# 查看启动日志中的环境变量
grep THUNDER /tmp/llama-server-30b.log
grep LLAMA_PAGED_ATTENTION /tmp/llama-server-30b.log
```

**验证优化是否启用**:
```bash
curl -s http://localhost:30000/lmcache/stats | jq .
```

**预期输出**:
- `l2_chunks` > 0（L2 缓存启用）
- `l3_chunks` > 0（L3 磁盘缓存启用）
- `l2_hit_rate` 接近 1.0（缓存命中率）

---

## 已知限制

1. **配置修改需要重启**: 修改 `thunderllama.conf` 后必须停止并重启服务器。

2. **单实例假设**: 配置系统假设只运行一个实例。多实例需要手动管理不同配置文件。

3. **日志无轮转**: 日志文件会无限增长，需要定期手动清理：
   ```bash
   rm /tmp/llama-server-30b.log
   ```

4. **KV Cache 策略限制**: 仅支持 `fixed` 策略（固定量化级别）。`threshold` 和 `adaptive` 策略需要修改 C++ 代码。

5. **macOS 专用**: 内存检查代码使用 `vm_stat`（macOS 工具），在 Linux 上需要修改。

---

## 下一步改进

### 短期（可立即实现）

- [ ] 日志轮转配置（logrotate）
- [ ] 多实例支持（不同配置文件、端口、PID 文件）
- [ ] `restart-thunderllama.sh` 快捷脚本
- [ ] 配置验证命令（检查语法和参数有效性）
- [ ] 性能基准测试脚本（自动测试吞吐量）

### 中期（需要较多工作）

- [ ] 修改 C++ 代码让 `llama-server` 原生读取 `thunderllama.conf`
- [ ] 实现 `threshold` 和 `adaptive` KV Cache 策略
- [ ] 配置热重载（无需重启）
- [ ] Web 配置管理界面
- [ ] 监控仪表盘（Grafana）

### 长期（需要架构变更）

- [ ] 分布式配置管理（etcd/Consul）
- [ ] 自动性能调优（基于负载动态调整参数）
- [ ] 配置版本管理和回滚
- [ ] A/B 测试框架（对比不同配置的性能）
- [ ] Kubernetes Operator（云原生部署）

---

## 相关文档

- [ThunderLLAMA 深度研究报告](./thunderllama-deep-dive.md)
- [30B 模型现实优化方案](./realistic-optimization-30b.md)
- [吞吐量测试报告（30B）](../benchmarks/throughput-report-30b.md)

---

## 更新历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-03-13 | 1.0 | 初始版本，创建统一配置系统 |

---

*ThunderLLAMA 统一配置系统 v1.0*
*创建于: 2026-03-13*
*维护者: Solar*
