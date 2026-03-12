# 统一配置系统

> **所有配置集中在 `config.sh`，一处修改，全局生效**

## 🎯 设计理念

**问题**: 之前配置分散在：
- 环境变量 (`THUNDER_LMCACHE`, `LMCACHE_ENABLED`, `THUNDER_LMCACHE_DISK_PATH`, ...)
- 命令行参数 (`--n-gpu-layers`, `--ctx-size`, ...)
- 脚本硬编码 (模型路径, 端口号, ...)

**解决**: 所有配置集中到 `config.sh`，使用简单的 `run.sh` 脚本操作

## 📁 文件结构

```
clawgate-integration/
├── config.sh           # 统一配置文件 (修改这个!)
├── run.sh              # 一键启动/测试脚本
└── UNIFIED_CONFIG.md   # 本文档
```

## 🚀 快速开始

### 1. 查看配置

```bash
cd /Users/lisihao/ThunderLLAMA/clawgate-integration
./run.sh config
```

输出：
```
========================================================================
  ThunderLLAMA + LMCache Configuration
========================================================================

Model:
  Path:        /Users/lisihao/models/qwen3-30b-a3b-gguf/...
  GPU Layers:  20
  Context:     4096
  Batch:       256

Server:
  URL:         http://127.0.0.1:30000

LMCache:
  Enabled:     YES
  Disk Path:   /Users/lisihao/.openclaw/lmcache_bench.bin
  L2 (Memory): 8GB
  L3 (Disk):   256GB

ContextPilot:
  Server Mode: Embedded
  GPU:         false

========================================================================
```

### 2. 启动服务器

```bash
./run.sh start
```

### 3. 运行性能测试

```bash
./run.sh bench
```

### 4. 一键测试（启动+测试）

```bash
./run.sh test
```

### 5. 停止服务器

```bash
./run.sh stop
```

## ⚙️ 修改配置

**只需编辑 `config.sh` 一个文件！**

### 示例 1: 换模型

```bash
# 编辑 config.sh
vim config.sh

# 修改这一行：
export MODEL_PATH="/Users/lisihao/models/qwen3.5-35b-a3b-gguf/Qwen3.5-35B-A3B-Q4_K_M.gguf"

# 重启服务器
./run.sh restart
```

### 示例 2: 调整 GPU 层数

```bash
# 编辑 config.sh
export MODEL_GPU_LAYERS=30  # 改成 30 层

# 重启
./run.sh restart
```

### 示例 3: 禁用 LMCache

```bash
# 编辑 config.sh
export THUNDER_LMCACHE=0    # 0 = 禁用

# 重启
./run.sh restart
```

### 示例 4: 更换缓存路径

```bash
# 编辑 config.sh
export THUNDER_LMCACHE_DISK_PATH="/Volumes/ExternalSSD/lmcache.bin"

# 重启
./run.sh restart
```

## 📋 完整配置选项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| **模型配置** | | |
| `MODEL_PATH` | 模型文件路径 | qwen3-30b |
| `MODEL_GPU_LAYERS` | GPU 层数 | 20 |
| `MODEL_CTX_SIZE` | 上下文大小 | 4096 |
| `MODEL_BATCH_SIZE` | 批处理大小 | 256 |
| `MODEL_THREADS` | CPU 线程数 | 8 |
| **服务器配置** | | |
| `THUNDERLLAMA_HOST` | 监听地址 | 127.0.0.1 |
| `THUNDERLLAMA_PORT` | 监听端口 | 30000 |
| **LMCache 配置** | | |
| `THUNDER_LMCACHE` | 启用开关 (1/0) | 1 |
| `THUNDER_LMCACHE_DISK_PATH` | 缓存文件路径 | ~/.openclaw/lmcache_bench.bin |
| `THUNDER_LMCACHE_L2_SIZE` | L2 内存大小 | 8GB |
| `THUNDER_LMCACHE_L3_SIZE` | L3 磁盘大小 | 256GB |
| **ContextPilot 配置** | | |
| `CONTEXTPILOT_URL` | 服务器 URL (空=嵌入) | "" |
| `CONTEXTPILOT_GPU` | 是否使用 GPU | false |
| `EVICTION_SYNC_ENABLED` | 启用 eviction sync | true |

## 🛠️ run.sh 命令参考

| 命令 | 说明 |
|------|------|
| `./run.sh config` | 显示当前配置 |
| `./run.sh start` | 启动服务器 |
| `./run.sh stop` | 停止服务器 |
| `./run.sh restart` | 重启服务器 |
| `./run.sh bench` | 运行性能测试 |
| `./run.sh test` | 启动并测试 |
| `./run.sh logs` | 查看实时日志 |
| `./run.sh status` | 查看服务器状态 |

## 🔍 故障排查

### 服务器启动失败

```bash
# 查看日志
./run.sh logs

# 或
tail -100 ~/.openclaw/logs/llama-server.log
```

### LMCache 未启用

```bash
# 检查配置
./run.sh config

# 确认 LMCache 是 YES
# 如果是 NO，编辑 config.sh：
export THUNDER_LMCACHE=1
```

### 端口占用

```bash
# 修改端口
vim config.sh
# 改成：
export THUNDERLLAMA_PORT=30001

./run.sh restart
```

## 📖 工作流程示例

### 对比 LMCache 性能

```bash
# 1. 禁用 LMCache 测试
vim config.sh  # 设置 THUNDER_LMCACHE=0
./run.sh test  # 记录结果

# 2. 启用 LMCache 测试
vim config.sh  # 设置 THUNDER_LMCACHE=1
./run.sh restart
./run.sh bench  # 记录结果

# 3. 对比两次结果
```

### 测试不同模型

```bash
# 测试 30B 模型
vim config.sh  # MODEL_PATH=...qwen3-30b...
./run.sh test

# 测试 35B 模型
vim config.sh  # MODEL_PATH=...qwen3.5-35b...
./run.sh restart
./run.sh bench
```

## ✨ 优势

1. **集中管理**: 所有配置在一个文件，不会忘记或遗漏
2. **简单操作**: 一键启动/停止/测试，无需记忆复杂命令
3. **避免错误**: 不会因为环境变量名打错而失效 (之前 `LMCACHE_ENABLED` vs `THUNDER_LMCACHE`)
4. **易于调试**: 配置错误一目了然
5. **可复现**: 配置文件可以共享，保证环境一致

## 🎉 总结

**以后只需要：**

1. 修改配置 → `vim config.sh`
2. 启动服务器 → `./run.sh start`
3. 运行测试 → `./run.sh bench`

**不再需要记忆：**
- ❌ 长长的命令行参数
- ❌ 各种环境变量名
- ❌ 不同脚本的不同配置

**一切集中在 config.sh！**
