# ThunderLLAMA 快速参考

## 启动

```bash
cd /Users/lisihao/ThunderLLAMA
source setup_cache_env.sh  # 自动配置 toshiba
export THUNDER_LMCACHE=1
build/bin/llama-simple -m model.gguf -n 10 "prompt"
```

## 安全拔出 U盘

```bash
# 1. 找到 PID（查看启动日志）
#    [ThunderLMCache] ✓ Signal handler registered: kill -USR1 <PID>

# 2. 发送信号
kill -USR1 <PID>

# 3. 等待确认
#    [ThunderChunkStorage] ✅ Safe unmount complete - external storage can be safely removed

# 4. 拔出 U盘
```

## 状态检查

```bash
# 查看日志
grep "ThunderChunkStorage" /path/to/log

# 正常：
[ThunderChunkStorage] ✓ L3 disk cache initialized successfully

# 降级：
[ThunderChunkStorage] ⚠️  L3 disk cache DISABLED - running in memory-only mode
```

## 故障处理

| 问题 | 原因 | 解决 |
|------|------|------|
| 磁盘不可用 | U盘损坏/未挂载 | 程序自动降级，修复磁盘后重启 |
| 信号不起作用 | PID 错误 | `ps aux \| grep llama-simple` 确认 PID |
| 性能下降 | L3 禁用 | 检查磁盘状态，重启程序 |

## 环境变量

```bash
# 自定义磁盘路径
export THUNDER_LMCACHE_DISK_PATH="/Volumes/toshiba/thunderllama/kv_cache.bin"

# 启用缓存
export THUNDER_LMCACHE=1
```

## 文件

- `setup_cache_env.sh` - 自动配置脚本
- `SAFE_UNMOUNT_GUIDE.md` - 完整使用指南
- `example_safe_unmount.sh` - 使用示例

---

**重要**：拔出 U盘前务必发送 `kill -USR1 <PID>` 信号！
