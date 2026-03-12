# ThunderLLAMA Cache Management Tool

命令行工具，用于管理 ThunderLLAMA KV 缓存。

## 功能

- **stats**: 显示缓存统计信息
- **clear**: 清空缓存文件
- **compact**: 触发手动压缩（自动进行）
- **info**: 显示详细缓存信息

## 使用方法

```bash
# 显示统计信息（默认路径）
thunder-cache stats

# 显示统计信息（自定义路径）
thunder-cache stats /Volumes/toshiba/thunderllama/kv_cache.bin

# 清空缓存
thunder-cache clear

# 压缩缓存
thunder-cache compact

# 帮助信息
thunder-cache help
```

## 缓存路径

默认路径：`~/.cache/thunderllama/kv_cache.bin`

可通过以下方式指定：
1. 命令行参数：`thunder-cache stats /path/to/cache.bin`
2. 环境变量：`export THUNDER_LMCACHE_DISK_PATH=/path/to/cache.bin`

## 统计信息

```
=========================================
  CACHE STATISTICS
=========================================
Path: /Users/xxx/.cache/thunderllama/kv_cache.bin
Size: 156.7 MB
L2 (Memory) Usage: 8192.0 MB
L3 (Disk) Usage: 15678.9 MB
Hit Rate: 87.5%
L3 Enabled: Yes
```

## 构建

工具会随 ThunderLLAMA 主项目一起构建：

```bash
cd ThunderLLAMA/build
cmake ..
make thunder-cache
```
