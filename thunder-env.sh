#!/bin/bash
# ThunderLLAMA 环境变量配置
# 使用方法: source thunder-env.sh

# LMCache 存储路径（toshiba 盘，745GB 可用空间）
export THUNDER_LMCACHE_DISK_PATH="/Volumes/toshiba/thunderllama-cache/kv_cache.bin"

# 启用 LMCache
export THUNDER_LMCACHE=1

# 前缀匹配模式（默认启用）
# 1 = 允许部分匹配（prefix matching，配合 ContextPilot 使用）
# 0 = 仅完全匹配（exact matching，向后兼容模式）
export THUNDER_PREFIX_MATCHING=1

# 启用 Paged Attention（经测试验证无乱码，收益：-40%内存，8K→128K上下文，+50%并发）
export LLAMA_PAGED_ATTENTION=1

echo "✅ ThunderLLAMA 环境变量已设置:"
echo "   THUNDER_LMCACHE_DISK_PATH=$THUNDER_LMCACHE_DISK_PATH"
echo "   THUNDER_LMCACHE=$THUNDER_LMCACHE"
echo "   THUNDER_PREFIX_MATCHING=$THUNDER_PREFIX_MATCHING"
echo "   LLAMA_PAGED_ATTENTION=$LLAMA_PAGED_ATTENTION"
