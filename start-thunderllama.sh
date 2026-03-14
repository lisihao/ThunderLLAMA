#!/bin/bash
# ThunderLLAMA 统一启动脚本
# 版本: 1.0
# 说明: 从 thunderllama.conf 读取配置并启动服务器

set -e  # 遇到错误立即退出

# ============================================================================
# 配置文件路径
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/thunderllama.conf"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "✅ 读取配置文件: $CONFIG_FILE"

# ============================================================================
# 加载配置
# ============================================================================
# 读取配置文件（忽略注释和空行）
source <(grep -v '^#' "$CONFIG_FILE" | grep -v '^$')

echo "✅ 配置加载完成"

# ============================================================================
# 预检查
# ============================================================================
echo ""
echo "=== 预检查 ==="

# 1. 检查模型文件
MODEL_PATH_EXPANDED="${MODEL_PATH/#\~/$HOME}"
if [ ! -f "$MODEL_PATH_EXPANDED" ]; then
    echo "❌ 模型文件不存在: $MODEL_PATH_EXPANDED"
    exit 1
fi
echo "✅ 模型文件: $MODEL_PATH_EXPANDED"

# 2. 检查磁盘缓存路径
if [ -n "$THUNDER_LMCACHE_DISK_PATH" ]; then
    CACHE_DIR=$(dirname "$THUNDER_LMCACHE_DISK_PATH")
    if [ ! -d "$CACHE_DIR" ]; then
        echo "⚠️  缓存目录不存在，创建: $CACHE_DIR"
        mkdir -p "$CACHE_DIR"
    fi

    # 预分配缓存文件（如果不存在）
    if [ ! -f "$THUNDER_LMCACHE_DISK_PATH" ]; then
        echo "⚠️  预分配 10GB 缓存文件..."
        truncate -s 10G "$THUNDER_LMCACHE_DISK_PATH" 2>/dev/null || \
            dd if=/dev/zero of="$THUNDER_LMCACHE_DISK_PATH" bs=1m count=10240 2>/dev/null
    fi
    echo "✅ 磁盘缓存: $THUNDER_LMCACHE_DISK_PATH ($(du -h "$THUNDER_LMCACHE_DISK_PATH" | cut -f1))"
fi

# 3. 检查端口是否被占用
if lsof -ti:$SERVER_PORT > /dev/null 2>&1; then
    echo "⚠️  端口 $SERVER_PORT 已被占用"
    read -p "是否停止现有服务？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "停止现有服务..."
        kill $(lsof -ti:$SERVER_PORT) || true
        sleep 2
    else
        echo "❌ 退出启动"
        exit 1
    fi
fi

# 4. 检查可用内存（macOS）
if command -v vm_stat &> /dev/null; then
    FREE_MEM_MB=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.' | awk '{print $1 * 4096 / 1024 / 1024}')
    echo "✅ 可用内存: ${FREE_MEM_MB} MB"

    if [ "${FREE_MEM_MB%%.*}" -lt 10000 ]; then
        echo "⚠️  可用内存不足 10GB，建议清理内存后再启动"
    fi
fi

# ============================================================================
# 设置环境变量
# ============================================================================
echo ""
echo "=== 设置环境变量 ==="

export THUNDER_LMCACHE="$THUNDER_LMCACHE"
export THUNDER_LMCACHE_DISK_PATH="$THUNDER_LMCACHE_DISK_PATH"
export THUNDER_PREFIX_MATCHING="$THUNDER_PREFIX_MATCHING"
export LLAMA_PAGED_ATTENTION="$LLAMA_PAGED_ATTENTION"
export THUNDERLLAMA_CHUNK_PREFILL="$THUNDERLLAMA_CHUNK_PREFILL"

if [ -n "$METAL_MAX_BUFFER_SIZE" ]; then
    export GGML_METAL_MAX_BUFFER_SIZE="$METAL_MAX_BUFFER_SIZE"
fi

echo "✅ THUNDER_LMCACHE=$THUNDER_LMCACHE"
echo "✅ THUNDER_LMCACHE_DISK_PATH=$THUNDER_LMCACHE_DISK_PATH"
echo "✅ THUNDER_PREFIX_MATCHING=$THUNDER_PREFIX_MATCHING"
echo "✅ LLAMA_PAGED_ATTENTION=$LLAMA_PAGED_ATTENTION"
echo "✅ THUNDERLLAMA_CHUNK_PREFILL=$THUNDERLLAMA_CHUNK_PREFILL"

# ============================================================================
# 构建启动命令
# ============================================================================
echo ""
echo "=== 构建启动命令 ==="

CMD="$SCRIPT_DIR/build/bin/llama-server"
CMD="$CMD -m $MODEL_PATH_EXPANDED"
CMD="$CMD -c $CONTEXT_SIZE"
CMD="$CMD -ngl $GPU_LAYERS"
CMD="$CMD --port $SERVER_PORT"

# Flash Attention
if [ "$FLASH_ATTENTION" != "auto" ]; then
    CMD="$CMD -fa $FLASH_ATTENTION"
fi

# Parallel
if [ -n "$PARALLEL_SLOTS" ] && [ "$PARALLEL_SLOTS" -gt 1 ]; then
    CMD="$CMD --parallel $PARALLEL_SLOTS"
fi

# Batch
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD -b $BATCH_SIZE"
fi
if [ -n "$UBATCH_SIZE" ]; then
    CMD="$CMD -ub $UBATCH_SIZE"
fi

# Cache
if [ -n "$CACHE_REUSE" ] && [ "$CACHE_REUSE" -gt 0 ]; then
    CMD="$CMD --cache-reuse $CACHE_REUSE"
fi
if [ -n "$CACHE_RAM" ]; then
    CMD="$CMD --cache-ram $CACHE_RAM"
fi
if [ "$KV_UNIFIED" = "1" ]; then
    CMD="$CMD --kv-unified"
fi

# Continuous Batching
if [ "$CONT_BATCHING" = "1" ]; then
    CMD="$CMD --cont-batching"
fi
if [ -n "$PRIO_BATCH" ] && [ "$PRIO_BATCH" -gt 0 ]; then
    CMD="$CMD --prio-batch $PRIO_BATCH"
fi

# CPU
if [ -n "$CPU_THREADS" ]; then
    CMD="$CMD --threads $CPU_THREADS"
fi
if [ -n "$CPU_THREADS_BATCH" ]; then
    CMD="$CMD --threads-batch $CPU_THREADS_BATCH"
fi
if [ -n "$CPU_MASK" ]; then
    CMD="$CMD --cpu-mask $CPU_MASK"
fi
if [ -n "$PROCESS_PRIORITY" ] && [ "$PROCESS_PRIORITY" -gt 0 ]; then
    CMD="$CMD --prio $PROCESS_PRIORITY"
fi

# Metal
if [ "$NO_HOST" = "1" ]; then
    CMD="$CMD --no-host"
fi

# Auto-fit
if [ "$AUTO_FIT" = "1" ]; then
    CMD="$CMD --fit on"
fi

# 日志
if [ -n "$LOG_FILE" ]; then
    CMD="$CMD > $LOG_FILE 2>&1 &"
else
    CMD="$CMD &"
fi

echo "启动命令:"
echo "$CMD"

# ============================================================================
# 启动服务器
# ============================================================================
echo ""
echo "=== 启动 ThunderLLAMA 服务器 ==="
eval $CMD
SERVER_PID=$!

echo "✅ 服务器已启动 (PID: $SERVER_PID)"
echo "   日志文件: $LOG_FILE"
echo "   端口: $SERVER_PORT"

# 等待服务器就绪
echo ""
echo "等待服务器就绪..."
sleep 10

# 检查服务器是否运行
if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "❌ 服务器启动失败！"
    echo "查看日志: tail -100 $LOG_FILE"
    exit 1
fi

# 健康检查
MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
        echo "✅ 服务器就绪！"
        break
    fi

    if [ $i -eq $MAX_RETRIES ]; then
        echo "❌ 服务器健康检查超时"
        echo "查看日志: tail -100 $LOG_FILE"
        exit 1
    fi

    sleep 1
done

# ============================================================================
# 配置 KV Cache 策略
# ============================================================================
if [ -n "$KV_CACHE_STRATEGY" ] && [ -n "$KV_CACHE_LEVEL" ]; then
    echo ""
    echo "=== 配置 KV Cache 策略 ==="

    STRATEGY_PAYLOAD="{\"name\":\"$KV_CACHE_STRATEGY\",\"params\":{\"level\":\"$KV_CACHE_LEVEL\"},\"version\":1}"

    RESULT=$(curl -s http://localhost:$SERVER_PORT/thunder/kv-strategy \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$STRATEGY_PAYLOAD")

    CURRENT_LEVEL=$(echo "$RESULT" | jq -r '.current_level')
    REBUILD_SUCCESS=$(echo "$RESULT" | jq -r '.rebuild_success')

    if [ "$REBUILD_SUCCESS" = "true" ] && [ "$CURRENT_LEVEL" = "$KV_CACHE_LEVEL" ]; then
        echo "✅ KV Cache 策略: $KV_CACHE_STRATEGY ($KV_CACHE_LEVEL)"
    else
        echo "⚠️  KV Cache 策略配置可能失败"
        echo "$RESULT" | jq .
    fi
fi

# ============================================================================
# 显示状态
# ============================================================================
echo ""
echo "=========================================="
echo "ThunderLLAMA 服务器启动成功！"
echo "=========================================="
echo ""
echo "模型:        $MODEL_TYPE"
echo "端口:        $SERVER_PORT"
echo "PID:         $SERVER_PID"
echo "日志:        $LOG_FILE"
echo ""
echo "优化特性:"
echo "  ✅ LMCache (L2+L3 双层缓存)"
echo "  ✅ Prefix Matching (前缀匹配)"
echo "  ✅ Paged Attention (-40% 内存)"
echo "  ✅ Flash Attention ($FLASH_ATTENTION)"
echo "  ✅ KV Cache $KV_CACHE_LEVEL 量化 (-72% 内存)"
echo ""
echo "监控命令:"
echo "  日志: tail -f $LOG_FILE"
echo "  统计: curl -s http://localhost:$SERVER_PORT/lmcache/stats | jq ."
echo "  健康: curl -s http://localhost:$SERVER_PORT/health"
echo ""
echo "停止命令:"
echo "  kill $SERVER_PID"
echo "=========================================="

# 保存 PID 到文件
echo "$SERVER_PID" > "$SCRIPT_DIR/.thunderllama.pid"
