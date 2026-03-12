#!/bin/bash
# ============================================================================
# ThunderLLAMA + LMCache + Clawgate 统一配置文件
#
# 所有配置集中在这里，一处修改，全局生效
# ============================================================================

# ----------------------------------------------------------------------------
# 模型配置
# ----------------------------------------------------------------------------
export MODEL_PATH="/Users/lisihao/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf"
export MODEL_GPU_LAYERS=20          # GPU 层数（根据显存调整）
export MODEL_CTX_SIZE=4096          # 上下文大小
export MODEL_BATCH_SIZE=256         # 批处理大小
export MODEL_THREADS=8              # CPU 线程数

# ----------------------------------------------------------------------------
# 服务器配置
# ----------------------------------------------------------------------------
export THUNDERLLAMA_HOST="127.0.0.1"
export THUNDERLLAMA_PORT=30000
export THUNDERLLAMA_URL="http://${THUNDERLLAMA_HOST}:${THUNDERLLAMA_PORT}"

# ----------------------------------------------------------------------------
# LMCache 配置 (核心！)
# ----------------------------------------------------------------------------
export THUNDER_LMCACHE=1                                    # 启用 LMCache (1=启用, 0=禁用)
export THUNDER_LMCACHE_DISK_PATH="$HOME/.openclaw/lmcache_bench.bin"  # 缓存文件路径
export THUNDER_LMCACHE_L2_SIZE=$((8 * 1024 * 1024 * 1024)) # L2 内存缓存: 8GB
export THUNDER_LMCACHE_L3_SIZE=$((256 * 1024 * 1024 * 1024))  # L3 磁盘缓存: 256GB

# ----------------------------------------------------------------------------
# ContextPilot 配置
# ----------------------------------------------------------------------------
export CONTEXTPILOT_URL=""          # 空 = 嵌入模式，填 URL = 服务器模式
export CONTEXTPILOT_GPU=false       # ContextPilot 是否使用 GPU
export CONTEXTPILOT_INDEX_URL="http://localhost:8000/evict"  # Eviction sync webhook

# ----------------------------------------------------------------------------
# Clawgate 配置
# ----------------------------------------------------------------------------
export EVICTION_SYNC_ENABLED=true   # 启用 eviction 同步
export LMCACHE_ENABLED=true         # Clawgate 层面的 LMCache 开关

# ----------------------------------------------------------------------------
# OpenMP 配置（避免冲突）
# ----------------------------------------------------------------------------
export KMP_DUPLICATE_LIB_OK=TRUE

# ----------------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------------
export LOG_DIR="$HOME/.openclaw/logs"
export LOG_LEVEL="INFO"             # DEBUG, INFO, WARNING, ERROR

# ============================================================================
# 辅助函数
# ============================================================================

# 显示当前配置
show_config() {
    echo "========================================================================"
    echo "  ThunderLLAMA + LMCache Configuration"
    echo "========================================================================"
    echo ""
    echo "Model:"
    echo "  Path:        $MODEL_PATH"
    echo "  GPU Layers:  $MODEL_GPU_LAYERS"
    echo "  Context:     $MODEL_CTX_SIZE"
    echo "  Batch:       $MODEL_BATCH_SIZE"
    echo ""
    echo "Server:"
    echo "  URL:         $THUNDERLLAMA_URL"
    echo ""
    echo "LMCache:"
    echo "  Enabled:     $([ "$THUNDER_LMCACHE" = "1" ] && echo "YES" || echo "NO")"
    echo "  Disk Path:   $THUNDER_LMCACHE_DISK_PATH"
    echo "  L2 (Memory): $(numfmt --to=iec $THUNDER_LMCACHE_L2_SIZE 2>/dev/null || echo "8GB")"
    echo "  L3 (Disk):   $(numfmt --to=iec $THUNDER_LMCACHE_L3_SIZE 2>/dev/null || echo "256GB")"
    echo ""
    echo "ContextPilot:"
    echo "  Server Mode: $([ -z "$CONTEXTPILOT_URL" ] && echo "Embedded" || echo "$CONTEXTPILOT_URL")"
    echo "  GPU:         $CONTEXTPILOT_GPU"
    echo ""
    echo "========================================================================"
}

# 启动服务器
start_server() {
    echo "Starting ThunderLLAMA server..."

    # 创建日志目录
    mkdir -p "$LOG_DIR"

    # 启动服务器
    cd "$(dirname "$0")/../build" || exit 1

    ./bin/llama-server \
        --model "$MODEL_PATH" \
        --port "$THUNDERLLAMA_PORT" \
        --host "$THUNDERLLAMA_HOST" \
        --n-gpu-layers "$MODEL_GPU_LAYERS" \
        --ctx-size "$MODEL_CTX_SIZE" \
        --batch-size "$MODEL_BATCH_SIZE" \
        --threads "$MODEL_THREADS" \
        > "$LOG_DIR/llama-server.log" 2>&1 &

    SERVER_PID=$!
    echo $SERVER_PID > "$LOG_DIR/llama-server.pid"

    echo "Server starting (PID: $SERVER_PID)"
    echo "Log: $LOG_DIR/llama-server.log"
    echo ""
    echo "Waiting for server to be ready..."

    for i in {1..60}; do
        if curl -s "http://$THUNDERLLAMA_HOST:$THUNDERLLAMA_PORT/health" > /dev/null 2>&1; then
            echo "✓ Server ready!"
            return 0
        fi
        sleep 1
    done

    echo "❌ Server failed to start"
    return 1
}

# 停止服务器
stop_server() {
    if [ -f "$LOG_DIR/llama-server.pid" ]; then
        PID=$(cat "$LOG_DIR/llama-server.pid")
        echo "Stopping server (PID: $PID)..."
        kill "$PID" 2>/dev/null
        sleep 2
        rm -f "$LOG_DIR/llama-server.pid"
        echo "✓ Server stopped"
    else
        echo "No server PID found"
    fi
}

# 运行性能测试
run_benchmark() {
    cd "$(dirname "$0")" || exit 1
    python3 benchmark_lmcache_proper.py
}

# ============================================================================
# 使用说明
# ============================================================================

# 如果直接运行此脚本，显示配置
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    show_config
    echo ""
    echo "Usage:"
    echo "  source config.sh           # 加载配置到当前 shell"
    echo "  source config.sh && start_server    # 加载配置并启动服务器"
    echo "  source config.sh && run_benchmark   # 加载配置并运行测试"
fi
