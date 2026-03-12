#!/bin/bash
# ============================================================================
# ThunderLLAMA 配置加载器
#
# 从全局配置文件 ~/.openclaw/config.yaml 生成启动参数
# ============================================================================

set -e

CONFIG_FILE="${OPENCLAW_CONFIG:-$HOME/.openclaw/config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ============================================================================
# YAML 解析器（使用 Python）
# ============================================================================

parse_yaml() {
    python3 -c "
import yaml
import sys
import os

with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)

# 展开 ~ 路径
def expand_path(path):
    if path and isinstance(path, str):
        return os.path.expanduser(path)
    return path

# 模型配置
model = config['model']
server = config['server']
lmcache = config['lmcache']
perf = config['performance']
env = config['environment']

# 生成启动参数
args = []

# 必需参数
args.append(f\"--model {expand_path(model['path'])}\")
args.append(f\"--host {server['host']}\")
args.append(f\"--port {server['port']}\")

# GPU 配置
if model['gpu_layers'] > 0:
    args.append(f\"--n-gpu-layers {model['gpu_layers']}\")
if model['main_gpu'] is not None:
    args.append(f\"--main-gpu {model['main_gpu']}\")
if model['tensor_split']:
    args.append(f\"--tensor-split {','.join(map(str, model['tensor_split']))}\")

# 上下文配置
args.append(f\"--ctx-size {model['context_size']}\")
args.append(f\"--batch-size {model['batch_size']}\")
if model.get('ubatch_size'):
    args.append(f\"--ubatch-size {model['ubatch_size']}\")

# 线程配置
args.append(f\"--threads {model['threads']}\")
args.append(f\"--threads-batch {model['threads_batch']}\")

# 并行配置
args.append(f\"--parallel {server['n_parallel']}\")

# Flash Attention
if model.get('flash_attention'):
    args.append('--flash-attn')

# RoPE 配置
if model.get('rope_freq_base'):
    args.append(f\"--rope-freq-base {model['rope_freq_base']}\")
if model.get('rope_freq_scale'):
    args.append(f\"--rope-freq-scale {model['rope_freq_scale']}\")

# KV Cache 优化
if perf.get('kv_cache_type'):
    args.append(f\"--cache-type-k {perf['kv_cache_type']}\")
    args.append(f\"--cache-type-v {perf['kv_cache_type']}\")

if perf.get('no_kv_offload'):
    args.append('--no-kv-offload')

# Prompt 缓存
if perf.get('prompt_cache_all'):
    args.append('--cache-prompt')

if perf.get('prompt_cache_ro'):
    args.append('--cache-prompt-ro')

# 内存优化
if perf.get('mlock'):
    args.append('--mlock')

if perf.get('numa'):
    args.append('--numa isolate')

# Metal 配置
if perf.get('metal_enabled') and not perf.get('metal_embed_library', True):
    args.append('--no-mmap')

# 日志
if server.get('verbose'):
    args.append('--verbose')

# 输出启动参数
print(' '.join(args))

# 输出环境变量
print('---ENV---')

# LMCache 环境变量
if lmcache['enabled']:
    print(f\"THUNDER_LMCACHE=1\")
    print(f\"THUNDER_LMCACHE_DISK_PATH={expand_path(lmcache['disk_path'])}\")
    # 注意：L2/L3 size 在 C++ 代码中硬编码，这里只是记录
else:
    print(f\"THUNDER_LMCACHE=0\")

# OpenMP 配置
if env.get('kmp_duplicate_lib_ok'):
    print('KMP_DUPLICATE_LIB_OK=TRUE')

# HuggingFace 缓存
if env.get('huggingface_hub_cache'):
    print(f\"HUGGINGFACE_HUB_CACHE={expand_path(env['huggingface_hub_cache'])}\")
if env.get('transformers_cache'):
    print(f\"TRANSFORMERS_CACHE={expand_path(env['transformers_cache'])}\")
" 2>/dev/null || echo "Error: Python3 or PyYAML not found"
}

# ============================================================================
# 生成启动命令
# ============================================================================

OUTPUT=$(parse_yaml)

if [ $? -ne 0 ]; then
    echo "Error parsing config file"
    exit 1
fi

# 分离参数和环境变量
ARGS=$(echo "$OUTPUT" | sed -n '1,/---ENV---/p' | grep -v -- "---ENV---")
ENV_VARS=$(echo "$OUTPUT" | sed -n '/---ENV---/,$p' | grep -v -- "---ENV---")

# 导出环境变量
while IFS= read -r line; do
    if [ -n "$line" ]; then
        export "$line"
    fi
done <<< "$ENV_VARS"

# ============================================================================
# 主函数
# ============================================================================

case "${1:-help}" in
    args)
        # 仅输出启动参数
        echo "$ARGS"
        ;;

    env)
        # 仅输出环境变量
        echo "$ENV_VARS"
        ;;

    start)
        # 启动服务器
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        BIN_DIR="$SCRIPT_DIR/../build/bin"

        if [ ! -f "$BIN_DIR/llama-server" ]; then
            echo "Error: llama-server not found in $BIN_DIR"
            exit 1
        fi

        # 创建日志目录
        LOG_DIR=$(python3 -c "import yaml, os; print(os.path.expanduser(yaml.safe_load(open('$CONFIG_FILE'))['server']['log_dir']))")
        mkdir -p "$LOG_DIR"

        echo "Starting ThunderLLAMA with config: $CONFIG_FILE"
        echo ""
        echo "Command: $BIN_DIR/llama-server $ARGS"
        echo ""
        echo "Environment:"
        echo "$ENV_VARS"
        echo ""

        # 启动服务器
        cd "$BIN_DIR"
        eval "./llama-server $ARGS > $LOG_DIR/llama-server.log 2>&1 &"
        SERVER_PID=$!
        echo $SERVER_PID > "$LOG_DIR/llama-server.pid"

        echo "Server started (PID: $SERVER_PID)"
        echo "Log: $LOG_DIR/llama-server.log"

        # 等待服务器就绪
        HOST=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['server']['host'])")
        PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['server']['port'])")

        echo "Waiting for server to be ready..."
        for i in {1..60}; do
            if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
                echo "✓ Server ready!"
                exit 0
            fi
            sleep 1
        done

        echo "⚠️ Server may not be ready yet. Check logs: $LOG_DIR/llama-server.log"
        ;;

    stop)
        # 停止服务器
        LOG_DIR=$(python3 -c "import yaml, os; print(os.path.expanduser(yaml.safe_load(open('$CONFIG_FILE'))['server']['log_dir']))")
        PID_FILE="$LOG_DIR/llama-server.pid"

        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            echo "Stopping server (PID: $PID)..."
            kill "$PID" 2>/dev/null
            sleep 2
            rm -f "$PID_FILE"
            echo "✓ Server stopped"
        else
            echo "No server PID file found"
        fi
        ;;

    *)
        cat <<EOF
ThunderLLAMA Config Loader

Usage: $0 <command>

Commands:
  args    - Print startup arguments
  env     - Print environment variables
  start   - Start ThunderLLAMA server
  stop    - Stop ThunderLLAMA server

Config file: $CONFIG_FILE

Example:
  $0 start              # Start server with config
  $0 args               # Show what args would be used
  $0 env                # Show what env vars would be set

To use custom config:
  OPENCLAW_CONFIG=/path/to/config.yaml $0 start
EOF
        ;;
esac
