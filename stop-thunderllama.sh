#!/bin/bash
# ThunderLLAMA 停止脚本
# 版本: 1.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.thunderllama.pid"
CONFIG_FILE="$SCRIPT_DIR/thunderllama.conf"

# 读取配置（获取端口和日志路径）
if [ -f "$CONFIG_FILE" ]; then
    source <(grep -E '^(SERVER_PORT|LOG_FILE)=' "$CONFIG_FILE")
fi

echo "=== 停止 ThunderLLAMA 服务器 ==="

# 方式 1: 从 PID 文件停止
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 找到服务器进程 (PID: $PID)"
        kill $PID
        sleep 2

        # 检查是否成功停止
        if ps -p $PID > /dev/null 2>&1; then
            echo "⚠️  进程未响应 SIGTERM，强制停止..."
            kill -9 $PID
        fi

        rm -f "$PID_FILE"
        echo "✅ 服务器已停止"
        exit 0
    else
        echo "⚠️  PID 文件存在但进程不在运行"
        rm -f "$PID_FILE"
    fi
fi

# 方式 2: 从端口查找进程
if [ -n "$SERVER_PORT" ]; then
    if lsof -ti:$SERVER_PORT > /dev/null 2>&1; then
        PID=$(lsof -ti:$SERVER_PORT)
        echo "✅ 通过端口 $SERVER_PORT 找到进程 (PID: $PID)"
        kill $PID
        sleep 2

        if lsof -ti:$SERVER_PORT > /dev/null 2>&1; then
            echo "⚠️  进程未响应 SIGTERM，强制停止..."
            kill -9 $(lsof -ti:$SERVER_PORT)
        fi

        echo "✅ 服务器已停止"
        exit 0
    fi
fi

# 方式 3: 查找所有 llama-server 进程
PIDS=$(pgrep -f "llama-server")
if [ -n "$PIDS" ]; then
    echo "⚠️  找到 llama-server 进程: $PIDS"
    read -p "是否停止所有 llama-server 进程？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $PIDS
        sleep 2
        echo "✅ 已停止所有 llama-server 进程"
        exit 0
    fi
fi

echo "❌ 未找到运行中的 ThunderLLAMA 服务器"
exit 1
