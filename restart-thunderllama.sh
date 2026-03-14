#!/bin/bash
# ThunderLLAMA 重启脚本
# 版本: 1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 重启 ThunderLLAMA 服务器 ==="

# 停止
if [ -f "$SCRIPT_DIR/stop-thunderllama.sh" ]; then
    "$SCRIPT_DIR/stop-thunderllama.sh"
else
    echo "❌ 找不到 stop-thunderllama.sh"
    exit 1
fi

# 等待停止完成
sleep 2

# 启动
if [ -f "$SCRIPT_DIR/start-thunderllama.sh" ]; then
    "$SCRIPT_DIR/start-thunderllama.sh"
else
    echo "❌ 找不到 start-thunderllama.sh"
    exit 1
fi
