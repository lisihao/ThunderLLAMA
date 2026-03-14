#!/bin/bash
# ThunderLLAMA 状态检查脚本
# 版本: 1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.thunderllama.pid"
CONFIG_FILE="$SCRIPT_DIR/thunderllama.conf"

# 读取配置
if [ -f "$CONFIG_FILE" ]; then
    source <(grep -E '^(SERVER_PORT|LOG_FILE|MODEL_TYPE)=' "$CONFIG_FILE")
fi

echo "==========================================="
echo "ThunderLLAMA 服务器状态"
echo "==========================================="
echo ""

# 检查进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 服务器运行中"
        echo "   PID:    $PID"
        echo "   模型:   ${MODEL_TYPE:-未知}"
        echo "   端口:   ${SERVER_PORT:-未知}"
        echo "   日志:   ${LOG_FILE:-未知}"

        # CPU 和内存使用
        if command -v ps &> /dev/null; then
            CPU=$(ps -p $PID -o %cpu= | xargs)
            MEM=$(ps -p $PID -o rss= | awk '{print $1/1024/1024 " GB"}')
            echo "   CPU:    ${CPU}%"
            echo "   内存:   ${MEM}"
        fi

        echo ""

        # 检查服务器健康状态
        if [ -n "$SERVER_PORT" ]; then
            if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
                echo "✅ 服务器健康检查通过"
                echo ""

                # LMCache 统计
                echo "=== LMCache 统计 ==="
                STATS=$(curl -s http://localhost:$SERVER_PORT/lmcache/stats 2>/dev/null)
                if [ $? -eq 0 ]; then
                    echo "$STATS" | jq '{
                        total_prefills,
                        skip_count,
                        skip_rate,
                        total_skip_rate,
                        l2_hit_rate,
                        l2_chunks,
                        l3_chunks,
                        l3_usage_bytes
                    }'
                else
                    echo "⚠️  无法获取 LMCache 统计"
                fi
            else
                echo "⚠️  服务器未响应健康检查"
            fi
        fi
    else
        echo "❌ 服务器未运行 (PID 文件存在但进程不在)"
        rm -f "$PID_FILE"
    fi
else
    # 尝试通过端口查找
    if [ -n "$SERVER_PORT" ] && lsof -ti:$SERVER_PORT > /dev/null 2>&1; then
        PID=$(lsof -ti:$SERVER_PORT)
        echo "⚠️  服务器运行中但 PID 文件缺失"
        echo "   PID:  $PID"
        echo "   端口: $SERVER_PORT"
    else
        echo "❌ 服务器未运行"
    fi
fi

echo ""
echo "==========================================="
echo "命令快捷方式:"
echo "  启动: ./start-thunderllama.sh"
echo "  停止: ./stop-thunderllama.sh"
echo "  日志: tail -f ${LOG_FILE:-/tmp/llama-server-30b.log}"
echo "  统计: curl -s http://localhost:${SERVER_PORT:-30000}/lmcache/stats | jq ."
echo "==========================================="
