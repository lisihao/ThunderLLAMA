#!/bin/bash
# ============================================================================
# ThunderLLAMA 一键启动/测试脚本
#
# 使用统一配置文件，简化操作
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载配置
source "$SCRIPT_DIR/config.sh"

# ============================================================================
# 命令处理
# ============================================================================

case "${1:-help}" in
    config|show)
        show_config
        ;;

    start)
        show_config
        echo ""
        start_server
        ;;

    stop)
        stop_server
        ;;

    restart)
        stop_server
        sleep 2
        start_server
        ;;

    bench|benchmark)
        show_config
        echo ""
        echo "Running benchmark..."
        run_benchmark
        ;;

    test)
        show_config
        echo ""
        echo "Starting server and running benchmark..."
        start_server
        sleep 5
        run_benchmark
        ;;

    logs)
        tail -f "$LOG_DIR/llama-server.log"
        ;;

    status)
        if [ -f "$LOG_DIR/llama-server.pid" ]; then
            PID=$(cat "$LOG_DIR/llama-server.pid")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "✓ Server running (PID: $PID)"
                curl -s "$THUNDERLLAMA_URL/health" | jq . || echo "Server not responding"
            else
                echo "❌ Server not running (stale PID file)"
            fi
        else
            echo "❌ Server not running"
        fi
        ;;

    help|*)
        cat <<EOF
ThunderLLAMA + LMCache 统一管理脚本

Usage: $0 <command>

Commands:
  config   - 显示当前配置
  start    - 启动服务器
  stop     - 停止服务器
  restart  - 重启服务器
  bench    - 运行性能测试（假设服务器已启动）
  test     - 启动服务器并运行性能测试
  logs     - 查看服务器日志
  status   - 查看服务器状态
  help     - 显示此帮助

配置文件: $SCRIPT_DIR/config.sh

示例:
  $0 config                # 查看配置
  $0 start                 # 启动服务器
  $0 test                  # 启动服务器并测试
  $0 stop                  # 停止服务器

修改配置:
  编辑 config.sh 文件，修改任何配置项
EOF
        ;;
esac
