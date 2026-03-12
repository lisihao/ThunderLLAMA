#!/bin/bash
# ThunderLLAMA 安全卸载使用示例

MODEL="/Users/lisihao/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf"
PROMPT="Explain quantum computing in simple terms."

echo "========================================="
echo "  ThunderLLAMA 安全卸载示例"
echo "========================================="
echo ""

# 1. 配置环境
echo ">>> 步骤 1: 配置缓存存储"
source setup_cache_env.sh
export THUNDER_LMCACHE=1
echo ""

# 2. 启动程序（后台运行）
echo ">>> 步骤 2: 启动 ThunderLLAMA (后台运行)"
build/bin/llama-simple -m "$MODEL" -n 100 "$PROMPT" > /tmp/thunder_example.log 2>&1 &
PID=$!
echo "进程 PID: $PID"
echo "查看完整日志: tail -f /tmp/thunder_example.log"
sleep 2

# 3. 查看信号处理器注册信息
echo ""
echo ">>> 步骤 3: 查看信号处理器"
grep "Signal handler registered" /tmp/thunder_example.log
echo ""

# 4. 演示安全卸载
echo ">>> 步骤 4: 发送安全卸载信号"
echo "命令: kill -USR1 $PID"
echo "按 Enter 键发送信号，或 Ctrl+C 取消..."
read

kill -USR1 $PID
sleep 2

echo ""
echo ">>> 步骤 5: 查看卸载结果"
grep -A 5 "unmount" /tmp/thunder_example.log | tail -10
echo ""

echo "========================================="
echo "  现在可以安全拔出 U盘/硬盘"
echo "  程序继续运行（使用内存缓存）"
echo "========================================="
echo ""
echo "等待进程结束..."
wait $PID 2>/dev/null

echo "示例完成！"
