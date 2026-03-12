#!/bin/bash
# 测试单 slot 配置（排查 context 问题）

cd /Users/lisihao/ThunderLLAMA

# 确保旧进程已停止
pkill -9 -f llama-server
sleep 2

# 查找模型文件
MODEL_PATH=$(find ~/models -name "Qwen3-30B-A3B-128K-Q5_K_M.gguf" 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "❌ 模型文件未找到: Qwen3-30B-A3B-128K-Q5_K_M.gguf"
    exit 1
fi

echo "✅ 模型文件: $MODEL_PATH"
echo ""
echo "🔥 启动单 slot 测试配置"
echo "  - Paged Attention: ✅"
echo "  - Flash Attention: on"
echo "  - Slots: 1 (单 slot)"
echo "  - Context: 8192"
echo ""

# 启动服务器 (单 slot)
export LLAMA_PAGED_ATTENTION=1

./build/bin/llama-server \
  -m "$MODEL_PATH" \
  -ngl 99 \
  -fa on \
  -np 1 \
  -c 8192 \
  --host 0.0.0.0 \
  --port 8090 \
  > /tmp/single-slot-test-30b.log 2>&1 &

SERVER_PID=$!
echo "🚀 Server PID: $SERVER_PID"
echo "📝 Logs: /tmp/single-slot-test-30b.log"
echo ""
echo "等待服务器启动..."

# 等待服务器启动
for i in {1..30}; do
    if curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo "✅ 服务器启动成功!"

        # 显示 slot context
        echo ""
        echo "检查 slot context:"
        sleep 2
        grep "new slot" /tmp/single-slot-test-30b.log | head -5

        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "❌ 服务器启动超时，检查日志: /tmp/single-slot-test-30b.log"
exit 1
