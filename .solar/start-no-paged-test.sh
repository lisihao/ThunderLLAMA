#!/bin/bash
# 测试关闭 Paged Attention

cd /Users/lisihao/ThunderLLAMA

MODEL_PATH=$(find ~/models -name "Qwen3-30B-A3B-128K-Q5_K_M.gguf" 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "❌ 模型文件未找到"
    exit 1
fi

echo "✅ 模型文件: $MODEL_PATH"
echo ""
echo "🔥 测试关闭 Paged Attention"
echo "  - Paged Attention: ❌ 禁用"
echo "  - Flash Attention: auto"
echo "  - Slots: 1"
echo ""

# 不设置 LLAMA_PAGED_ATTENTION 环境变量

./build/bin/llama-server \
  -m "$MODEL_PATH" \
  -ngl 99 \
  -fa auto \
  -np 1 \
  -c 8192 \
  --host 0.0.0.0 \
  --port 8090 \
  > /tmp/no-paged-test-30b.log 2>&1 &

SERVER_PID=$!
echo "🚀 Server PID: $SERVER_PID"
echo "📝 Logs: /tmp/no-paged-test-30b.log"
echo ""
echo "等待服务器启动..."

for i in {1..30}; do
    if curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo "✅ 服务器启动成功!"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "❌ 服务器启动超时"
exit 1
