#!/bin/bash
# 测试 35B 模型 Paged Attention 是否乱码

cd /Users/lisihao/ThunderLLAMA

MODEL_PATH="/Users/lisihao/models/qwen3.5-35b-a3b-gguf/Qwen3.5-35B-A3B-Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件未找到: $MODEL_PATH"
    exit 1
fi

echo "✅ 模型文件: $MODEL_PATH"
echo ""
echo "🔥 测试 35B 模型 + Paged Attention"
echo "  - 模型: Qwen3.5-35B-A3B-Q4_K_M"
echo "  - Paged Attention: ✅ 启用"
echo "  - Flash Attention: auto"
echo "  - Slots: 1"
echo ""

export LLAMA_PAGED_ATTENTION=1

./build/bin/llama-server \
  -m "$MODEL_PATH" \
  -ngl 99 \
  -fa auto \
  -np 1 \
  -c 8192 \
  --host 0.0.0.0 \
  --port 8090 \
  > /tmp/35b-paged-test.log 2>&1 &

SERVER_PID=$!
echo "🚀 Server PID: $SERVER_PID"
echo "📝 Logs: /tmp/35b-paged-test.log"
echo ""
echo "等待服务器启动..."

for i in {1..60}; do
    if curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo "✅ 服务器启动成功!"

        # 显示 Paged Attention 状态
        echo ""
        echo "检查 Paged Attention 状态:"
        sleep 2
        grep -E "(LLAMA_PAGED_ATTENTION|paged attention enabled)" /tmp/35b-paged-test.log | head -5

        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "❌ 服务器启动超时"
exit 1
