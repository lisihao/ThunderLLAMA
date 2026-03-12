#!/bin/bash
# 启动优化配置的 llama-server (Configuration A)

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
echo "🔥 启动 Configuration A (生产环境标配)"
echo "  - Paged Attention: ✅"
echo "  - Flash Attention: on"
echo "  - KV Cache: q8_0"
echo "  - Continuous Batching: ✅"
echo "  - Slots: 8"
echo "  - Cache RAM: 4096 MB"
echo "  - Prompt Reuse: auto"
echo ""

# 启动服务器 (Configuration A)
export LLAMA_PAGED_ATTENTION=1

./build/bin/llama-server \
  -m "$MODEL_PATH" \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -cram 4096 \
  -kvo \
  -np 8 \
  -cb \
  -b 2048 \
  -ub 512 \
  --prompt-reuse-mode auto \
  -sps 0.5 \
  --mmap \
  --mlock \
  --repack \
  -t 10 \
  -tb 10 \
  -c 8192 \
  --host 0.0.0.0 \
  --port 8090 \
  > /tmp/optimized-server-30b.log 2>&1 &

SERVER_PID=$!
echo "🚀 Server PID: $SERVER_PID"
echo "📝 Logs: /tmp/optimized-server-30b.log"
echo ""
echo "等待服务器启动..."

# 等待服务器启动
for i in {1..30}; do
    if curl -s http://localhost:8090/health > /dev/null 2>&1; then
        echo "✅ 服务器启动成功!"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "❌ 服务器启动超时，检查日志: /tmp/optimized-server-30b.log"
exit 1
