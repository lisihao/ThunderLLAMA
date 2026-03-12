#!/bin/bash
# 测试小模型 (0.6B, 1.7B) Paged Attention 是否正常

cd /Users/lisihao/ThunderLLAMA

test_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$2

    echo ""
    echo "=" ================================================================
    echo "测试 $MODEL_NAME + Paged Attention"
    echo "=" ================================================================

    # 停止旧服务器
    pkill -9 -f llama-server
    sleep 2

    # 启动服务器
    export LLAMA_PAGED_ATTENTION=1

    ./build/bin/llama-server \
      -m "$MODEL_PATH" \
      -ngl 99 \
      -fa auto \
      -np 1 \
      -c 8192 \
      --host 0.0.0.0 \
      --port 8090 \
      > /tmp/test-small-paged.log 2>&1 &

    SERVER_PID=$!
    echo "🚀 Server PID: $SERVER_PID"

    # 等待启动
    for i in {1..30}; do
        if curl -s http://localhost:8090/health > /dev/null 2>&1; then
            echo "✅ 服务器启动成功"
            break
        fi
        sleep 1
    done

    # 检查 Paged Attention 状态
    echo ""
    echo "检查 Paged Attention 状态:"
    grep -E "(use_paged_attention|NOT creating block_pool)" /tmp/test-small-paged.log | head -2

    # 测试输出
    echo ""
    echo "测试输出质量:"
    sleep 2

    RESPONSE=$(curl -s -X POST http://localhost:8090/completion \
      -H "Content-Type: application/json" \
      -d '{"prompt": "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is", "n_predict": 50, "temperature": 0.0}')

    CONTENT=$(echo "$RESPONSE" | python3 -c "import json, sys; print(json.load(sys.stdin)['content'][:150])" 2>/dev/null || echo "解析失败")

    echo "输出: $CONTENT"

    # 简单判断是否乱码
    if echo "$CONTENT" | grep -q "通用)衣"; then
        echo "❌ 检测到乱码!"
    elif [ -z "$CONTENT" ] || [ "$CONTENT" = "解析失败" ]; then
        echo "⚠️  输出异常"
    else
        echo "✅ 输出正常"
    fi

    sleep 2
}

# 测试 0.6B
test_model "/Users/lisihao/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q5_K_M.gguf" "Qwen3-0.6B"

# 测试 1.7B
test_model "/Users/lisihao/models/qwen3-1.7b-gguf/Qwen3-1.7B-Q8_0.gguf" "Qwen3-1.7B"

echo ""
echo "=" ================================================================
echo "测试完成"
echo "=" ================================================================
