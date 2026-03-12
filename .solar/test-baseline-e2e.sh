#!/bin/bash
# Baseline 端到端性能测试
# 测试: TTFT (首token时延) + tok/s (生成速度)

set -e

MODEL="${MODEL:-$HOME/models/Qwen3-1.7B-Q4_K_M.gguf}"
SERVER="./build/bin/llama-server"
PORT="8090"
LOG_FILE="/tmp/thunderllama-baseline-server.log"
RESULT_FILE="/tmp/thunderllama-baseline-e2e-result.json"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 ThunderLLAMA Baseline 端到端性能测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model: $(basename $MODEL)"
echo "Port: $PORT"
echo ""

# 检查模型
if [ ! -f "$MODEL" ]; then
    echo "❌ 模型不存在: $MODEL"
    exit 1
fi

# 检查可执行文件
if [ ! -f "$SERVER" ]; then
    echo "❌ llama-server 不存在"
    exit 1
fi

# 停止已有进程
echo "🧹 清理旧进程..."
pkill -f "llama-server.*$PORT" || true
sleep 2

# 启动 baseline 服务器
echo "🚀 启动 Baseline Server..."
echo ""
echo "配置:"
echo "  - GPU Layers: 99 (全GPU)"
echo "  - Context: 8192"
echo "  - Threads: 10"
echo "  - Flash Attention: auto"
echo "  - Paged Attention: ❌ 禁用"
echo "  - KV Cache: f16 (无量化)"
echo "  - Continuous Batching: ❌ 禁用"
echo "  - Slots: 1 (默认)"
echo ""

$SERVER \
    -m "$MODEL" \
    -ngl 99 \
    -c 8192 \
    -t 10 \
    --port "$PORT" \
    --host 0.0.0.0 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务器启动
echo "⏳ 等待服务器启动..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ 服务器就绪"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# 检查是否启动成功
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "❌ 服务器启动失败"
    cat "$LOG_FILE"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 运行性能测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试 1: 短提示，测量 TTFT
echo "Test 1: 短提示 (512 tokens prompt, 128 tokens generation)"
echo "-----------------------------------------------"

START_TIME=$(date +%s.%N)

RESPONSE=$(curl -s -w "\n%{time_total}" http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "'"$(python3 -c 'print("What is the capital of France? " * 50)')"'"}
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": false
    }')

TOTAL_TIME=$(echo "$RESPONSE" | tail -1)
RESPONSE_BODY=$(echo "$RESPONSE" | head -n -1)

# 提取指标
COMPLETION_TOKENS=$(echo "$RESPONSE_BODY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['usage']['completion_tokens'])")
PROMPT_TOKENS=$(echo "$RESPONSE_BODY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['usage']['prompt_tokens'])")

# 计算 tok/s (生成速度)
TOK_PER_SEC=$(python3 -c "print(f'{$COMPLETION_TOKENS / $TOTAL_TIME:.2f}')")

# 估算 TTFT (假设生成是均匀的)
# TTFT ≈ total_time - generation_time
# generation_time ≈ completion_tokens / tok_per_sec
GENERATION_TIME=$(python3 -c "print($COMPLETION_TOKENS / $TOK_PER_SEC)")
TTFT=$(python3 -c "print(f'{$TOTAL_TIME - $GENERATION_TIME:.3f}')")

echo ""
echo "结果:"
echo "  Prompt Tokens:     $PROMPT_TOKENS"
echo "  Completion Tokens: $COMPLETION_TOKENS"
echo "  Total Time:        ${TOTAL_TIME}s"
echo "  ⏱️  TTFT:            ${TTFT}s"
echo "  🚀 Tok/s:           $TOK_PER_SEC tokens/s"
echo ""

# 测试 2: 长提示，测量生成速度
echo "Test 2: 长提示 (2048 tokens prompt, 256 tokens generation)"
echo "-----------------------------------------------"

START_TIME=$(date +%s.%N)

RESPONSE=$(curl -s -w "\n%{time_total}" http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "'"$(python3 -c 'print("Explain quantum computing in detail. " * 200)')"'"}
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": false
    }')

TOTAL_TIME=$(echo "$RESPONSE" | tail -1)
RESPONSE_BODY=$(echo "$RESPONSE" | head -n -1)

# 提取指标
COMPLETION_TOKENS=$(echo "$RESPONSE_BODY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['usage']['completion_tokens'])")
PROMPT_TOKENS=$(echo "$RESPONSE_BODY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['usage']['prompt_tokens'])")

# 计算 tok/s
TOK_PER_SEC=$(python3 -c "print(f'{$COMPLETION_TOKENS / $TOTAL_TIME:.2f}')")

# 估算 TTFT
GENERATION_TIME=$(python3 -c "print($COMPLETION_TOKENS / $TOK_PER_SEC)")
TTFT=$(python3 -c "print(f'{$TOTAL_TIME - $GENERATION_TIME:.3f}')")

echo ""
echo "结果:"
echo "  Prompt Tokens:     $PROMPT_TOKENS"
echo "  Completion Tokens: $COMPLETION_TOKENS"
echo "  Total Time:        ${TOTAL_TIME}s"
echo "  ⏱️  TTFT:            ${TTFT}s"
echo "  🚀 Tok/s:           $TOK_PER_SEC tokens/s"
echo ""

# 保存结果
cat > "$RESULT_FILE" <<EOF
{
    "config": "baseline",
    "model": "$(basename $MODEL)",
    "test_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "optimization": {
        "paged_attention": false,
        "flash_attention": "auto",
        "kv_cache_type": "f16",
        "continuous_batching": false,
        "slots": 1
    },
    "test_1_short_prompt": {
        "prompt_tokens": $PROMPT_TOKENS,
        "completion_tokens": $COMPLETION_TOKENS,
        "total_time_sec": $TOTAL_TIME,
        "ttft_sec": $TTFT,
        "tok_per_sec": $TOK_PER_SEC
    },
    "test_2_long_prompt": {
        "prompt_tokens": $PROMPT_TOKENS,
        "completion_tokens": $COMPLETION_TOKENS,
        "total_time_sec": $TOTAL_TIME,
        "ttft_sec": $TTFT,
        "tok_per_sec": $TOK_PER_SEC
    }
}
EOF

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Baseline 性能汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat "$RESULT_FILE" | python3 -m json.tool
echo ""
echo "✅ 结果已保存: $RESULT_FILE"
echo ""

# 停止服务器
echo "🛑 停止服务器..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Baseline 测试完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
