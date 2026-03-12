#!/bin/bash
# Paged Attention 功能测试脚本
# 测试修复后的 Paged Attention 是否正常输出

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
LLAMA_SERVER="./build/bin/llama-server"
MODEL_PATH="${1:-}"  # 从命令行参数获取模型路径
PORT=18090

# 检查 llama-server 是否存在
if [ ! -f "$LLAMA_SERVER" ]; then
    echo -e "${RED}错误: llama-server 不存在: $LLAMA_SERVER${NC}"
    exit 1
fi

# 检查模型路径
if [ -z "$MODEL_PATH" ]; then
    echo -e "${YELLOW}用法: $0 <模型路径>${NC}"
    echo ""
    echo "示例:"
    echo "  $0 ~/models/Qwen3-0.6B.gguf"
    echo "  $0 ~/models/Qwen3-1.7B-Q4_K_M.gguf"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型文件不存在: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Paged Attention 功能测试${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "模型: $MODEL_PATH"
echo "端口: $PORT"
echo ""

# 停止旧的 llama-server 进程
echo -e "${YELLOW}停止旧的 llama-server 进程...${NC}"
pkill -f "llama-server.*$PORT" 2>/dev/null || true
sleep 2

# 启动 llama-server (启用 Paged Attention + Flash Attention)
echo -e "${YELLOW}启动 llama-server (Paged Attention + Flash Attention)...${NC}"
LLAMA_PAGED_ATTENTION=1 $LLAMA_SERVER \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -fa auto \
    -np 1 \
    -c 8192 \
    --port $PORT \
    > /tmp/llama-server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务器启动
echo -e "${YELLOW}等待服务器启动...${NC}"
MAX_WAIT=30
WAITED=0
while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo -e "${RED}错误: 服务器启动超时${NC}"
        cat /tmp/llama-server.log
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done

echo -e "${GREEN}服务器启动成功！${NC}"
echo ""

# 测试 1: 简单补全测试
echo -e "${YELLOW}测试 1: 简单补全测试${NC}"
echo "Prompt: 'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is'"

RESPONSE=$(curl -s http://localhost:$PORT/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
        "n_predict": 50,
        "temperature": 0.0,
        "cache_prompt": true
    }')

echo "Response:"
echo "$RESPONSE" | jq -r '.content' 2>/dev/null || echo "$RESPONSE"
echo ""

# 检查输出是否包含乱码
if echo "$RESPONSE" | jq -r '.content' | grep -qE '[^\x00-\x7F]|[�]'; then
    echo -e "${RED}❌ 测试失败: 输出包含乱码！${NC}"
    FAILED=true
else
    echo -e "${GREEN}✅ 测试通过: 输出正常${NC}"
    FAILED=false
fi

echo ""
echo -e "${YELLOW}测试 2: 长上下文测试 (KV cache reuse)${NC}"

# 第一次请求 (填充 KV cache)
echo "Request 1: 建立 KV cache..."
curl -s http://localhost:$PORT/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Once upon a time, in a land far far away, there lived a wise old wizard named",
        "n_predict": 30,
        "temperature": 0.0,
        "cache_prompt": true
    }' > /tmp/response1.json

echo "Response 1:"
jq -r '.content' /tmp/response1.json 2>/dev/null || cat /tmp/response1.json
echo ""

# 第二次请求 (复用 KV cache - 这是触发 Paged Attention bug 的关键场景)
echo "Request 2: 复用 KV cache (测试 Paged Attention)..."
RESPONSE2=$(curl -s http://localhost:$PORT/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Once upon a time, in a land far far away, there lived a wise old wizard named",
        "n_predict": 30,
        "temperature": 0.0,
        "cache_prompt": true
    }')

echo "Response 2:"
echo "$RESPONSE2" | jq -r '.content' 2>/dev/null || echo "$RESPONSE2"
echo ""

# 验证两次输出一致性 (temperature=0.0 应该输出一致)
CONTENT1=$(jq -r '.content' /tmp/response1.json 2>/dev/null)
CONTENT2=$(echo "$RESPONSE2" | jq -r '.content' 2>/dev/null)

if [ "$CONTENT1" = "$CONTENT2" ]; then
    echo -e "${GREEN}✅ 测试通过: KV cache 复用正确，输出一致${NC}"
else
    echo -e "${RED}❌ 测试失败: KV cache 复用异常，输出不一致${NC}"
    echo "Expected: $CONTENT1"
    echo "Got:      $CONTENT2"
    FAILED=true
fi

echo ""
echo -e "${YELLOW}测试 3: 多轮对话测试 (Continuous Batching)${NC}"

# 发送多个并发请求测试 Paged Attention 的块分配
for i in {1..3}; do
    echo "Request $i..."
    curl -s http://localhost:$PORT/completion \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"Count from 1 to 10: \",
            \"n_predict\": 20,
            \"temperature\": 0.0
        }" > /tmp/response_multi_$i.json &
done

wait

echo "Responses:"
for i in {1..3}; do
    echo "Request $i:"
    jq -r '.content' /tmp/response_multi_$i.json 2>/dev/null || cat /tmp/response_multi_$i.json

    # 检查乱码
    if jq -r '.content' /tmp/response_multi_$i.json | grep -qE '[^\x00-\x7F]|[�]'; then
        echo -e "${RED}❌ Request $i 包含乱码！${NC}"
        FAILED=true
    fi
done

echo ""
echo -e "${GREEN}===========================================${NC}"
if [ "$FAILED" = true ]; then
    echo -e "${RED}测试结果: 失败 ❌${NC}"
    echo ""
    echo "查看详细日志:"
    echo "  tail -100 /tmp/llama-server.log"
else
    echo -e "${GREEN}测试结果: 通过 ✅${NC}"
    echo ""
    echo "Paged Attention 修复验证成功！"
fi
echo -e "${GREEN}===========================================${NC}"

# 清理
echo ""
echo -e "${YELLOW}停止服务器...${NC}"
kill $SERVER_PID 2>/dev/null || true
sleep 1

exit 0
