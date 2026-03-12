#!/bin/bash
# LMCache 性能对比测试
# 对比: Baseline (THUNDER_LMCACHE=0) vs LMCache (THUNDER_LMCACHE=1)

set -e

MODEL="/Users/lisihao/llama.cpp/models/tinyllama-1.1b-chat-q4_k_m.gguf"
BIN="./build/bin/llama-cli"
PROMPT="What is the capital of France? Please explain in detail."
TOKENS=50
REPEATS=3

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 LMCache 性能对比测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model: $(basename $MODEL)"
echo "Prompt: $PROMPT"
echo "Tokens: $TOKENS"
echo "Repeats: $REPEATS"
echo ""

# 检查模型
if [ ! -f "$MODEL" ]; then
    echo "❌ 模型不存在: $MODEL"
    exit 1
fi

# 检查可执行文件
if [ ! -f "$BIN" ]; then
    echo "❌ llama-cli 不存在，尝试编译..."
    cd /Users/lisihao/ThunderLLAMA
    cmake --build build --target llama-cli
fi

# 提取性能数据的函数
extract_eval_time() {
    grep "eval time" | sed -n 's/.*= *\([0-9.]*\) ms.*/\1/p' | tail -1
}

extract_tokens_per_sec() {
    grep "eval time" | sed -n 's/.*(\([0-9.]*\) T\/s).*/\1/p' | tail -1
}

# 测试 Baseline (THUNDER_LMCACHE=0)
echo "========================================"
echo "测试 1: Baseline (LMCache 禁用)"
echo "========================================"
echo ""

rm -f /tmp/lmcache_baseline_times.txt /tmp/lmcache_baseline_tps.txt

for i in $(seq 1 $REPEATS); do
    echo "Run $i/$REPEATS..."

    output=$($BIN \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n $TOKENS \
        --temp 0.0 \
        -ngl 99 \
        2>&1)

    eval_time=$(echo "$output" | extract_eval_time)
    tps=$(echo "$output" | extract_tokens_per_sec)

    echo "  eval time: $eval_time ms/token, tokens/s: $tps"
    echo "$eval_time" >> /tmp/lmcache_baseline_times.txt
    echo "$tps" >> /tmp/lmcache_baseline_tps.txt
done

# 计算 Baseline 平均值
baseline_avg_time=$(awk '{s+=$1} END {printf "%.2f", s/NR}' /tmp/lmcache_baseline_times.txt)
baseline_avg_tps=$(awk '{s+=$1} END {printf "%.2f", s/NR}' /tmp/lmcache_baseline_tps.txt)

echo ""
echo "Baseline 平均: $baseline_avg_time ms/token, $baseline_avg_tps tokens/s"
echo ""

# 测试 LMCache Enabled (THUNDER_LMCACHE=1)
echo "========================================"
echo "测试 2: LMCache Enabled"
echo "========================================"
echo ""

rm -f /tmp/lmcache_enabled_times.txt /tmp/lmcache_enabled_tps.txt

for i in $(seq 1 $REPEATS); do
    echo "Run $i/$REPEATS..."

    output=$(THUNDER_LMCACHE=1 $BIN \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n $TOKENS \
        --temp 0.0 \
        -ngl 99 \
        2>&1)

    eval_time=$(echo "$output" | extract_eval_time)
    tps=$(echo "$output" | extract_tokens_per_sec)

    echo "  eval time: $eval_time ms/token, tokens/s: $tps"
    echo "$eval_time" >> /tmp/lmcache_enabled_times.txt
    echo "$tps" >> /tmp/lmcache_enabled_tps.txt
done

# 计算 LMCache 平均值
lmcache_avg_time=$(awk '{s+=$1} END {printf "%.2f", s/NR}' /tmp/lmcache_enabled_times.txt)
lmcache_avg_tps=$(awk '{s+=$1} END {printf "%.2f", s/NR}' /tmp/lmcache_enabled_tps.txt)

echo ""
echo "LMCache 平均: $lmcache_avg_time ms/token, $lmcache_avg_tps tokens/s"
echo ""

# 计算性能差距
time_diff=$(echo "scale=2; (($baseline_avg_time - $lmcache_avg_time) / $baseline_avg_time) * 100" | bc)
tps_diff=$(echo "scale=2; (($lmcache_avg_tps - $baseline_avg_tps) / $baseline_avg_tps) * 100" | bc)

# 输出对比表格
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 性能对比结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-20s %-20s %-20s %-15s\n" "指标" "Baseline" "LMCache" "差距"
echo "────────────────────────────────────────────────────────────────────────"
printf "%-20s %-20s %-20s %-15s\n" "eval time (ms/tok)" "$baseline_avg_time" "$lmcache_avg_time" "${time_diff}%"
printf "%-20s %-20s %-20s %-15s\n" "tokens/s" "$baseline_avg_tps" "$lmcache_avg_tps" "${tps_diff}%"
echo ""

# 判断结果
if (( $(echo "$time_diff > 5" | bc -l) )); then
    echo "⚠️  警告: LMCache 开销超过 5%（约束违反）"
    echo "   预期: < 5%, 实际: ${time_diff}%"
elif (( $(echo "$time_diff < -5" | bc -l) )); then
    echo "⚠️  注意: LMCache 性能反而下降 ${time_diff}%"
    echo "   可能原因: K/V copy 逻辑未实现，只有框架开销"
else
    echo "✅ LMCache 框架开销符合预期 (< 5%)"
fi

echo ""
echo "详细日志:"
echo "  - Baseline 原始数据: /tmp/lmcache_baseline_*.txt"
echo "  - LMCache 原始数据: /tmp/lmcache_enabled_*.txt"
echo ""
