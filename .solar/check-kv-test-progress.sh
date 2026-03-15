#!/bin/bash
# KV Cache 量化测试进度监控

LOG_FILE="/Users/lisihao/ThunderLLAMA/.solar/kv-quant-test.log"
RESULTS_DIR="/Users/lisihao/ThunderLLAMA/.solar/kv-quant-results"

echo "========================================"
echo "KV Cache 量化测试 - 进度监控"
echo "========================================"
echo ""

# 检查测试是否在运行
if pgrep -f "test-kv-cache-quantization.sh" > /dev/null; then
    echo "✅ 测试正在运行中..."
else
    echo "⏸️  测试未运行或已完成"
fi

echo ""
echo "📊 当前进度:"
echo "----------------------------------------"

# 统计已完成的运行次数
if [ -d "$RESULTS_DIR" ]; then
    f16_count=$(ls -1 "$RESULTS_DIR"/f16_run*.json 2>/dev/null | wc -l | tr -d ' ')
    q8_0_count=$(ls -1 "$RESULTS_DIR"/q8_0_run*.json 2>/dev/null | wc -l | tr -d ' ')
    q4_0_count=$(ls -1 "$RESULTS_DIR"/q4_0_run*.json 2>/dev/null | wc -l | tr -d ' ')
    total_count=$((f16_count + q8_0_count + q4_0_count))

    echo "  f16:   $f16_count / 5 完成"
    echo "  q8_0:  $q8_0_count / 5 完成"
    echo "  q4_0:  $q4_0_count / 5 完成"
    echo ""
    echo "  总进度: $total_count / 15 ($(echo "scale=1; $total_count * 100 / 15" | bc)%)"
else
    echo "  结果目录不存在，测试尚未开始"
fi

echo ""
echo "📝 最新日志 (最后 15 行):"
echo "----------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
else
    echo "  日志文件不存在"
fi

echo ""
echo "========================================"
echo "命令:"
echo "  查看完整日志: tail -f $LOG_FILE"
echo "  查看结果: ls -lh $RESULTS_DIR"
echo "========================================"
