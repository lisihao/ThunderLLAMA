#!/bin/bash
# KV Cache 量化测试 - 持续监控
# 每 30 秒自动刷新进度

SCRIPT_DIR="/Users/lisihao/ThunderLLAMA/.solar"

echo "========================================"
echo "KV Cache 量化测试 - 持续监控"
echo "========================================"
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    clear

    # 显示当前时间
    echo "📅 更新时间: $(date '+%H:%M:%S')"
    echo ""

    # 运行进度检查
    "$SCRIPT_DIR/check-kv-test-progress.sh"

    # 检查测试是否完成
    if ! pgrep -f "test-kv-cache-quantization.sh" > /dev/null; then
        echo ""
        echo "🎉 测试已完成！"
        echo ""
        echo "正在生成最终报告..."
        python3 "$SCRIPT_DIR/postprocess-kv-results.py"
        echo ""
        echo "✅ 最终报告已生成"
        echo ""
        echo "查看结果: cat $SCRIPT_DIR/kv-quant-results/summary_*.json | tail -1"
        break
    fi

    # 等待 30 秒
    echo ""
    echo "⏱️  30 秒后自动刷新..."
    sleep 30
done
