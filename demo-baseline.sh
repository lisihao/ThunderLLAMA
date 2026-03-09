#!/bin/bash
# ThunderLLAMA Baseline 性能测试

set -e

MODEL="${MODEL:-$HOME/models/Qwen3-1.7B-Q4_K_M.gguf}"
BUILD_DIR="${BUILD_DIR:-build}"
BIN="./$BUILD_DIR/bin/llama-bench"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 ThunderLLAMA Baseline 性能测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model: $(basename $MODEL)"
echo "Device: $(sysctl -n machdep.cpu.brand_string)"
echo "Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')GB"
echo ""

# 检查模型是否存在
if [ ! -f "$MODEL" ]; then
    echo "❌ 模型不存在: $MODEL"
    exit 1
fi

# 检查可执行文件
if [ ! -f "$BIN" ]; then
    echo "❌ 可执行文件不存在，需要先编译"
    echo "运行: cmake -B build && cmake --build build --config Release -j10"
    exit 1
fi

# 运行 benchmark
echo "🧪 运行 Baseline Benchmark..."
echo ""

$BIN \
    -m "$MODEL" \
    -ngl 99 \
    -p 512 \
    -n 128 \
    2>&1 | tee /tmp/thunderllama-baseline.log

# 提取性能指标
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 性能指标"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

grep "pp512" /tmp/thunderllama-baseline.log
grep "tg128" /tmp/thunderllama-baseline.log

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Baseline 测试完成"
echo "日志已保存到: /tmp/thunderllama-baseline.log"
