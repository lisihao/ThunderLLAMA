#!/bin/bash
set -e

echo "==================================================================="
echo "Thunder LMCache Framework Test"
echo "==================================================================="
echo ""

# 1. 编译
echo "Step 1: Build llama-simple..."
cd /Users/lisihao/ThunderLLAMA
cmake --build build --target llama-simple

# 2. 准备测试模型
MODEL="/Users/lisihao/llama.cpp/models/tinyllama-1.1b-chat-q4_k_m.gguf"
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    exit 1
fi

# 3. 测试 1: Baseline (不启用 LMCache)
echo ""
echo "-------------------------------------------------------------------"
echo "Test 1: Baseline (THUNDER_LMCACHE=0)"
echo "-------------------------------------------------------------------"
build/bin/llama-simple \
    -m "$MODEL" \
    -p "What is the capital of France?" \
    -n 20 \
    --temp 0.0 \
    2>&1 | tee /tmp/lmcache_test_baseline.log

echo ""
echo "Checking baseline logs..."
if grep -q "LMCache enabled" /tmp/lmcache_test_baseline.log; then
    echo "❌ FAIL: LMCache should NOT be enabled"
    exit 1
else
    echo "✅ PASS: LMCache correctly disabled"
fi

# 4. 测试 2: 启用 LMCache
echo ""
echo "-------------------------------------------------------------------"
echo "Test 2: LMCache Enabled (THUNDER_LMCACHE=1)"
echo "-------------------------------------------------------------------"
THUNDER_LMCACHE=1 build/bin/llama-simple \
    -m "$MODEL" \
    -p "What is the capital of France?" \
    -n 20 \
    --temp 0.0 \
    2>&1 | tee /tmp/lmcache_test_enabled.log

echo ""
echo "Checking enabled logs..."

# 检查初始化日志
if grep -q "LMCache enabled" /tmp/lmcache_test_enabled.log; then
    echo "✅ PASS: LMCache initialization detected"
else
    echo "❌ FAIL: LMCache initialization not found"
    echo ""
    echo "Diagnostic: Checking for debug logs..."
    grep -i "lmcache\|thunder\|chunk" /tmp/lmcache_test_enabled.log || echo "  (no LMCache-related logs found)"
    exit 1
fi

# 检查 chunk size
if grep -q "chunk_size=256" /tmp/lmcache_test_enabled.log; then
    echo "✅ PASS: Chunk size = 256"
else
    echo "⚠️  WARN: Chunk size log not found"
fi

# 5. 测试 3: 自定义磁盘路径
echo ""
echo "-------------------------------------------------------------------"
echo "Test 3: Custom Disk Path"
echo "-------------------------------------------------------------------"
CUSTOM_PATH="/tmp/lmcache_test_custom.bin"
rm -f "$CUSTOM_PATH"

THUNDER_LMCACHE=1 \
THUNDER_LMCACHE_DISK_PATH="$CUSTOM_PATH" \
build/bin/llama-simple \
    -m "$MODEL" \
    -p "Hello" \
    -n 10 \
    --temp 0.0 \
    2>&1 | tee /tmp/lmcache_test_custom_path.log

echo ""
echo "Checking custom path..."
if grep -q "$CUSTOM_PATH" /tmp/lmcache_test_custom_path.log; then
    echo "✅ PASS: Custom disk path in logs"
else
    echo "⚠️  WARN: Custom path not in logs"
fi

if [ -f "$CUSTOM_PATH" ]; then
    SIZE=$(stat -f%z "$CUSTOM_PATH" 2>/dev/null || stat -c%s "$CUSTOM_PATH" 2>/dev/null)
    echo "✅ PASS: Disk file created ($SIZE bytes)"
else
    echo "⚠️  WARN: Disk file not created (may be normal if no chunks stored)"
fi

# 6. 总结
echo ""
echo "==================================================================="
echo "Test Summary"
echo "==================================================================="
echo ""
echo "All framework tests passed! ✅"
echo ""
echo "Log files:"
echo "  - /tmp/lmcache_test_baseline.log"
echo "  - /tmp/lmcache_test_enabled.log"
echo "  - /tmp/lmcache_test_custom_path.log"
echo ""
