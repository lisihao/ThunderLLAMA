#!/bin/bash
# ThunderLLAMA Decode-First + Adaptive Chunk Prefill Demo
# Tests mixed workload TTFT improvement with chunked prefill scheduling

set -e

MODEL="${MODEL:-$HOME/models/Qwen3-1.7B-Q4_K_M.gguf}"
BUILD_DIR="${BUILD_DIR:-build}"
SERVER="./$BUILD_DIR/bin/llama-server"
CLIENT="$(dirname "$0")/demo-decode-first-client.py"
PORT="${PORT:-8080}"
N_SLOTS=4
CTX_SIZE=32768

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 ThunderLLAMA Decode-First Scheduling Demo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model: $(basename "$MODEL")"
echo "Device: $(sysctl -n machdep.cpu.brand_string)"
echo "Slots: $N_SLOTS | Context: $CTX_SIZE"
echo ""

# Validate files
if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

if [ ! -f "$SERVER" ]; then
    echo "❌ Server binary not found: $SERVER"
    echo "Run: cmake -B build && cmake --build build --config Release -j10"
    exit 1
fi

if [ ! -f "$CLIENT" ]; then
    echo "❌ Test client not found: $CLIENT"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found"
    exit 1
fi

# Check if requests module is available
if ! python3 -c "import requests" 2>/dev/null; then
    echo "❌ Python 'requests' module not found. Install with: pip3 install requests"
    exit 1
fi

cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_server() {
    echo -n "⏳ Waiting for server to start"
    for i in $(seq 1 30); do
        if curl -s "http://localhost:$PORT/health" | grep -q "ok" 2>/dev/null; then
            echo " ✅"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " ❌ Timeout"
    return 1
}

run_test() {
    local label="$1"
    local chunk_val="$2"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 Test: $label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Start server
    if [ -n "$chunk_val" ]; then
        export THUNDERLLAMA_CHUNK_PREFILL="$chunk_val"
        echo "   THUNDERLLAMA_CHUNK_PREFILL=$chunk_val"
    else
        unset THUNDERLLAMA_CHUNK_PREFILL
        echo "   THUNDERLLAMA_CHUNK_PREFILL=(unset, default adaptive)"
    fi

    "$SERVER" \
        -m "$MODEL" \
        -ngl 99 \
        -np "$N_SLOTS" \
        -c "$CTX_SIZE" \
        --cont-batching \
        --port "$PORT" \
        --log-disable \
        2>/dev/null &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "❌ Server failed to start"
        kill "$SERVER_PID" 2>/dev/null || true
        return 1
    fi

    # Run the mixed workload test
    python3 "$CLIENT" \
        --port "$PORT" \
        --long-tokens 4096 \
        --short-requests 3 \
        --delay 0.5

    # Stop server
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""

    echo ""
    sleep 1
}

# ───────────────────────────────────────────────
# Test A: No chunking (original behavior - large n_batch)
# ───────────────────────────────────────────────
run_test "Baseline (no chunk limit)" "999999"

# ───────────────────────────────────────────────
# Test B: Adaptive chunking (default)
# ───────────────────────────────────────────────
run_test "Adaptive chunk (default: n_batch/n_slots)" ""

# ───────────────────────────────────────────────
# Test C: Fixed chunk size 512
# ───────────────────────────────────────────────
run_test "Fixed chunk (512 tokens/slot/iter)" "512"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All tests completed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Expected results:"
echo "  • Baseline: short request TTFT blocked by long prefill (~seconds)"
echo "  • Chunked:  short request TTFT significantly lower (<1s)"
echo "  • Long request total time may increase slightly (5-10%)"
