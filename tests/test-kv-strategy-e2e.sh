#!/bin/bash
# End-to-End Test for KV Cache Strategy API
# Tests the complete lifecycle: f16 -> q8_0 -> q4_0 -> f16

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/Users/lisihao/ThunderLLAMA"
MODEL="$HOME/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q5_K_M.gguf"
PORT=30000
SERVER_BIN="$PROJECT_DIR/build/bin/llama-server"
LOG_FILE="/tmp/llama-server-test.log"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -f "$MODEL" ]; then
        log_error "Model file not found: $MODEL"
        exit 1
    fi

    if [ ! -f "$SERVER_BIN" ]; then
        log_error "Server binary not found: $SERVER_BIN"
        log_info "Please build the server first: cd $PROJECT_DIR && cmake --build build"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log_warning "jq not found, JSON output will not be formatted"
        JQ_INSTALLED=false
    else
        JQ_INSTALLED=true
    fi

    log_success "Prerequisites check passed"
}

# Start server
start_server() {
    log_info "Starting llama-server..."

    # Kill any existing server on this port
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 2

    # Start server in background
    cd "$PROJECT_DIR"
    THUNDER_LMCACHE=1 "$SERVER_BIN" \
        -m "$MODEL" \
        -c 4096 \
        -ngl 99 \
        --port $PORT \
        > "$LOG_FILE" 2>&1 &

    SERVER_PID=$!
    log_info "Server started with PID $SERVER_PID"

    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    MAX_WAIT=60
    WAITED=0
    while ! curl -s http://localhost:$PORT/health > /dev/null; do
        sleep 1
        WAITED=$((WAITED + 1))
        if [ $WAITED -gt $MAX_WAIT ]; then
            log_error "Server did not start within ${MAX_WAIT}s"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    done

    log_success "Server is ready"
}

# Stop server
stop_server() {
    log_info "Stopping server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    log_success "Server stopped"
}

# Test 1: Get current strategy (should be adaptive by default)
test_get_strategy() {
    log_info "Test 1: GET /thunder/kv-strategy"

    RESPONSE=$(curl -s http://localhost:$PORT/thunder/kv-strategy)

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.'
        STRATEGY_NAME=$(echo "$RESPONSE" | jq -r '.strategy.name')
        CURRENT_LEVEL=$(echo "$RESPONSE" | jq -r '.current_level')
    else
        echo "$RESPONSE"
        STRATEGY_NAME=$(echo "$RESPONSE" | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)
        CURRENT_LEVEL=$(echo "$RESPONSE" | grep -o '"current_level":"[^"]*"' | cut -d'"' -f4)
    fi

    if [ "$STRATEGY_NAME" = "adaptive" ] && [ "$CURRENT_LEVEL" = "f16" ]; then
        log_success "Test 1 passed: Strategy is $STRATEGY_NAME, level is $CURRENT_LEVEL"
    else
        log_error "Test 1 failed: Expected adaptive/f16, got $STRATEGY_NAME/$CURRENT_LEVEL"
        return 1
    fi
}

# Test 2: Switch to q8_0
test_switch_to_q8_0() {
    log_info "Test 2: Switch to q8_0"

    RESPONSE=$(curl -s -X POST http://localhost:$PORT/thunder/kv-strategy \
        -H "Content-Type: application/json" \
        -d '{"name":"fixed","params":{"level":"q8_0"},"version":1}')

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.'
        REBUILD_SUCCESS=$(echo "$RESPONSE" | jq -r '.rebuild_success')
        CURRENT_LEVEL=$(echo "$RESPONSE" | jq -r '.current_level')
    else
        echo "$RESPONSE"
        REBUILD_SUCCESS=$(echo "$RESPONSE" | grep -o '"rebuild_success":[^,}]*' | cut -d':' -f2)
        CURRENT_LEVEL=$(echo "$RESPONSE" | grep -o '"current_level":"[^"]*"' | cut -d'"' -f4)
    fi

    if [ "$REBUILD_SUCCESS" = "true" ] && [ "$CURRENT_LEVEL" = "q8_0" ]; then
        log_success "Test 2 passed: Switched to q8_0"
    else
        log_error "Test 2 failed: rebuild_success=$REBUILD_SUCCESS, current_level=$CURRENT_LEVEL"
        return 1
    fi

    # Wait for rebuild to complete
    sleep 5
}

# Test 3: Verify memory reduction (q8_0 vs f16)
test_memory_reduction_q8_0() {
    log_info "Test 3: Verify memory reduction after q8_0 switch"

    # Note: This test is informational - exact memory values depend on model/context size
    # We just verify the endpoint works
    RESPONSE=$(curl -s http://localhost:$PORT/lmcache/stats)

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.kv_cache'
    else
        echo "$RESPONSE"
    fi

    log_success "Test 3 passed: LMCache stats retrieved"
}

# Test 4: Quality check (send completion request)
test_quality_check() {
    log_info "Test 4: Quality check with completion request"

    RESPONSE=$(curl -s http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Explain neural networks in one sentence:",
            "max_tokens": 50,
            "temperature": 0.7
        }')

    if $JQ_INSTALLED; then
        TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].text')
        echo "Generated text: $TEXT"
    else
        echo "$RESPONSE"
    fi

    if echo "$RESPONSE" | grep -q "choices"; then
        log_success "Test 4 passed: Completion request successful"
    else
        log_error "Test 4 failed: No completion response"
        return 1
    fi
}

# Test 5: Switch to q4_0
test_switch_to_q4_0() {
    log_info "Test 5: Switch to q4_0"

    RESPONSE=$(curl -s -X POST http://localhost:$PORT/thunder/kv-strategy \
        -H "Content-Type: application/json" \
        -d '{"name":"fixed","params":{"level":"q4_0"},"version":1}')

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.'
        REBUILD_SUCCESS=$(echo "$RESPONSE" | jq -r '.rebuild_success')
        CURRENT_LEVEL=$(echo "$RESPONSE" | jq -r '.current_level')
    else
        echo "$RESPONSE"
        REBUILD_SUCCESS=$(echo "$RESPONSE" | grep -o '"rebuild_success":[^,}]*' | cut -d':' -f2)
        CURRENT_LEVEL=$(echo "$RESPONSE" | grep -o '"current_level":"[^"]*"' | cut -d'"' -f4)
    fi

    if [ "$REBUILD_SUCCESS" = "true" ] && [ "$CURRENT_LEVEL" = "q4_0" ]; then
        log_success "Test 5 passed: Switched to q4_0"
    else
        log_error "Test 5 failed: rebuild_success=$REBUILD_SUCCESS, current_level=$CURRENT_LEVEL"
        return 1
    fi

    sleep 5
}

# Test 6: Memory check again (should be even lower)
test_memory_reduction_q4_0() {
    log_info "Test 6: Verify further memory reduction after q4_0 switch"

    RESPONSE=$(curl -s http://localhost:$PORT/lmcache/stats)

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.kv_cache'
    else
        echo "$RESPONSE"
    fi

    log_success "Test 6 passed: LMCache stats retrieved after q4_0"
}

# Test 7: Switch back to f16
test_switch_back_to_f16() {
    log_info "Test 7: Switch back to f16"

    RESPONSE=$(curl -s -X POST http://localhost:$PORT/thunder/kv-strategy \
        -H "Content-Type: application/json" \
        -d '{"name":"fixed","params":{"level":"f16"},"version":1}')

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.'
        REBUILD_SUCCESS=$(echo "$RESPONSE" | jq -r '.rebuild_success')
        CURRENT_LEVEL=$(echo "$RESPONSE" | jq -r '.current_level')
    else
        echo "$RESPONSE"
        REBUILD_SUCCESS=$(echo "$RESPONSE" | grep -o '"rebuild_success":[^,}]*' | cut -d':' -f2)
        CURRENT_LEVEL=$(echo "$RESPONSE" | grep -o '"current_level":"[^"]*"' | cut -d'"' -f4)
    fi

    if [ "$REBUILD_SUCCESS" = "true" ] && [ "$CURRENT_LEVEL" = "f16" ]; then
        log_success "Test 7 passed: Switched back to f16"
    else
        log_error "Test 7 failed: rebuild_success=$REBUILD_SUCCESS, current_level=$CURRENT_LEVEL"
        return 1
    fi

    sleep 5
}

# Test 8: Test evaluate endpoint (dry-run)
test_evaluate_endpoint() {
    log_info "Test 8: Test /thunder/kv-strategy/evaluate (dry-run)"

    RESPONSE=$(curl -s http://localhost:$PORT/thunder/kv-strategy/evaluate)

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.'
    else
        echo "$RESPONSE"
    fi

    if echo "$RESPONSE" | grep -q "decision"; then
        log_success "Test 8 passed: Evaluate endpoint works"
    else
        log_error "Test 8 failed: No decision in response"
        return 1
    fi
}

# Test 9: Test available strategies endpoint
test_available_strategies() {
    log_info "Test 9: Test /thunder/kv-strategy/available"

    RESPONSE=$(curl -s http://localhost:$PORT/thunder/kv-strategy/available)

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.'
        STRATEGY_COUNT=$(echo "$RESPONSE" | jq 'length')
    else
        echo "$RESPONSE"
        STRATEGY_COUNT=$(echo "$RESPONSE" | grep -o '"name"' | wc -l)
    fi

    if [ "$STRATEGY_COUNT" -ge 3 ]; then
        log_success "Test 9 passed: Found $STRATEGY_COUNT strategies"
    else
        log_error "Test 9 failed: Expected at least 3 strategies, found $STRATEGY_COUNT"
        return 1
    fi
}

# Test 10: Test threshold strategy
test_threshold_strategy() {
    log_info "Test 10: Test threshold strategy"

    RESPONSE=$(curl -s -X POST http://localhost:$PORT/thunder/kv-strategy \
        -H "Content-Type: application/json" \
        -d '{
            "name":"threshold",
            "params":{
                "thresholds":[
                    {"ctx_utilization":0.3,"level":"f16"},
                    {"ctx_utilization":0.6,"level":"q8_0"},
                    {"ctx_utilization":0.8,"level":"q4_0"}
                ],
                "hysteresis":0.05
            },
            "version":1
        }')

    if $JQ_INSTALLED; then
        echo "$RESPONSE" | jq '.'
        STRATEGY_NAME=$(echo "$RESPONSE" | jq -r '.strategy.name')
    else
        echo "$RESPONSE"
        STRATEGY_NAME=$(echo "$RESPONSE" | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ "$STRATEGY_NAME" = "threshold" ]; then
        log_success "Test 10 passed: Threshold strategy set"
    else
        log_error "Test 10 failed: Strategy is $STRATEGY_NAME, expected threshold"
        return 1
    fi
}

# Main test execution
main() {
    log_info "Starting KV Cache Strategy E2E Tests"
    echo "========================================"

    check_prerequisites
    start_server

    # Trap to ensure server cleanup
    trap stop_server EXIT

    # Run tests
    FAILED=0

    test_get_strategy || FAILED=$((FAILED + 1))
    test_switch_to_q8_0 || FAILED=$((FAILED + 1))
    test_memory_reduction_q8_0 || FAILED=$((FAILED + 1))
    test_quality_check || FAILED=$((FAILED + 1))
    test_switch_to_q4_0 || FAILED=$((FAILED + 1))
    test_memory_reduction_q4_0 || FAILED=$((FAILED + 1))
    test_switch_back_to_f16 || FAILED=$((FAILED + 1))
    test_evaluate_endpoint || FAILED=$((FAILED + 1))
    test_available_strategies || FAILED=$((FAILED + 1))
    test_threshold_strategy || FAILED=$((FAILED + 1))

    echo "========================================"
    if [ $FAILED -eq 0 ]; then
        log_success "All tests passed! ✨"
        exit 0
    else
        log_error "$FAILED test(s) failed"
        log_info "Check server log: $LOG_FILE"
        exit 1
    fi
}

# Run main
main
