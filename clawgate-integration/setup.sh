#!/bin/bash
#
# Setup Script for Clawgate Integration
#
# This script sets up the complete ContextPilot + Clawgate + ThunderLLAMA + LMCache stack
#
# Author: Claude (Anthropic AI)
# Date: 2026-03-12

set -e  # Exit on error

echo "======================================================================"
echo "  Clawgate Integration Setup"
echo "======================================================================"

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

echo ""
echo "[1/6] Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python: $PYTHON_VERSION"

# Check if in ThunderLLAMA directory
if [ ! -f "../build/CMakeCache.txt" ]; then
    echo "❌ Not in ThunderLLAMA/clawgate-integration directory"
    exit 1
fi

echo "✓ ThunderLLAMA directory detected"

# ============================================================================
# Step 2: Install Python Dependencies
# ============================================================================

echo ""
echo "[2/6] Installing Python dependencies..."

pip3 install -q contextpilot httpx || {
    echo "❌ Failed to install Python packages"
    exit 1
}

echo "✓ ContextPilot installed"
echo "✓ httpx installed"

# ============================================================================
# Step 3: Build ThunderLLAMA with Eviction Support
# ============================================================================

echo ""
echo "[3/6] Building ThunderLLAMA with eviction support..."

cd ..
mkdir -p build
cd build

if [ ! -f "CMakeCache.txt" ]; then
    echo "  Running cmake..."
    cmake .. > /dev/null 2>&1 || {
        echo "❌ CMake failed"
        exit 1
    }
fi

echo "  Compiling llama library..."
make llama -j8 > /dev/null 2>&1 || {
    echo "❌ Build failed"
    exit 1
}

echo "✓ ThunderLLAMA built successfully"

# Check if thunder-cache exists
if [ -f "bin/thunder-cache" ]; then
    echo "✓ thunder-cache CLI tool available"
else
    echo "⚠ thunder-cache not found (optional)"
fi

cd ../clawgate-integration

# ============================================================================
# Step 4: Setup Environment Variables
# ============================================================================

echo ""
echo "[4/6] Setting up environment..."

# Create env file
cat > .env <<EOF
# Clawgate Integration Environment
# Generated on $(date)

# ThunderLLAMA Configuration
THUNDERLLAMA_URL=http://localhost:30000

# LMCache Configuration
LMCACHE_ENABLED=true
THUNDER_LMCACHE_DISK_PATH=~/.openclaw/lmcache.bin

# ContextPilot Configuration
CONTEXTPILOT_URL=  # Empty = embedded mode
CONTEXTPILOT_GPU=false

# Eviction Sync
EVICTION_SYNC_ENABLED=true
CONTEXTPILOT_INDEX_URL=http://localhost:8000/evict
EOF

echo "✓ Environment file created: .env"
echo "  Edit .env to customize configuration"

# ============================================================================
# Step 5: Create Cache Directory
# ============================================================================

echo ""
echo "[5/6] Creating cache directory..."

mkdir -p ~/.openclaw
echo "✓ Cache directory: ~/.openclaw"

# ============================================================================
# Step 6: Run Tests
# ============================================================================

echo ""
echo "[6/6] Running integration tests..."

# Make scripts executable
chmod +x test_integration.py
chmod +x benchmark_multi_agent.py

echo "  Running basic test..."
if python3 test_integration.py --test basic 2>&1 | grep -q "TEST PASSED"; then
    echo "✓ Basic test passed"
else
    echo "⚠ Basic test failed (ThunderLLAMA may not be running)"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "======================================================================"
echo "  Setup Complete!"
echo "======================================================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Start ThunderLLAMA server:"
echo "   cd ../build"
echo "   ./bin/llama-server --model /path/to/model.gguf"
echo ""
echo "2. Run integration tests:"
echo "   cd ../clawgate-integration"
echo "   python3 test_integration.py --test all"
echo ""
echo "3. Run performance benchmarks:"
echo "   python3 benchmark_multi_agent.py"
echo ""
echo "4. Check cache stats:"
echo "   ../build/bin/thunder-cache stats ~/.openclaw/lmcache.bin"
echo ""
echo "Documentation: README.md"
echo ""
echo "======================================================================"
