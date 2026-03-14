#!/bin/bash
# Build and run chunk index optimization test

set -e

echo "Building test..."
clang++ -std=c++17 -O3 -I../include -I../ggml/include \
    test_chunk_index_optimization.cpp \
    -o test_chunk_index_optimization

echo "Running test..."
./test_chunk_index_optimization

echo ""
echo "Test completed!"
