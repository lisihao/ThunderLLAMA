#!/bin/bash
# Build and run disk I/O async optimization test

set -e

echo "Building disk I/O async test..."
clang++ -std=c++17 -O3 -I../include -I../ggml/include -I../ggml/src \
    test_disk_io_async.cpp \
    ../build/src/libllama.a \
    ../build/ggml/src/libggml.a \
    ../build/ggml/src/libggml-cpu.a \
    ../build/ggml/src/libggml-base.a \
    ../build/ggml/src/ggml-blas/libggml-blas.a \
    ../build/ggml/src/ggml-metal/libggml-metal.a \
    -framework Accelerate \
    -framework Foundation \
    -framework Metal \
    -framework MetalKit \
    -lz \
    -lpthread \
    -o test_disk_io_async

echo ""
echo "Running test..."
./test_disk_io_async

echo ""
echo "Test completed!"
