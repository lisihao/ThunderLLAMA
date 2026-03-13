#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Building Hash Lookup Benchmark                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Compile with optimizations
clang++ -std=c++17 -O3 -o benchmark_hash_lookup benchmark_hash_lookup.cpp

echo "✅ Compiled successfully"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Running Benchmark                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

./benchmark_hash_lookup

echo ""
echo "Benchmark complete. Binary: $(pwd)/benchmark_hash_lookup"
