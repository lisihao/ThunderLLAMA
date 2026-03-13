/**
 * ContextPilot Chunk Hash Lookup Performance Benchmark
 *
 * Compares:
 * - Current: std::hash<std::string>{}(chunk_hash + "_layer_" + std::to_string(il))
 * - Optimized: (chunk_base_hash << 32) | static_cast<uint64_t>(il)
 *
 * Expected improvement: +10% query performance
 */

#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace std::chrono;

// ============================================================
// Test Configuration
// ============================================================

const int NUM_ITERATIONS = 100000;  // 100K hash lookups
const int NUM_LAYERS = 80;          // Qwen3-30B has 80 layers
const int NUM_CHUNKS = 10;          // Simulate 10 chunks per request

// ============================================================
// Current Implementation (String Concatenation)
// ============================================================

uint64_t hash_lookup_string(const string& chunk_hash, int il) {
    return std::hash<std::string>{}(chunk_hash + "_layer_" + std::to_string(il));
}

// ============================================================
// Optimized Implementation (Numeric Hash)
// ============================================================

uint64_t hash_lookup_numeric(uint64_t chunk_base_hash, int il) {
    return (chunk_base_hash << 32) | static_cast<uint64_t>(il);
}

// ============================================================
// Benchmark Function with Multiple Runs for Variance
// ============================================================

// Prevent compiler from optimizing away the computation
static void escape(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

template <typename Func>
vector<double> benchmark_multi_run(const string& name, Func&& func, int num_runs) {
    cout << "Running " << name << " (" << num_runs << " runs)..." << flush;

    vector<double> times;

    for (int run = 0; run < num_runs; ++run) {
        auto start = high_resolution_clock::now();

        volatile uint64_t result = 0;  // Volatile to prevent optimization
        for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
            for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
                for (int layer = 0; layer < NUM_LAYERS; ++layer) {
                    uint64_t val = func(chunk, layer);
                    // Force computation - compiler cannot optimize away volatile writes
                    result = result + val;
                }
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start).count();

        double avg_ns = static_cast<double>(duration) / (NUM_ITERATIONS * NUM_CHUNKS * NUM_LAYERS);
        times.push_back(avg_ns);
    }

    cout << " Done" << endl;
    return times;
}

// Calculate median
double median(vector<double> times) {
    sort(times.begin(), times.end());
    size_t n = times.size();
    return (n % 2 == 0) ? (times[n/2-1] + times[n/2]) / 2.0 : times[n/2];
}

// Calculate standard deviation
double stddev(const vector<double>& times, double mean) {
    double sum_sq_diff = 0.0;
    for (double t : times) {
        sum_sq_diff += (t - mean) * (t - mean);
    }
    return sqrt(sum_sq_diff / times.size());
}

// ============================================================
// Main
// ============================================================

int main() {
    cout << "╔══════════════════════════════════════════════════════════════╗" << endl;
    cout << "║  ContextPilot Chunk Hash Lookup Performance Benchmark       ║" << endl;
    cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
    cout << "║  Compiler:    " << __VERSION__ << endl;
    cout << "║  Build Date:  " << __DATE__ " " << __TIME__ << endl;
    cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
    cout << "║  Iterations:  " << setw(10) << NUM_ITERATIONS << " × " << NUM_CHUNKS << " chunks × " << NUM_LAYERS << " layers" << endl;
    cout << "║  Total Ops:   " << setw(10) << (NUM_ITERATIONS * NUM_CHUNKS * NUM_LAYERS) << " per run" << endl;
    cout << "║  Runs:        " << setw(10) << 10 << " (for variance analysis)" << endl;
    cout << "╚══════════════════════════════════════════════════════════════╝" << endl;
    cout << endl;

    // Pre-generate test data
    vector<string> chunk_hashes(NUM_CHUNKS);
    vector<uint64_t> chunk_base_hashes(NUM_CHUNKS);

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        chunk_hashes[i] = "chunk_hash_" + to_string(i) + "_random_suffix_12345";
        chunk_base_hashes[i] = std::hash<std::string>{}(chunk_hashes[i]);
    }

    // ============================================================
    // Benchmark 1: Current String Concatenation (10 runs)
    // ============================================================

    vector<double> times_string = benchmark_multi_run("String Concatenation", [&](int chunk, int layer) {
        return hash_lookup_string(chunk_hashes[chunk], layer);
    }, 10);

    // ============================================================
    // Benchmark 2: Optimized Numeric Hash (10 runs)
    // ============================================================

    vector<double> times_numeric = benchmark_multi_run("Numeric Hash", [&](int chunk, int layer) {
        return hash_lookup_numeric(chunk_base_hashes[chunk], layer);
    }, 10);

    // ============================================================
    // Calculate Statistics
    // ============================================================

    double median_string = median(times_string);
    double median_numeric = median(times_numeric);

    double mean_string = 0, mean_numeric = 0;
    for (double t : times_string) mean_string += t;
    for (double t : times_numeric) mean_numeric += t;
    mean_string /= times_string.size();
    mean_numeric /= times_numeric.size();

    double stddev_string = stddev(times_string, mean_string);
    double stddev_numeric = stddev(times_numeric, mean_numeric);

    double min_string = *min_element(times_string.begin(), times_string.end());
    double max_string = *max_element(times_string.begin(), times_string.end());
    double min_numeric = *min_element(times_numeric.begin(), times_numeric.end());
    double max_numeric = *max_element(times_numeric.begin(), times_numeric.end());

    // Use median for speedup (more robust)
    double speedup = median_string / median_numeric;
    double improvement = (1.0 - median_numeric / median_string) * 100.0;

    cout << endl;
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
    cout << "Results (10 runs, using MEDIAN for robustness)" << endl;
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;

    cout << fixed << setprecision(2);
    cout << "String Concatenation:" << endl;
    cout << "  Median:              " << setw(8) << median_string << " ns/op" << endl;
    cout << "  Mean ± StdDev:       " << setw(8) << mean_string << " ± " << stddev_string << " ns/op" << endl;
    cout << "  Range:               " << setw(8) << min_string << " - " << max_string << " ns/op" << endl;
    cout << "  Variance:            " << setw(8) << (stddev_string / mean_string * 100.0) << "%" << endl;
    cout << endl;

    cout << "Numeric Hash:" << endl;
    cout << "  Median:              " << setw(8) << median_numeric << " ns/op" << endl;
    cout << "  Mean ± StdDev:       " << setw(8) << mean_numeric << " ± " << stddev_numeric << " ns/op" << endl;
    cout << "  Range:               " << setw(8) << min_numeric << " - " << max_numeric << " ns/op" << endl;
    cout << "  Variance:            " << setw(8) << (stddev_numeric / mean_numeric * 100.0) << "%" << endl;
    cout << endl;

    cout << "Speedup (median):      " << setw(8) << speedup << "x" << endl;
    cout << "Improvement:           " << setw(8) << improvement << "%" << endl;

    cout << endl;

    // ============================================================
    // Throughput Analysis (based on median)
    // ============================================================

    double throughput_string = 1e9 / median_string;  // ops/second
    double throughput_numeric = 1e9 / median_numeric;

    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
    cout << "Throughput" << endl;
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;

    cout << scientific << setprecision(2);
    cout << "String Concatenation:  " << throughput_string << " ops/sec" << endl;
    cout << "Numeric Hash:          " << throughput_numeric << " ops/sec" << endl;
    cout << endl;

    // ============================================================
    // Conclusion
    // ============================================================

    cout << fixed << setprecision(0);
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;
    cout << "Conclusion" << endl;
    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;

    if (improvement >= 5.0) {
        cout << "✅ Optimization is WORTHWHILE (+";
        cout << fixed << setprecision(1) << improvement << "%)" << endl;
        cout << "   Proceed with implementation." << endl;
    } else {
        cout << "❌ Optimization is NOT WORTHWHILE (+";
        cout << fixed << setprecision(1) << improvement << "%)" << endl;
        cout << "   Improvement < 5% threshold." << endl;
    }

    cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << endl;

    return 0;
}
