#pragma once

#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <mutex>
#include <chrono>

using json = nlohmann::ordered_json;

// User-friendly quantization levels for KV cache
enum class kv_quant_level {
    F16,    // Full precision (ggml_type: GGML_TYPE_F16)
    Q8_0,   // 8-bit quantization (ggml_type: GGML_TYPE_Q8_0)
    Q4_0,   // 4-bit quantization (ggml_type: GGML_TYPE_Q4_0)
};

// Strategy configuration (JSON serializable for HTTP injection)
struct kv_strategy_config {
    std::string name;        // Strategy name: "fixed", "threshold", "adaptive"
    json params;             // Strategy-specific parameters
    int version;             // Config version for compatibility
    std::string source;      // Source: "default", "env", "api"

    // Serialization
    static kv_strategy_config from_json(const json & j);
    json to_json() const;
};

// Runtime metrics for strategy evaluation
struct kv_strategy_metrics {
    // Context utilization
    int32_t n_ctx_total;       // Total context size
    int32_t n_ctx_used;        // Currently used context tokens

    // Memory pressure
    size_t kv_cache_size;      // Current KV cache memory usage (bytes)
    size_t kv_cache_max;       // Maximum KV cache capacity (bytes)

    // Slot load
    int32_t n_slots_total;     // Total number of slots
    int32_t n_slots_active;    // Number of active slots

    // Prompt characteristics
    int32_t avg_prompt_length; // Average prompt length of active slots

    // Timestamps
    int64_t timestamp_us;      // Timestamp when metrics were collected

    // Serialization
    json to_json() const;
};

// Decision result from strategy evaluation
struct kv_strategy_decision {
    kv_quant_level level;      // Chosen quantization level
    bool requires_rebuild;     // Whether context rebuild is needed
    std::string reason;        // Human-readable reason for this decision
    json metadata;             // Strategy-specific metadata (e.g., score, signals)

    // Serialization
    json to_json() const;
};

// Strategy evaluator function type
using kv_strategy_evaluator_fn = std::function<kv_strategy_decision(
    const kv_strategy_config & config,
    const kv_strategy_metrics & metrics,
    kv_quant_level current_level
)>;

// Strategy registry: manages available strategies
class kv_strategy_registry {
public:
    kv_strategy_registry();

    // Register a new strategy
    void register_strategy(const std::string & name, kv_strategy_evaluator_fn evaluator);

    // Check if strategy exists
    bool has_strategy(const std::string & name) const;

    // Get strategy evaluator
    kv_strategy_evaluator_fn get_evaluator(const std::string & name) const;

    // List all registered strategy names
    std::vector<std::string> list_strategies() const;

private:
    std::map<std::string, kv_strategy_evaluator_fn> strategies_;
    mutable std::mutex mutex_;
};

// Main strategy manager
class kv_strategy_manager {
public:
    kv_strategy_manager();

    // Initialize with default strategy
    void init();

    // Get/Set current strategy
    kv_strategy_config get_strategy() const;
    void set_strategy(const kv_strategy_config & config);

    // Get current quantization level
    kv_quant_level get_current_level() const;

    // Update runtime metrics
    void update_metrics(const kv_strategy_metrics & metrics);

    // Get latest metrics
    kv_strategy_metrics get_metrics() const;

    // Evaluate strategy (returns decision without applying)
    kv_strategy_decision evaluate(const kv_strategy_metrics & metrics);

    // Apply decision (update internal state)
    void apply_decision(const kv_strategy_decision & decision);

    // Get decision history (last N decisions)
    json get_history(size_t max_count = 10) const;

    // Get strategy registry
    const kv_strategy_registry & registry() const { return registry_; }

    // Conversion utilities
    static ggml_type level_to_ggml_type(kv_quant_level level);
    static kv_quant_level ggml_type_to_level(ggml_type type);
    static std::string level_to_string(kv_quant_level level);
    static kv_quant_level string_to_level(const std::string & str);

private:
    kv_strategy_registry registry_;
    kv_strategy_config current_strategy_;
    kv_quant_level current_level_;
    kv_strategy_metrics latest_metrics_;

    // Decision history (ring buffer)
    struct decision_record {
        int64_t timestamp_us;
        kv_strategy_decision decision;
        kv_strategy_metrics metrics;
    };
    std::vector<decision_record> history_;
    static constexpr size_t MAX_HISTORY = 100;

    // Cooldown tracking (to prevent oscillation)
    int64_t last_rebuild_us_;
    static constexpr int64_t MIN_REBUILD_INTERVAL_US = 30 * 1000000; // 30 seconds

    mutable std::mutex mutex_;
};

// Built-in strategy evaluators (implemented in server-kv-strategy.cpp)
namespace kv_strategies {
    kv_strategy_decision evaluate_fixed_strategy(
        const kv_strategy_config & config,
        const kv_strategy_metrics & metrics,
        kv_quant_level current_level
    );

    kv_strategy_decision evaluate_threshold_strategy(
        const kv_strategy_config & config,
        const kv_strategy_metrics & metrics,
        kv_quant_level current_level
    );

    kv_strategy_decision evaluate_adaptive_strategy(
        const kv_strategy_config & config,
        const kv_strategy_metrics & metrics,
        kv_quant_level current_level
    );
}
