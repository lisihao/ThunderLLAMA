#include "server-kv-strategy.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <sstream>

// ============================================================================
// kv_strategy_config
// ============================================================================

kv_strategy_config kv_strategy_config::from_json(const json & j) {
    kv_strategy_config config;
    config.name = j.value("name", "adaptive");
    config.params = j.value("params", json::object());
    config.version = j.value("version", 1);
    config.source = j.value("source", "api");
    return config;
}

json kv_strategy_config::to_json() const {
    return json{
        {"name", name},
        {"params", params},
        {"version", version},
        {"source", source}
    };
}

// ============================================================================
// kv_strategy_metrics
// ============================================================================

json kv_strategy_metrics::to_json() const {
    double ctx_utilization = n_ctx_total > 0 ? (double)n_ctx_used / n_ctx_total : 0.0;
    double memory_pressure = kv_cache_max > 0 ? (double)kv_cache_size / kv_cache_max : 0.0;
    double slot_load = n_slots_total > 0 ? (double)n_slots_active / n_slots_total : 0.0;

    return json{
        {"n_ctx_total", n_ctx_total},
        {"n_ctx_used", n_ctx_used},
        {"ctx_utilization", ctx_utilization},
        {"kv_cache_size", kv_cache_size},
        {"kv_cache_max", kv_cache_max},
        {"memory_pressure", memory_pressure},
        {"n_slots_total", n_slots_total},
        {"n_slots_active", n_slots_active},
        {"slot_load", slot_load},
        {"avg_prompt_length", avg_prompt_length},
        {"timestamp_us", timestamp_us}
    };
}

// ============================================================================
// kv_strategy_decision
// ============================================================================

json kv_strategy_decision::to_json() const {
    return json{
        {"level", kv_strategy_manager::level_to_string(level)},
        {"requires_rebuild", requires_rebuild},
        {"reason", reason},
        {"metadata", metadata}
    };
}

// ============================================================================
// kv_strategy_registry
// ============================================================================

kv_strategy_registry::kv_strategy_registry() {
    // Register built-in strategies
    register_strategy("fixed", kv_strategies::evaluate_fixed_strategy);
    register_strategy("threshold", kv_strategies::evaluate_threshold_strategy);
    register_strategy("adaptive", kv_strategies::evaluate_adaptive_strategy);
}

void kv_strategy_registry::register_strategy(const std::string & name, kv_strategy_evaluator_fn evaluator) {
    std::lock_guard<std::mutex> lock(mutex_);
    strategies_[name] = evaluator;
}

bool kv_strategy_registry::has_strategy(const std::string & name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return strategies_.find(name) != strategies_.end();
}

kv_strategy_evaluator_fn kv_strategy_registry::get_evaluator(const std::string & name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = strategies_.find(name);
    if (it == strategies_.end()) {
        throw std::runtime_error("strategy not found: " + name);
    }
    return it->second;
}

std::vector<std::string> kv_strategy_registry::list_strategies() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto & [name, _] : strategies_) {
        names.push_back(name);
    }
    return names;
}

// ============================================================================
// kv_strategy_manager
// ============================================================================

kv_strategy_manager::kv_strategy_manager()
    : current_level_(kv_quant_level::F16)
    , last_rebuild_us_(0) {
}

void kv_strategy_manager::init() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Default strategy: adaptive
    current_strategy_.name = "adaptive";
    current_strategy_.params = json{
        {"ctx_weight", 0.4},
        {"memory_weight", 0.3},
        {"load_weight", 0.2},
        {"prompt_weight", 0.1},
        {"memory_emergency_threshold", 0.85},
        {"high_load_threshold", 0.7},
        {"medium_load_threshold", 0.4},
        {"hysteresis", 0.05}
    };
    current_strategy_.version = 1;
    current_strategy_.source = "default";
}

kv_strategy_config kv_strategy_manager::get_strategy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_strategy_;
}

void kv_strategy_manager::set_strategy(const kv_strategy_config & config) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_strategy_ = config;
}

kv_quant_level kv_strategy_manager::get_current_level() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_level_;
}

void kv_strategy_manager::update_metrics(const kv_strategy_metrics & metrics) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_metrics_ = metrics;
}

kv_strategy_metrics kv_strategy_manager::get_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_metrics_;
}

kv_strategy_decision kv_strategy_manager::evaluate(const kv_strategy_metrics & metrics) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Get evaluator for current strategy
    auto evaluator = registry_.get_evaluator(current_strategy_.name);

    // Evaluate
    auto decision = evaluator(current_strategy_, metrics, current_level_);

    // Check cooldown
    int64_t now_us = metrics.timestamp_us;
    if (decision.requires_rebuild) {
        int64_t time_since_last_rebuild = now_us - last_rebuild_us_;
        if (time_since_last_rebuild < MIN_REBUILD_INTERVAL_US) {
            // Too soon, reject rebuild
            decision.requires_rebuild = false;
            decision.reason += " (suppressed by cooldown)";
            decision.metadata["cooldown_remaining_s"] =
                (MIN_REBUILD_INTERVAL_US - time_since_last_rebuild) / 1000000.0;
        }
    }

    return decision;
}

void kv_strategy_manager::apply_decision(const kv_strategy_decision & decision) {
    std::lock_guard<std::mutex> lock(mutex_);

    current_level_ = decision.level;

    if (decision.requires_rebuild) {
        last_rebuild_us_ = latest_metrics_.timestamp_us;
    }

    // Record to history
    decision_record record;
    record.timestamp_us = latest_metrics_.timestamp_us;
    record.decision = decision;
    record.metrics = latest_metrics_;

    history_.push_back(record);
    if (history_.size() > MAX_HISTORY) {
        history_.erase(history_.begin());
    }
}

json kv_strategy_manager::get_history(size_t max_count) const {
    std::lock_guard<std::mutex> lock(mutex_);

    json result = json::array();
    size_t start_idx = history_.size() > max_count ? history_.size() - max_count : 0;

    for (size_t i = start_idx; i < history_.size(); ++i) {
        const auto & record = history_[i];
        result.push_back(json{
            {"timestamp_us", record.timestamp_us},
            {"decision", record.decision.to_json()},
            {"metrics", record.metrics.to_json()}
        });
    }

    return result;
}

// ============================================================================
// Conversion utilities
// ============================================================================

ggml_type kv_strategy_manager::level_to_ggml_type(kv_quant_level level) {
    switch (level) {
        case kv_quant_level::F16:  return GGML_TYPE_F16;
        case kv_quant_level::Q8_0: return GGML_TYPE_Q8_0;
        case kv_quant_level::Q4_0: return GGML_TYPE_Q4_0;
        default: return GGML_TYPE_F16;
    }
}

kv_quant_level kv_strategy_manager::ggml_type_to_level(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:  return kv_quant_level::F16;
        case GGML_TYPE_F32:  return kv_quant_level::F16; // Treat F32 as F16
        case GGML_TYPE_Q8_0: return kv_quant_level::Q8_0;
        case GGML_TYPE_Q4_0: return kv_quant_level::Q4_0;
        default: return kv_quant_level::F16;
    }
}

std::string kv_strategy_manager::level_to_string(kv_quant_level level) {
    switch (level) {
        case kv_quant_level::F16:  return "f16";
        case kv_quant_level::Q8_0: return "q8_0";
        case kv_quant_level::Q4_0: return "q4_0";
        default: return "f16";
    }
}

kv_quant_level kv_strategy_manager::string_to_level(const std::string & str) {
    if (str == "f16" || str == "F16") return kv_quant_level::F16;
    if (str == "q8_0" || str == "Q8_0") return kv_quant_level::Q8_0;
    if (str == "q4_0" || str == "Q4_0") return kv_quant_level::Q4_0;
    return kv_quant_level::F16; // Default
}

// ============================================================================
// Built-in strategies
// ============================================================================

namespace kv_strategies {

// Strategy 1: Fixed level
kv_strategy_decision evaluate_fixed_strategy(
    const kv_strategy_config & config,
    const kv_strategy_metrics & metrics,
    kv_quant_level current_level
) {
    kv_strategy_decision decision;

    // Get target level from params
    std::string level_str = config.params.value("level", "f16");
    decision.level = kv_strategy_manager::string_to_level(level_str);

    // Check if rebuild is needed
    decision.requires_rebuild = (decision.level != current_level);

    decision.reason = "fixed strategy: " + level_str;
    decision.metadata = json{
        {"strategy", "fixed"},
        {"target_level", level_str}
    };

    return decision;
}

// Strategy 2: Threshold-based switching
kv_strategy_decision evaluate_threshold_strategy(
    const kv_strategy_config & config,
    const kv_strategy_metrics & metrics,
    kv_quant_level current_level
) {
    kv_strategy_decision decision;

    // Get thresholds from params
    // Default thresholds:
    // [{"ctx_utilization": 0.3, "level": "f16"},
    //  {"ctx_utilization": 0.6, "level": "q8_0"},
    //  {"ctx_utilization": 0.8, "level": "q4_0"}]
    json thresholds = config.params.value("thresholds", json::array({
        {{"ctx_utilization", 0.3}, {"level", "f16"}},
        {{"ctx_utilization", 0.6}, {"level", "q8_0"}},
        {{"ctx_utilization", 0.8}, {"level", "q4_0"}}
    }));

    double hysteresis = config.params.value("hysteresis", 0.05);

    // Calculate context utilization
    double ctx_util = metrics.n_ctx_total > 0
        ? (double)metrics.n_ctx_used / metrics.n_ctx_total
        : 0.0;

    // Apply hysteresis to avoid oscillation
    // If current level suggests lower utilization, add hysteresis to threshold
    double adjusted_util = ctx_util;
    if (current_level == kv_quant_level::Q8_0 || current_level == kv_quant_level::Q4_0) {
        adjusted_util += hysteresis;
    }

    // Find matching threshold
    decision.level = kv_quant_level::F16; // Default
    for (const auto & threshold : thresholds) {
        double thresh_val = threshold.value("ctx_utilization", 0.0);
        if (adjusted_util >= thresh_val) {
            std::string level_str = threshold.value("level", "f16");
            decision.level = kv_strategy_manager::string_to_level(level_str);
        }
    }

    decision.requires_rebuild = (decision.level != current_level);

    std::ostringstream reason;
    reason << "threshold strategy: ctx_util=" << ctx_util
           << " -> " << kv_strategy_manager::level_to_string(decision.level);
    decision.reason = reason.str();

    decision.metadata = json{
        {"strategy", "threshold"},
        {"ctx_utilization", ctx_util},
        {"hysteresis_applied", hysteresis}
    };

    return decision;
}

// Strategy 3: Adaptive multi-signal
kv_strategy_decision evaluate_adaptive_strategy(
    const kv_strategy_config & config,
    const kv_strategy_metrics & metrics,
    kv_quant_level current_level
) {
    kv_strategy_decision decision;

    // Get weights from params
    double ctx_weight = config.params.value("ctx_weight", 0.4);
    double memory_weight = config.params.value("memory_weight", 0.3);
    double load_weight = config.params.value("load_weight", 0.2);
    double prompt_weight = config.params.value("prompt_weight", 0.1);

    double memory_emergency = config.params.value("memory_emergency_threshold", 0.85);
    double high_load = config.params.value("high_load_threshold", 0.7);
    double medium_load = config.params.value("medium_load_threshold", 0.4);
    double hysteresis = config.params.value("hysteresis", 0.05);

    // Calculate signals
    double ctx_util = metrics.n_ctx_total > 0
        ? (double)metrics.n_ctx_used / metrics.n_ctx_total
        : 0.0;

    double mem_pressure = metrics.kv_cache_max > 0
        ? (double)metrics.kv_cache_size / metrics.kv_cache_max
        : 0.0;

    double slot_load = metrics.n_slots_total > 0
        ? (double)metrics.n_slots_active / metrics.n_slots_total
        : 0.0;

    double prompt_ratio = metrics.n_ctx_total > 0
        ? (double)metrics.avg_prompt_length / metrics.n_ctx_total
        : 0.0;

    // Weighted score
    double score = ctx_weight * ctx_util
                 + memory_weight * mem_pressure
                 + load_weight * slot_load
                 + prompt_weight * prompt_ratio;

    // Emergency memory protection
    if (mem_pressure > memory_emergency) {
        decision.level = kv_quant_level::Q4_0;
        decision.reason = "adaptive: emergency memory protection (mem_pressure="
                        + std::to_string(mem_pressure) + ")";
    }
    // Score-based decision with hysteresis
    else {
        double adjusted_score = score;

        // Apply hysteresis based on current level
        if (current_level == kv_quant_level::Q4_0) {
            adjusted_score -= hysteresis; // Make it harder to upgrade from Q4_0
        } else if (current_level == kv_quant_level::Q8_0) {
            adjusted_score -= hysteresis * 0.5; // Partial hysteresis for Q8_0
        }

        // Decision based on adjusted score
        if (adjusted_score > high_load) {
            decision.level = kv_quant_level::Q4_0;
        } else if (adjusted_score > medium_load) {
            decision.level = kv_quant_level::Q8_0;
        } else {
            decision.level = kv_quant_level::F16;
        }

        std::ostringstream reason;
        reason << "adaptive: score=" << score
               << " (ctx=" << ctx_util
               << ", mem=" << mem_pressure
               << ", load=" << slot_load
               << ") -> " << kv_strategy_manager::level_to_string(decision.level);
        decision.reason = reason.str();
    }

    decision.requires_rebuild = (decision.level != current_level);

    decision.metadata = json{
        {"strategy", "adaptive"},
        {"score", score},
        {"ctx_utilization", ctx_util},
        {"memory_pressure", mem_pressure},
        {"slot_load", slot_load},
        {"prompt_ratio", prompt_ratio},
        {"weights", json{
            {"ctx", ctx_weight},
            {"memory", memory_weight},
            {"load", load_weight},
            {"prompt", prompt_weight}
        }}
    };

    return decision;
}

} // namespace kv_strategies
