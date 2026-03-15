// ThunderLLAMA Configuration File Parser
// Version: 1.0
// Purpose: Parse thunderllama.conf and populate common_params

#pragma once

#include "common.h"
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>

// Default config file path
#define THUNDERLLAMA_DEFAULT_CONFIG "thunderllama.conf"

// Config file parser
class ThunderLLAMAConfig {
public:
    std::map<std::string, std::string> config;
    bool loaded = false;

    // Load config file
    bool load(const std::string & config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // Remove inline comments
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            // Parse key=value
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string value = trim(line.substr(eq_pos + 1));

                // Remove quotes
                if (!value.empty() && value[0] == '"' && value[value.length() - 1] == '"') {
                    value = value.substr(1, value.length() - 2);
                }

                // Expand $HOME and $VARIABLE
                value = expand_env(value);

                config[key] = value;
            }
        }

        file.close();
        loaded = true;
        return true;
    }

    // Get string value
    std::string get(const std::string & key, const std::string & default_value = "") const {
        auto it = config.find(key);
        return (it != config.end()) ? it->second : default_value;
    }

    // Get int value
    int get_int(const std::string & key, int default_value = 0) const {
        auto it = config.find(key);
        if (it == config.end()) return default_value;
        return std::stoi(it->second);
    }

    // Get bool value (1/0, true/false, on/off)
    bool get_bool(const std::string & key, bool default_value = false) const {
        auto it = config.find(key);
        if (it == config.end()) return default_value;

        std::string value = to_lower(it->second);
        return (value == "1" || value == "true" || value == "on");
    }

    // Get float value
    float get_float(const std::string & key, float default_value = 0.0f) const {
        auto it = config.find(key);
        if (it == config.end()) return default_value;
        return std::stof(it->second);
    }

private:
    // Trim whitespace
    static std::string trim(const std::string & str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }

    // Convert to lowercase
    static std::string to_lower(const std::string & str) {
        std::string result = str;
        for (char & c : result) {
            c = std::tolower(c);
        }
        return result;
    }

    // Expand environment variables
    static std::string expand_env(const std::string & str) {
        std::string result = str;

        // Expand $HOME
        size_t pos = 0;
        while ((pos = result.find("$HOME", pos)) != std::string::npos) {
            const char * home = std::getenv("HOME");
            if (home) {
                result.replace(pos, 5, home);
                pos += strlen(home);
            } else {
                pos += 5;
            }
        }

        // Expand ~
        if (!result.empty() && result[0] == '~') {
            const char * home = std::getenv("HOME");
            if (home) {
                result = std::string(home) + result.substr(1);
            }
        }

        return result;
    }
};

// Apply config to common_params
inline void thunderllama_config_apply(common_params & params, const ThunderLLAMAConfig & config) {
    if (!config.loaded) return;

    // === Model Configuration ===
    std::string model_path = config.get("MODEL_PATH");
    if (!model_path.empty()) {
        params.model.path = model_path;
    }

    int context_size = config.get_int("CONTEXT_SIZE", -1);
    if (context_size > 0) {
        params.n_ctx = context_size;
    }

    // === Server Configuration ===
    int port = config.get_int("SERVER_PORT", -1);
    if (port > 0) {
        params.port = port;
    }

    int gpu_layers = config.get_int("GPU_LAYERS", -1);
    if (gpu_layers >= 0) {
        params.n_gpu_layers = gpu_layers;
    }

    int parallel_slots = config.get_int("PARALLEL_SLOTS", -1);
    if (parallel_slots > 0) {
        params.n_parallel = parallel_slots;
    }

    // === Performance Optimization ===
    std::string flash_attention = config.get("FLASH_ATTENTION");
    if (!flash_attention.empty()) {
        if (flash_attention == "on") {
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (flash_attention == "off") {
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        // "auto" keeps default (LLAMA_FLASH_ATTN_TYPE_AUTO)
    }

    int batch_size = config.get_int("BATCH_SIZE", -1);
    if (batch_size > 0) {
        params.n_batch = batch_size;
    }

    int ubatch_size = config.get_int("UBATCH_SIZE", -1);
    if (ubatch_size > 0) {
        params.n_ubatch = ubatch_size;
    }

    // === Cache Configuration ===
    int cache_reuse = config.get_int("CACHE_REUSE", -1);
    if (cache_reuse > 0) {
        params.n_cache_reuse = cache_reuse;
    }

    // === Continuous Batching ===
    bool cont_batching = config.get_bool("CONT_BATCHING", false);
    if (cont_batching) {
        params.cont_batching = true;
    }

    // === CPU Configuration ===
    int cpu_threads = config.get_int("CPU_THREADS", -1);
    if (cpu_threads > 0) {
        params.cpuparams.n_threads = cpu_threads;
    }

    int cpu_threads_batch = config.get_int("CPU_THREADS_BATCH", -1);
    if (cpu_threads_batch > 0) {
        params.cpuparams_batch.n_threads = cpu_threads_batch;
    }

    // CPU mask (e.g., "0xFF" -> first 8 cores)
    std::string cpu_mask = config.get("CPU_MASK");
    if (!cpu_mask.empty()) {
        unsigned long mask = std::stoul(cpu_mask, nullptr, 16);
        for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
            params.cpuparams.cpumask[i] = (mask & (1UL << i)) != 0;
        }
        params.cpuparams.mask_valid = true;
    }

    int process_priority = config.get_int("PROCESS_PRIORITY", -1);
    if (process_priority >= 0 && process_priority <= 3) {
        params.cpuparams.priority = static_cast<ggml_sched_priority>(process_priority);
    }

    // === KV Cache ===
    std::string kv_level = config.get("KV_CACHE_LEVEL");
    if (!kv_level.empty()) {
        ggml_type kv_type = GGML_TYPE_F16; // default
        if (kv_level == "f16")       kv_type = GGML_TYPE_F16;
        else if (kv_level == "q8_0") kv_type = GGML_TYPE_Q8_0;
        else if (kv_level == "q8_0_metal") {
            // GPU-side dynamic quantization (ThunderLLAMA)
            kv_type = GGML_TYPE_F16;  // Storage uses FP16, quantize on-the-fly
            // TODO: Set a flag to enable Metal quantization
        }
        else if (kv_level == "q4_0") kv_type = GGML_TYPE_Q4_0;
        else if (kv_level == "q4_1") kv_type = GGML_TYPE_Q4_1;
        else if (kv_level == "bf16") kv_type = GGML_TYPE_BF16;
        params.cache_type_k = kv_type;
        params.cache_type_v = kv_type;
    }

    bool kv_unified = config.get_bool("KV_UNIFIED", false);
    if (kv_unified) {
        params.kv_unified = true;
    }

    // === Metal/GPU Optimization ===
    bool no_host = config.get_bool("NO_HOST", false);
    if (no_host) {
        params.no_host = true;
    }

    // === Logging ===
    std::string log_file = config.get("LOG_FILE");
    if (!log_file.empty()) {
        // Note: llama.cpp doesn't have built-in log file redirection in params
        // This would need to be handled separately in main()
    }

    // === ThunderLLAMA Environment Variables ===
    // These are handled via setenv() in main() before llama initialization
    // We store them for later use
}

// Load config file and apply to params
// Returns true if config was loaded (even if file doesn't exist - that's OK)
inline bool thunderllama_config_load_and_apply(common_params & params, const std::string & config_path = THUNDERLLAMA_DEFAULT_CONFIG) {
    ThunderLLAMAConfig config;

    // Try to load config file
    if (!config.load(config_path)) {
        // Config file doesn't exist or can't be read - that's OK, use defaults/CLI args
        return true;
    }

    fprintf(stderr, "ThunderLLAMA: loaded config from %s\n", config_path.c_str());

    // === Set Environment Variables (for ThunderLLAMA features) ===
    // These must be set BEFORE llama_backend_init()

    // THUNDER_LMCACHE
    if (config.get_bool("THUNDER_LMCACHE", false)) {
        setenv("THUNDER_LMCACHE", "1", 1);
    }

    // THUNDER_LMCACHE_DISK_PATH
    std::string disk_path = config.get("THUNDER_LMCACHE_DISK_PATH");
    if (!disk_path.empty()) {
        setenv("THUNDER_LMCACHE_DISK_PATH", disk_path.c_str(), 1);
    }

    // THUNDER_PREFIX_MATCHING
    if (config.get_bool("THUNDER_PREFIX_MATCHING", false)) {
        setenv("THUNDER_PREFIX_MATCHING", "1", 1);
    }

    // LLAMA_PAGED_ATTENTION
    if (config.get_bool("LLAMA_PAGED_ATTENTION", false)) {
        setenv("LLAMA_PAGED_ATTENTION", "1", 1);
    }

    // THUNDERLLAMA_CHUNK_PREFILL
    if (config.get_bool("THUNDERLLAMA_CHUNK_PREFILL", false)) {
        setenv("THUNDERLLAMA_CHUNK_PREFILL", "1", 1);
    }

    // GGML_METAL_MAX_BUFFER_SIZE
    std::string metal_buffer = config.get("METAL_MAX_BUFFER_SIZE");
    if (!metal_buffer.empty()) {
        setenv("GGML_METAL_MAX_BUFFER_SIZE", metal_buffer.c_str(), 1);
    }

    // METAL_FUSION (Metal kernel fusion: ADD fusion + MoE gating fusion)
    // Default: enabled (1). Set to 0 to disable for debugging.
    if (!config.get_bool("METAL_FUSION", true)) {
        setenv("GGML_METAL_FUSION_DISABLE", "1", 1);
    }

    // FUSED_QKV (QKV Projection Fusion - experimental)
    // Default: disabled (0). Set to 1 to enable.
    if (config.get_bool("FUSED_QKV", false)) {
        setenv("FUSED_QKV", "1", 1);
    }

    // === Apply params ===
    thunderllama_config_apply(params, config);

    return true;
}
