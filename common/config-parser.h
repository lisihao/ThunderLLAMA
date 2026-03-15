// ThunderLLAMA 配置文件解析器
// 从 thunderllama.conf 读取配置并设置环境变量/参数
#pragma once

#include "common.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <map>
#include <string>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace thunderllama {

// 配置文件键值对
struct Config {
    std::map<std::string, std::string> values;
};

// 去除字符串两端空格
static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

// 去除引号
static std::string unquote(const std::string& str) {
    std::string s = trim(str);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// 展开路径中的 ~ 和 $HOME
static std::string expand_path(const std::string& path) {
    std::string expanded = path;

    // 展开 ~
    if (!expanded.empty() && expanded[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            expanded = std::string(home) + expanded.substr(1);
        }
    }

    // 展开 $HOME
    size_t pos = 0;
    while ((pos = expanded.find("$HOME", pos)) != std::string::npos) {
        const char* home = getenv("HOME");
        if (home) {
            expanded.replace(pos, 5, home);
            pos += strlen(home);
        } else {
            pos += 5;
        }
    }

    return expanded;
}

// 从文件加载配置
static Config load_config(const std::string& config_path) {
    Config config;
    std::ifstream file(config_path);

    if (!file.is_open()) {
        fprintf(stderr, "error: failed to load config file: %s\n", config_path.c_str());
        fprintf(stderr, "error: ThunderLLAMA requires thunderllama.conf to run\n");
        exit(1);
    }

    fprintf(stderr, "ThunderLLAMA: loaded config from thunderllama.conf\n");

    std::string line;
    while (std::getline(file, line)) {
        // 去除注释
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        line = trim(line);
        if (line.empty()) continue;

        // 解析 key=value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = unquote(line.substr(eq_pos + 1));

        config.values[key] = value;
    }

    return config;
}

// 获取字符串值
static std::string get_string(const Config& config, const std::string& key, const std::string& default_value = "") {
    auto it = config.values.find(key);
    return (it != config.values.end()) ? it->second : default_value;
}

// 获取整数值
static int get_int(const Config& config, const std::string& key, int default_value = 0) {
    auto it = config.values.find(key);
    if (it != config.values.end()) {
        return std::atoi(it->second.c_str());
    }
    return default_value;
}

// 获取布尔值
static bool get_bool(const Config& config, const std::string& key, bool default_value = false) {
    auto it = config.values.find(key);
    if (it != config.values.end()) {
        return (it->second == "1" || it->second == "true" || it->second == "on");
    }
    return default_value;
}

// 设置环境变量
static void set_env(const std::string& key, const std::string& value) {
    setenv(key.c_str(), value.c_str(), 1);
}

// 从配置文件加载并应用到 common_params
static void thunderllama_config_load_and_apply(common_params& params, const std::string& config_path_override = "") {
    // 查找配置文件
    std::string config_path = config_path_override;

    if (config_path.empty()) {
        // 尝试当前目录
        if (access("./thunderllama.conf", F_OK) == 0) {
            config_path = "./thunderllama.conf";
        }
        // 尝试可执行文件目录
        else {
            char exe_path[4096];
            ssize_t len = -1;

#ifdef __APPLE__
            uint32_t size = sizeof(exe_path);
            if (_NSGetExecutablePath(exe_path, &size) == 0) {
                len = strlen(exe_path);
            }
#else
            len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
#endif

            if (len > 0) {
                exe_path[len] = '\0';
                std::string exe_dir = std::string(exe_path);
                size_t last_slash = exe_dir.find_last_of("/\\");
                if (last_slash != std::string::npos) {
                    exe_dir = exe_dir.substr(0, last_slash);
                    // Go up two levels from build/bin to project root
                    size_t second_slash = exe_dir.find_last_of("/\\");
                    if (second_slash != std::string::npos) {
                        exe_dir = exe_dir.substr(0, second_slash);
                        size_t third_slash = exe_dir.find_last_of("/\\");
                        if (third_slash != std::string::npos) {
                            exe_dir = exe_dir.substr(0, third_slash);
                        }
                    }
                    std::string conf_path = exe_dir + "/thunderllama.conf";
                    if (access(conf_path.c_str(), F_OK) == 0) {
                        config_path = conf_path;
                    }
                }
            }
        }
    }

    if (config_path.empty()) {
        fprintf(stderr, "error: thunderllama.conf not found\n");
        fprintf(stderr, "error: ThunderLLAMA requires thunderllama.conf to run\n");
        exit(1);
    }

    // 加载配置
    Config config = load_config(config_path);

    // 设置环境变量 (ThunderLLAMA 特有)
    set_env("THUNDER_LMCACHE", get_string(config, "THUNDER_LMCACHE", "0"));
    set_env("THUNDER_LMCACHE_DISK_PATH", expand_path(get_string(config, "THUNDER_LMCACHE_DISK_PATH", "")));
    set_env("THUNDER_PREFIX_MATCHING", get_string(config, "THUNDER_PREFIX_MATCHING", "0"));
    set_env("THUNDER_LMCACHE_EXCLUSIVE", get_string(config, "THUNDER_LMCACHE_EXCLUSIVE", "0"));
    set_env("LLAMA_PAGED_ATTENTION", get_string(config, "LLAMA_PAGED_ATTENTION", "0"));

    std::string chunk_prefill_val = get_string(config, "THUNDERLLAMA_CHUNK_PREFILL", "0");
    set_env("THUNDERLLAMA_CHUNK_PREFILL", chunk_prefill_val);
    fprintf(stderr, "[CONFIG] THUNDERLLAMA_CHUNK_PREFILL set to: %s\n", chunk_prefill_val.c_str());
    fflush(stderr);
    
    std::string metal_buffer = get_string(config, "METAL_MAX_BUFFER_SIZE", "");
    if (!metal_buffer.empty()) {
        set_env("GGML_METAL_MAX_BUFFER_SIZE", metal_buffer);
    }

    // Metal Fusion
    if (get_int(config, "METAL_FUSION", 1) == 0) {
        set_env("GGML_METAL_FUSION_DISABLE", "1");
    }

    // 填充 common_params
    std::string model_path = expand_path(get_string(config, "MODEL_PATH", ""));
    if (!model_path.empty()) {
        params.model.path = model_path;
    }

    params.n_ctx = get_int(config, "CONTEXT_SIZE", 4096);
    params.n_gpu_layers = get_int(config, "GPU_LAYERS", 99);
    params.cpuparams.n_threads = get_int(config, "CPU_THREADS", 4);
    params.cpuparams_batch.n_threads = get_int(config, "CPU_THREADS_BATCH", 8);
    params.n_parallel = get_int(config, "PARALLEL_SLOTS", 4);
    params.n_batch = get_int(config, "BATCH_SIZE", 4096);
    params.n_ubatch = get_int(config, "UBATCH_SIZE", 1024);
    params.port = get_int(config, "SERVER_PORT", 30000);

    // Flash Attention
    std::string fa = get_string(config, "FLASH_ATTENTION", "on");
    if (fa == "on" || fa == "1") {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else if (fa == "off" || fa == "0") {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    } else {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }

    // Continuous Batching
    params.cont_batching = get_bool(config, "CONT_BATCHING", true);

    // === Cache Configuration ===
    int cache_reuse = get_int(config, "CACHE_REUSE", 0);
    if (cache_reuse > 0) {
        params.n_cache_reuse = cache_reuse;
    }

    // === CPU Configuration ===
    // CPU mask (e.g., "0xFF" -> first 8 cores)
    std::string cpu_mask = get_string(config, "CPU_MASK", "");
    if (!cpu_mask.empty()) {
        unsigned long mask = std::stoul(cpu_mask, nullptr, 16);
        for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
            params.cpuparams.cpumask[i] = (mask & (1UL << i)) != 0;
        }
        params.cpuparams.mask_valid = true;
    }

    int process_priority = get_int(config, "PROCESS_PRIORITY", -1);
    if (process_priority >= 0 && process_priority <= 3) {
        params.cpuparams.priority = static_cast<ggml_sched_priority>(process_priority);
    }

    // === KV Cache ===
    std::string kv_level = get_string(config, "KV_CACHE_LEVEL", "");
    if (!kv_level.empty()) {
        ggml_type kv_type = GGML_TYPE_F16; // default
        if (kv_level == "f16")       kv_type = GGML_TYPE_F16;
        else if (kv_level == "q8_0") kv_type = GGML_TYPE_Q8_0;
        else if (kv_level == "q8_0_metal") {
            // GPU-side dynamic quantization (ThunderLLAMA)
            kv_type = GGML_TYPE_F16;  // Storage uses FP16, quantize on-the-fly
        }
        else if (kv_level == "q4_0") kv_type = GGML_TYPE_Q4_0;
        else if (kv_level == "q4_1") kv_type = GGML_TYPE_Q4_1;
        else if (kv_level == "bf16") kv_type = GGML_TYPE_BF16;
        params.cache_type_k = kv_type;
        params.cache_type_v = kv_type;
    }

    bool kv_unified = get_bool(config, "KV_UNIFIED", false);
    if (kv_unified) {
        params.kv_unified = true;
    }

    // === Metal/GPU Optimization ===
    bool no_host = get_bool(config, "NO_HOST", false);
    if (no_host) {
        params.no_host = true;
    }

    // === LMCache Capacity (Task 9) ===
    // LMCACHE_L2_SIZE_GB (default 8)
    std::string l2_size = get_string(config, "LMCACHE_L2_SIZE_GB", "");
    if (!l2_size.empty()) {
        set_env("LMCACHE_L2_SIZE_GB", l2_size);
    }

    // LMCACHE_L3_SIZE_GB (default 256)
    std::string l3_size = get_string(config, "LMCACHE_L3_SIZE_GB", "");
    if (!l3_size.empty()) {
        set_env("LMCACHE_L3_SIZE_GB", l3_size);
    }

    // LMCACHE_FREQ_PROTECT threshold (default 5)
    std::string freq_protect = get_string(config, "LMCACHE_FREQ_PROTECT", "");
    if (!freq_protect.empty()) {
        set_env("LMCACHE_FREQ_PROTECT", freq_protect);
    }

    // LMCACHE_TTL_HOURS (default 0 = infinite)
    std::string ttl_hours = get_string(config, "LMCACHE_TTL_HOURS", "");
    if (!ttl_hours.empty()) {
        set_env("LMCACHE_TTL_HOURS", ttl_hours);
    }

    // THUNDER_LMCACHE_EXCLUSIVE
    set_env("THUNDER_LMCACHE_EXCLUSIVE", get_string(config, "THUNDER_LMCACHE_EXCLUSIVE", "0"));

    // FUSED_QKV (QKV Projection Fusion)
    // Default: disabled (0). Set to 1 to enable.
    if (get_bool(config, "FUSED_QKV", false)) {
        set_env("FUSED_QKV", "1");
    }

    fprintf(stderr, "ThunderLLAMA: config applied\n");
}

} // namespace thunderllama
