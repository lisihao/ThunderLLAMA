// ThunderLLAMA Cache Management Tool
#include "thunder-lmcache-storage.h"
#include <iostream>
#include <string>
#include <sys/stat.h>

void print_usage() {
    std::cout << R"(
ThunderLLAMA Cache Management Tool

Usage: thunder-cache <command> [options]

Commands:
  stats [path]         Show cache statistics
  compact [path]       Manually trigger compaction
  clear [path]         Clear cache file
  info [path]          Show detailed cache info

Options:
  path                 Path to cache file (default: ~/.cache/thunderllama/kv_cache.bin)
                       Can be overridden by THUNDER_LMCACHE_DISK_PATH

Examples:
  thunder-cache stats
  thunder-cache compact /Volumes/toshiba/thunderllama/kv_cache.bin
  thunder-cache clear
)";
}

std::string get_cache_path(int argc, char** argv, int arg_index) {
    if (argc > arg_index) {
        return argv[arg_index];
    }
    const char* env_path = getenv("THUNDER_LMCACHE_DISK_PATH");
    if (env_path) {
        return env_path;
    }
    const char* home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.cache/thunderllama/kv_cache.bin";
}

int cmd_stats(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        std::cerr << "Cache file not found: " << path << std::endl;
        return 1;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "  CACHE STATISTICS" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Path: " << path << std::endl;
    std::cout << "Size: " << (st.st_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    // Try to load and get detailed stats
    try {
        ThunderChunkStorage storage(
            8ULL * 1024 * 1024 * 1024,
            256ULL * 1024 * 1024 * 1024,
            path
        );

        std::cout << "L2 (Memory) Usage: " << (storage.get_cpu_usage_bytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "L3 (Disk) Usage: " << (storage.get_disk_usage_bytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Hit Rate: " << (storage.get_hit_rate() * 100.0) << "%" << std::endl;
        std::cout << "L3 Enabled: " << (storage.is_l3_enabled() ? "Yes" : "No") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading cache: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int cmd_clear(const std::string& path) {
    if (remove(path.c_str()) == 0) {
        std::cout << "✓ Cache cleared: " << path << std::endl;
    } else {
        std::cerr << "✗ Failed to clear cache (file may not exist)" << std::endl;
        return 1;
    }
    return 0;
}

int cmd_compact(const std::string& path) {
    std::cout << "Compacting cache: " << path << std::endl;
    std::cout << "(Compaction is automatic during normal operation)" << std::endl;
    std::cout << "To force compaction, clear the cache and let it rebuild." << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "stats") {
        return cmd_stats(get_cache_path(argc, argv, 2));
    } else if (cmd == "clear") {
        return cmd_clear(get_cache_path(argc, argv, 2));
    } else if (cmd == "compact") {
        return cmd_compact(get_cache_path(argc, argv, 2));
    } else if (cmd == "help" || cmd == "-h" || cmd == "--help") {
        print_usage();
        return 0;
    } else {
        std::cerr << "Unknown command: " << cmd << std::endl;
        print_usage();
        return 1;
    }
}
