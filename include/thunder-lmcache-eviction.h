#pragma once

#include <functional>
#include <vector>
#include <string>
#include <stdint.h>

#ifdef __cplusplus

/**
 * @brief Eviction callback for notifying external systems
 *
 * When LMCache evicts chunks, external systems (like ContextPilot)
 * need to be notified to keep their index synchronized.
 */

using EvictionCallback = std::function<void(const std::vector<uint64_t>&)>;

/**
 * @brief Global eviction callback registry
 *
 * Allows external systems to register callbacks that will be
 * invoked when chunks are evicted from L3 cache.
 */
class ThunderEvictionNotifier {
public:
    /**
     * @brief Get singleton instance
     */
    static ThunderEvictionNotifier& instance();

    /**
     * @brief Register eviction callback
     *
     * @param callback Function to call with evicted chunk hashes
     */
    void register_callback(EvictionCallback callback);

    /**
     * @brief Notify all registered callbacks
     *
     * @param evicted_hashes List of evicted chunk content hashes
     */
    void notify(const std::vector<uint64_t>& evicted_hashes);

    /**
     * @brief Configure HTTP webhook for eviction notification
     *
     * @param url ContextPilot /evict endpoint URL
     * @return true if webhook was configured successfully
     */
    bool configure_webhook(const std::string& url);

    /**
     * @brief Send eviction notification via HTTP
     *
     * @param url Target URL (e.g., http://localhost:8000/evict)
     * @param evicted_hashes List of evicted chunk hashes
     * @return true if notification was sent successfully
     */
    static bool notify_via_http(const std::string& url, const std::vector<uint64_t>& evicted_hashes);

private:
    ThunderEvictionNotifier() = default;
    ~ThunderEvictionNotifier() = default;
    ThunderEvictionNotifier(const ThunderEvictionNotifier&) = delete;
    ThunderEvictionNotifier& operator=(const ThunderEvictionNotifier&) = delete;

    std::vector<EvictionCallback> callbacks_;
    std::string webhook_url_;
};

/**
 * @brief C-style API for eviction notification
 */
extern "C" {
    /**
     * @brief Set ContextPilot eviction webhook URL
     *
     * @param url ContextPilot /evict endpoint (e.g., http://localhost:8000/evict)
     */
    void thunder_lmcache_set_eviction_webhook(const char* url);

    /**
     * @brief Manually trigger eviction notification
     *
     * @param hashes Array of evicted content hashes
     * @param count Number of hashes
     */
    void thunder_lmcache_notify_eviction(const uint64_t* hashes, size_t count);
}

#endif // __cplusplus
