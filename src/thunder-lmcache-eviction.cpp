#include "thunder-lmcache-eviction.h"

#include <iostream>
#include <sstream>
#include <mutex>
#include <curl/curl.h> // For HTTP POST (需要 libcurl)

// ============================================================================
// ThunderEvictionNotifier Implementation
// ============================================================================

ThunderEvictionNotifier& ThunderEvictionNotifier::instance() {
    static ThunderEvictionNotifier instance;
    return instance;
}

void ThunderEvictionNotifier::register_callback(EvictionCallback callback) {
    callbacks_.push_back(callback);
    fprintf(stderr, "[ThunderEvictionNotifier] Registered eviction callback\n");
}

void ThunderEvictionNotifier::notify(const std::vector<uint64_t>& evicted_hashes) {
    if (evicted_hashes.empty()) {
        return;
    }

    // Call all registered callbacks
    for (auto& callback : callbacks_) {
        try {
            callback(evicted_hashes);
        } catch (const std::exception& e) {
            fprintf(stderr, "[ThunderEvictionNotifier] Callback error: %s\n", e.what());
        }
    }

    // Send HTTP webhook if configured
    if (!webhook_url_.empty()) {
        notify_via_http(webhook_url_, evicted_hashes);
    }
}

bool ThunderEvictionNotifier::configure_webhook(const std::string& url) {
    webhook_url_ = url;
    fprintf(stderr, "[ThunderEvictionNotifier] Configured webhook: %s\n", url.c_str());
    return true;
}

bool ThunderEvictionNotifier::notify_via_http(
    const std::string& url,
    const std::vector<uint64_t>& evicted_hashes
) {
#ifdef USE_LIBCURL
    // libcurl 实现 (如果启用了 libcurl)
    CURL* curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "[ThunderEvictionNotifier] Failed to initialize curl\n");
        return false;
    }

    // Build JSON payload
    std::ostringstream json;
    json << "{\"evicted_hashes\":[";
    for (size_t i = 0; i < evicted_hashes.size(); ++i) {
        if (i > 0) json << ",";
        json << evicted_hashes[i];
    }
    json << "]}";

    std::string payload = json.str();

    // Set curl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Disable verbose output
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

    // Timeout
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    // Perform request
    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "[ThunderEvictionNotifier] HTTP POST failed: %s\n",
                curl_easy_strerror(res));
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return res == CURLE_OK;
#else
    // 简化实现: 使用 system() 调用 curl 命令
    // (不推荐生产环境，但可以快速测试)

    std::ostringstream cmd;
    cmd << "curl -s -X POST " << url
        << " -H 'Content-Type: application/json' -d '{\"evicted_hashes\":[";

    for (size_t i = 0; i < evicted_hashes.size(); ++i) {
        if (i > 0) cmd << ",";
        cmd << evicted_hashes[i];
    }
    cmd << "]}' > /dev/null 2>&1 &";

    // 后台执行，不阻塞主线程
    int ret = system(cmd.str().c_str());

    if (ret != 0) {
        fprintf(stderr, "[ThunderEvictionNotifier] curl command failed\n");
        return false;
    }

    return true;
#endif
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

void thunder_lmcache_set_eviction_webhook(const char* url) {
    if (!url) return;
    ThunderEvictionNotifier::instance().configure_webhook(url);
}

void thunder_lmcache_notify_eviction(const uint64_t* hashes, size_t count) {
    if (!hashes || count == 0) return;

    std::vector<uint64_t> evicted(hashes, hashes + count);
    ThunderEvictionNotifier::instance().notify(evicted);
}

} // extern "C"
