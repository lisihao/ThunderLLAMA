#include "thunder-lmcache-control.h"
#include "thunder-lmcache-storage.h"

#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <atomic>

// Global storage instance pointer (set by llama-context)
// Use atomic for thread-safety
static std::atomic<ThunderChunkStorage*> g_storage_instance{nullptr};

// Signal handler for SIGUSR1
static void sigusr1_handler(int sig) {
    (void)sig; // Unused

    ThunderChunkStorage* storage = g_storage_instance.load();
    if (storage != nullptr) {
        fprintf(stderr, "\n[ThunderLMCache] Received SIGUSR1 - triggering safe unmount...\n");
        storage->safe_unmount();
    } else {
        fprintf(stderr, "\n[ThunderLMCache] Received SIGUSR1 but no storage instance registered\n");
    }
}

void thunder_lmcache_register_signals() {
    struct sigaction sa;
    sa.sa_handler = sigusr1_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART; // Restart interrupted system calls

    if (sigaction(SIGUSR1, &sa, nullptr) < 0) {
        fprintf(stderr, "[ThunderLMCache] WARNING: Failed to register SIGUSR1 handler\n");
    } else {
        fprintf(stderr, "[ThunderLMCache] ✓ Signal handler registered: kill -USR1 %d (to safely unmount)\n", getpid());
    }
}

void thunder_lmcache_safe_unmount() {
    ThunderChunkStorage* storage = g_storage_instance.load();
    if (storage != nullptr) {
        storage->safe_unmount();
    } else {
        fprintf(stderr, "[ThunderLMCache] No storage instance to unmount\n");
    }
}

int thunder_lmcache_is_l3_enabled() {
    ThunderChunkStorage* storage = g_storage_instance.load();
    if (storage != nullptr) {
        return storage->is_l3_enabled() ? 1 : 0;
    }
    return 0;
}

// Internal API for llama-context to register storage instance
extern "C" void thunder_lmcache_register_storage(void* storage) {
    g_storage_instance.store(static_cast<ThunderChunkStorage*>(storage));
}

extern "C" void thunder_lmcache_unregister_storage() {
    g_storage_instance.store(nullptr);
}
