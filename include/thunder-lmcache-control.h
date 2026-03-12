#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register signal handlers for safe cache unmount.
 *
 * Sets up SIGUSR1 handler to safely unmount disk cache before external storage removal.
 * Call this once at program startup.
 *
 * Usage:
 *   1. In your program: thunder_lmcache_register_signals();
 *   2. Before ejecting disk: kill -USR1 <pid>
 *   3. Wait for "Safe unmount complete" message
 *   4. Eject the disk
 *
 * @note Thread-safe: signal handler uses internal locking.
 * @note After unmount, L2 (memory) cache continues to work.
 */
void thunder_lmcache_register_signals();

/**
 * @brief Manually trigger safe unmount (alternative to signal).
 *
 * Stops accepting new writes to L3, syncs all data, and closes disk file.
 * Safe to call before ejecting external drives.
 *
 * @note Thread-safe.
 * @note After unmount, L2 (memory) cache continues to work.
 */
void thunder_lmcache_safe_unmount();

/**
 * @brief Check if L3 disk cache is currently enabled.
 *
 * @return  1 if L3 is operational, 0 if disabled (memory-only mode).
 */
int thunder_lmcache_is_l3_enabled();

#ifdef __cplusplus
}
#endif
