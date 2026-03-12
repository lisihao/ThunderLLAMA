#!/bin/bash
# ThunderLLAMA Cache Configuration
# Source this file to configure cache storage location

# Detect available external storage
if [ -d "/Volumes/toshiba" ]; then
    echo "✓ Found external storage: /Volumes/toshiba"
    export THUNDER_LMCACHE_DISK_PATH="/Volumes/toshiba/thunderllama/kv_cache.bin"
    echo "→ Using toshiba for cache storage (256GB)"
elif [ -d "/Volumes" ]; then
    # List other volumes
    OTHER_VOLUMES=$(ls -1 /Volumes | grep -v "Macintosh HD" | head -1)
    if [ -n "$OTHER_VOLUMES" ]; then
        echo "✓ Found external storage: /Volumes/$OTHER_VOLUMES"
        export THUNDER_LMCACHE_DISK_PATH="/Volumes/$OTHER_VOLUMES/thunderllama/kv_cache.bin"
        echo "→ Using $OTHER_VOLUMES for cache storage"
    else
        echo "⚠ No external storage found, using default location"
        export THUNDER_LMCACHE_DISK_PATH="$HOME/.cache/thunderllama/kv_cache.bin"
    fi
else
    echo "→ Using default cache location: ~/.cache/thunderllama/"
    export THUNDER_LMCACHE_DISK_PATH="$HOME/.cache/thunderllama/kv_cache.bin"
fi

echo ""
echo "Cache configuration:"
echo "  Path: $THUNDER_LMCACHE_DISK_PATH"
echo "  L2 (Memory): 8GB"
echo "  L3 (Disk): 256GB"
echo ""
echo "To enable cache: export THUNDER_LMCACHE=1"
