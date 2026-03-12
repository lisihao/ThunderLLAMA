# ThunderLLAMA Service Layer

`thunder_service.py` is a service wrapper around `llama.cpp` (`llama-server`).

It keeps inference in the core engine, and moves request-time optimization to a clear server layer.

## What It Adds

- OpenAI-compatible proxy endpoint: `/v1/chat/completions`
- Tiered context optimization (built-in):
  - `must-have`
  - `nice-to-have`
  - `history-tail`
- Per-tier token caps
- Lightweight history summarization
- Prompt hotspot tracking (`/thunder/hot-prompts`)
- Request metrics persistence into SQLite (for optimization analytics)
- Auto cache-ram policy tuner (24h throughput/latency/failure based, `2G/4G/6G/8G`)
- Tiered prompt cache:
  - hot cache in memory
  - warm cache on disk
  - TTL eviction and deletion
- Dynamic `max_tokens` policy by task type (chat/summary/analysis/code/creative/translation)
- Startup warmup and cache priming (using fixed warmup requests + hotspot templates)
- Cache residency loop (periodic re-prime of top hotspot templates to keep cache hot)

## Endpoints

- `GET /healthz`
- `GET /thunder/features`
- `GET /thunder/hot-prompts`
- `GET /thunder/core/status`
- `GET /thunder/cache-policy/status`
- `GET /thunder/prompt-cache/status`
- `GET /thunder/warmup/status`
- `POST /thunder/core/switch-model`
- `POST /v1/chat/completions`

Other API paths are forwarded to upstream core.

## Usage

## 1) Use existing external core (llama-server already running)

```bash
python3 tools/thunder-service/thunder_service.py \
  --listen-host 127.0.0.1 \
  --listen-port 18081 \
  --upstream http://127.0.0.1:18082/v1
```

## 2) Let service auto-start core

```bash
python3 tools/thunder-service/thunder_service.py \
  --auto-start-core \
  --core-binary ./build/bin/llama-server \
  --core-model /path/to/model.gguf \
  --core-arg --ctx-size=32768 \
  --core-arg --threads=10
```

Then call service endpoint:

`http://127.0.0.1:18081/v1/chat/completions`

## Runtime model switch

Only available when service is started with `--auto-start-core`.

Check core status:

```bash
curl -s http://127.0.0.1:18081/thunder/core/status
```

Switch model:

```bash
curl -s -X POST http://127.0.0.1:18081/thunder/core/switch-model \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/path/to/new-model.gguf",
    "core_args": ["--ctx-size=32768", "--threads=10"]
  }'
```

## Key tuning flags

- `--must-have-cap` (default `1536`)
- `--nice-to-have-cap` (default `768`)
- `--history-tail-cap` (default `512`)
- `--history-recent-cap` (default `320`)
- `--history-archive-cap` (default `192`)
- `--preserve-last-turns` (default `6`)
- `--disable-dynamic-context-postfix`
- `--shared-prefix-file` (default `.solar/prompt-cache/openclaw-shared-prefix.txt`)
- `--disable-layering`
- `--disable-hotspot-tracking`
- `--metrics-db-path` (default `.solar/metrics/thunder_service.db`)
- `--disable-metrics-db`
- `--disable-auto-cache-ram`
- `--auto-cache-interval-sec` (default `300`)
- `--auto-cache-lookback-sec` (default `86400`)
- `--auto-cache-min-samples` (default `20`)
- `--auto-cache-candidates-mb` (default `2048,4096,6144,8192`)
- `--auto-cache-cooldown-sec` (default `1800`)
- `--auto-cache-min-improve` (default `0.03`)
- `--disable-tiered-prompt-cache`
- `--prompt-cache-warm-dir` (default `.solar/prompt-cache/warm`)
- `--prompt-cache-hot-max-entries` (default `256`)
- `--prompt-cache-hot-ttl-sec` (default `3600`)
- `--prompt-cache-warm-ttl-sec` (default `86400`)
- `--prompt-cache-prune-interval-sec` (default `300`)
- `--prompt-cache-hot-hit-threshold` (default `3`)
- `--disable-dynamic-max-tokens`
- `--dynamic-max-tokens-min` (default `64`)
- `--max-tokens-chat` (default `192`)
- `--max-tokens-summary` (default `192`)
- `--max-tokens-analysis` (default `384`)
- `--max-tokens-code` (default `512`)
- `--max-tokens-creative` (default `768`)
- `--max-tokens-translation` (default `256`)
- `--disable-startup-warmup`
- `--warmup-requests` (default `2`)
- `--warmup-max-tokens` (default `32`)
- `--disable-cache-priming`
- `--cache-prime-top-n` (default `10`)
- `--disable-cache-residency`
- `--cache-residency-interval-sec` (default `300`)
- `--cache-residency-top-n` (default `10`)
- `--disable-cache-residency-force-hot`

## Metrics DB schema

The service stores one row per chat request in table `request_metrics`, including:

- message counts (input/output)
- token estimates (input + layered output)
- layer caps and per-layer token usage
- upstream status + latency
- core cache-ram during request (`cache_ram_mb`)
- error and extra metadata JSON

## Core watchdog (auto-restart)

When running with `--auto-start-core`, service can watchdog the core process:

- process exit detection
- zombie process detection (`ps stat` contains `Z`)
- health probe failure detection (`GET /v1/models`)
- automatic kill + restart

Flags:

- `--disable-core-watchdog`
- `--core-watchdog-interval` (default `10` seconds)
- `--core-watchdog-timeout` (default `3` seconds)
- `--core-watchdog-failures` (default `3`)

## Notes

- `stream=true` is currently not supported in this service layer and returns `501`.
- Hotspot stats are flushed to `.solar/prompt-cache/hot_prompts_live.json` by default.
- If you launch with `.solar/run_thunder_service.sh`, you can enable parallel-tier profiles
  (`ENABLE_PARALLEL_TIER_PROFILE=1`) and select `PARALLEL_TIER=1|2|4|auto` to apply
  different context/output/cache presets per concurrency tier.
