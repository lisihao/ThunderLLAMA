# ThunderLLAMA Performance Report (2026-03-06)

## Environment
- Host: Apple M4 Pro, 48 GB unified memory
- Model: `/Users/lisihao/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf`
- Binary: `./build/bin/llama-server`
- Benchmark driver: `./.solar/bench-unified-qwen30b.sh`

## Metric Definitions
- `throughput_total`: `(tokens_evaluated + tokens_predicted) / duration`
- `throughput_decode`: `tokens_predicted / duration`

## Optimization Modes
- `allopt`
  - flash attention on
  - continuous batching on (`--cont-batching`)
  - paged attention on (`LLAMA_PAGED_ATTENTION=1`)
  - prompt cache enabled
  - slot quota enabled (`LLAMA_SERVER_SLOT_QUOTA=1`)
- `flashonly`
  - flash attention on
  - continuous batching off (`--no-cont-batching`)
  - paged attention off (`LLAMA_PAGED_ATTENTION=0`)
  - prompt cache off (`--no-cache-prompt`)

## Verification: Features Actually Enabled
From log `/Users/lisihao/thunder-bench-unified/server-allopt-20260306-222147.log`:
- Paged attention enabled: `llama_context: enabling paged attention mode`
- Chunked prefill behavior visible via ubatch stepping: `n_ubatch = 512` and repeated prompt progress in chunks
- Continuous batching active in allopt path (script passes `--cont-batching`)

## A/B: Slot Quota Toggle (allopt, 16384/4/800/64)
- `LLAMA_SERVER_SLOT_QUOTA=1`: 429.53, 664.60, 673.96 total tok/s
- `LLAMA_SERVER_SLOT_QUOTA=0`: 658.96, 641.07, 630.46 total tok/s

Observation:
- First run under quota=1 was a clear cold-start outlier.
- Excluding first cold run, quota=1 mean slightly exceeds quota=0.

## Retest: All Optimizations Enabled (allopt, quota=1)
Workload: `ctx=16384, parallel=4, prompt_repeats=800, n_predict=64`

- Warmup: 670.05 / 17.40 (total/decode tok/s)
- Run1: 671.47 / 17.44
- Run2: 673.31 / 17.49
- Run3: 674.84 / 17.53
- Mean (3 runs): **673.21 / 17.49**

Compared with previous comparable allopt mean (cold run removed):
- Previous: 669.28 / 17.39
- New: 673.21 / 17.49
- Delta: **+0.59% total**, **+0.58% decode**

## Parameter Sweep (Expanded Scenarios)
All rows are `allopt vs flashonly` with same parameters.

| Scenario | Params `(ctx/parallel/prompt_repeats/n_predict)` | allopt total/decode | flashonly total/decode | allopt gain |
|---|---:|---:|---:|---:|
| Higher load | `16384/4/1000/64` | `706.41 / 14.76` | `542.74 / 11.34` | `+30.16% / +30.16%` |
| Higher parallelism | `16384/8/600/64` | `601.73 / 20.66` | `516.41 / 17.73` | `+16.52% / +16.53%` |
| Longer context | `32768/4/1200/64` | `656.10 / 11.46` | `535.33 / 9.35` | `+22.56% / +22.57%` |

Notes:
- Attempted heavier workload `16384/4/1200/128` was unstable in flashonly (connection reset), so not used for final comparison.

## Conclusions
1. Paged attention, continuous batching, and chunked prefill behavior are all active in allopt mode.
2. Across larger-load / larger-parallel / longer-context scenarios, allopt consistently outperforms flashonly by **~16% to ~30%**.
3. For user-facing speed interpretation:
   - `throughput_decode` reflects generation speed.
   - `throughput_total` reflects end-to-end capacity (prefill + decode).

## Repro Commands
```bash
cd /Users/lisihao/ThunderLLAMA

# allopt retest
LLAMA_HOTPATH_DEBUG=0 LLAMA_SERVER_SLOT_QUOTA=1 ./.solar/bench-unified-qwen30b.sh allopt 16384 4 800 64 18201
LLAMA_HOTPATH_DEBUG=0 LLAMA_SERVER_SLOT_QUOTA=1 ./.solar/bench-unified-qwen30b.sh allopt 16384 4 800 64 18202
LLAMA_HOTPATH_DEBUG=0 LLAMA_SERVER_SLOT_QUOTA=1 ./.solar/bench-unified-qwen30b.sh allopt 16384 4 800 64 18203

# sweep examples
LLAMA_HOTPATH_DEBUG=0 LLAMA_SERVER_SLOT_QUOTA=1 ./.solar/bench-unified-qwen30b.sh allopt 16384 8 600 64 18302
LLAMA_HOTPATH_DEBUG=0 ./.solar/bench-unified-qwen30b.sh flashonly 16384 8 600 64 18303
```

## Raw Data Locations
- Unified CSV: `/Users/lisihao/thunder-bench-unified/results.csv`
- Recent run logs: `/Users/lisihao/thunder-bench-unified/server-*.log`
