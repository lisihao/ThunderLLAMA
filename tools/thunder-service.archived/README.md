# Thunder Service - Archived

> **Status**: ⚠️ ARCHIVED (2026-03-11)
> **Reason**: Core features migrated to [ClawGate](https://github.com/lisihao/ClawGate)
> **Original Documentation**: See [README.old.md](README.old.md)

---

## Why Archived?

Thunder Service was originally developed as a standalone service layer for ThunderLLAMA, implementing 4 production-grade optimizations:

1. **Context Shift** — Two-stage LLM summarization (0.6B extraction + 1.7B compression)
2. **Three-tier Layering** — Four-layer context organization (Must/Nice/History/Tail)
3. **Auto Cache-RAM Tuning** — Heuristic optimizer for data-driven cache-ram sizing
4. **Prompt Reuse** — Two-tier caching (hot in-memory + warm disk)

These features have been **successfully migrated to ClawGate** as part of the **Feature Overlay** strategy, enabling ClawGate to become a comprehensive hybrid inference gateway with both cloud routing and local optimization capabilities.

---

## Migration Summary

| Feature | Thunder Service | ClawGate | Status |
|---------|----------------|----------|--------|
| **Context Shift** | `context_shift_summarizer.py` (246 lines) | `clawgate/context/context_shift_client.py` (381 lines) | ✅ Migrated + Enhanced |
| **Three-tier Layering** | `thunder_service.py` L281-435 | `clawgate/context/strategies/layering.py` (334 lines) | ✅ Migrated |
| **Auto Cache-RAM Tuning** | `thunder_service.py` L779-931 | `clawgate/tuning/cache_tuner.py` (308 lines) | ✅ Migrated |
| **Prompt Reuse** | `thunder_service.py` L934-1033+ | `clawgate/context/prompt_cache.py` (340 lines) | ✅ Migrated |

### Key Enhancements in ClawGate

- **Async Architecture**: `httpx.AsyncClient` for non-blocking Context Shift calls
- **Circuit Breaker**: Automatic fallback when Context Shift services unavailable
- **Feature Flags**: Easy enable/disable via `config/models.yaml`
- **Monitoring**: Dashboard endpoint `/dashboard/cache` for real-time stats
- **Testing**: 201 tests including 5 Phase 2 E2E tests (all passing)

---

## ClawGate Reference

### Documentation
- **Main README**: `/Users/lisihao/ClawGate/README.md`
- **Phase 2 Features**: `/Users/lisihao/ClawGate/docs/PHASE2_FEATURES.md`
- **Context Shift Integration**: `/Users/lisihao/ClawGate/docs/CONTEXT_SHIFT_INTEGRATION.md`

### Implementation Files
```
ClawGate/
├── clawgate/
│   ├── context/
│   │   ├── context_shift_client.py          # Context Shift (httpx async)
│   │   ├── prompt_cache.py                  # Prompt Reuse (hot + warm)
│   │   └── strategies/
│   │       └── layering.py                  # Three-tier Layering
│   └── tuning/
│       └── cache_tuner.py                   # Auto Cache-RAM Tuning
├── scripts/
│   └── start_context_shift_services.sh      # Context Shift service launcher
└── tests/
    ├── test_context_shift_integration.py    # Context Shift E2E tests
    ├── test_cache_tuner.py                  # Cache Tuner unit tests
    ├── test_prompt_cache.py                 # Prompt Cache unit tests
    └── test_phase2_e2e.py                   # Full integration tests
```

### Configuration
```yaml
# ClawGate config/models.yaml

# Prompt Cache
prompt_cache:
  enabled: true
  hot_cache_size: 256
  hot_ttl_sec: 3600
  warm_ttl_sec: 86400

# Cache Tuning
thunderllama:
  cache_tuning:
    enabled: true
    tuner_type: heuristic
    heuristic:
      candidates_mb: [2048, 4096, 6144, 8192]
      lookback_sec: 86400
      cooling_period: 300

# Context Shift
context_shift:
  enabled: true
  mode: auto
  endpoints:
    stage1: "http://127.0.0.1:18083/completion"
    stage2: "http://127.0.0.1:18084/completion"
```

---

## Performance Data

### Prompt Cache (ClawGate)
- **Cache Hit Speedup**: 25,000x (100ms → 0.004ms)
- **Hit Rate**: Varies by workload (tested up to 71% in unit tests)
- **Storage**: Hot (256 entries, 1h TTL) + Warm (disk, 24h TTL)

### Auto Cache-RAM Tuning (ClawGate)
- **Scoring Algorithm**: 50% throughput + 35% (1-latency) + 15% (1-failure)
- **Candidates**: [2048, 4096, 6144, 8192] MB (configurable)
- **Tuning Interval**: Every 5 minutes
- **Cooling Period**: 5 minutes (prevents frequent restarts)

### Context Shift (ClawGate)
- **Fast Mode** (0.6B + 0.6B): ~1.5-2.5s, good quality
- **Quality Mode** (0.6B + 1.7B): ~2-2.7s, excellent quality
- **Auto Mode**: < 10 turns → fast, >= 10 turns → quality
- **Fallback**: Automatic degradation to simple compression on failure

---

## Why Not Just Keep Thunder Service?

**Two Independent Projects**:
- **ThunderLLAMA**: Local inference engine (llama.cpp + Apple Silicon optimizations)
- **ClawGate**: Hybrid gateway (cloud routing + local optimization + agent-aware scheduling)

**Feature Overlay Strategy**:
- Thunder Service features are now **first-class ClawGate features**
- ThunderLLAMA focuses on inference performance (Paged Attention, Flash Attention, Decode-First)
- ClawGate focuses on service orchestration (routing, caching, context management, scheduling)

**Maintainability**:
- Avoid duplicated code across two projects
- Single source of truth for production optimizations
- ClawGate benefits from both cloud scale and local optimizations

---

## Using the Archived Code

If you need to reference the original Thunder Service implementation:

1. **Original Documentation**: [README.old.md](README.old.md)
2. **Source Code**:
   - `thunder_service.py` (79 KB) — Full service implementation
   - `context_shift_summarizer.py` (7.2 KB) — Context Shift module

3. **Deployment Scripts**: Now in ClawGate
   - Context Shift: `ClawGate/scripts/start_context_shift_services.sh`

---

## Migration Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| **Phase 0**: Technical Evaluation | 1 week | ✅ Completed (2026-03-11) |
| **Phase 1**: Context Shift + Layering | 2 weeks | ✅ Completed (2026-03-11) |
| **Phase 2**: Cache Tuning + Prompt Cache | 2 weeks | ✅ Completed (2026-03-11) |
| **Phase 3**: Verification + Archive | 1 week | 🚧 In Progress |

**Total Work**: 6 weeks, 4 core features, 201 tests passing

---

## Questions?

For questions about the migrated features, see:
- **ClawGate Issues**: https://github.com/lisihao/ClawGate/issues
- **Documentation**: `/Users/lisihao/ClawGate/docs/`

For historical Thunder Service questions, see the archived code in this directory.

---

*Archived on 2026-03-11*
*Migrated to: [ClawGate](https://github.com/lisihao/ClawGate)*
