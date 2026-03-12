# Phase 2 Plan: Monitoring & Optimization

**Date**: 2026-03-12
**Status**: 🚧 In Progress
**Dependencies**: Phase 1 ✅ Complete

---

## Objectives

1. **Monitoring**: 实时可观测性，了解系统运行状态
2. **Optimization**: 基于真实数据调优决策阈值
3. **Integration**: 集成真实 ThunderChunkStorage 统计

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Monitoring Layer                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prometheus Metrics      Grafana Dashboard                 │
│  ├─ skip_rate           ├─ Performance Over Time            │
│  ├─ cache_hit_rate      ├─ Decision Breakdown               │
│  ├─ overlap_distribution├─ Cache Hit Rate Trends            │
│  ├─ decision_latency    └─ System Health                    │
│  └─ lmcache_stats                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓ Metrics Flow
┌─────────────────────────────────────────────────────────────┐
│                   ClawGate + ContextPilot                   │
│  ├─ Cache-aware routing decisions                          │
│  ├─ Prefix overlap calculation                             │
│  └─ Performance tracking                                    │
└─────────────────────────────────────────────────────────────┘
         ↓ API Calls
┌─────────────────────────────────────────────────────────────┐
│              ThunderLLAMA + LMCache                         │
│  ├─ /lmcache/stats endpoint                                │
│  ├─ Skip logic execution                                    │
│  └─ Hybrid hashing                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 2.1: Prometheus Metrics 集成

### Metrics to Expose

**ClawGate Metrics** (`/metrics` endpoint):
```python
# Decision metrics
clawgate_force_prefill_total{reason="high_overlap"} 150
clawgate_allow_cache_total{reason="low_overlap"} 45

# Performance metrics
clawgate_request_duration_seconds{decision="force_prefill"} 0.285
clawgate_request_duration_seconds{decision="allow_cache"} 0.920

# Overlap distribution
clawgate_prefix_overlap_bucket{le="0.3"} 45
clawgate_prefix_overlap_bucket{le="0.8"} 110
clawgate_prefix_overlap_bucket{le="1.0"} 150

# Cache hit rate
clawgate_lmcache_hit_rate 0.95
```

**ThunderLLAMA Metrics** (extend `/lmcache/stats`):
```json
{
  "l2_chunks": 336,
  "l2_usage_bytes": 86016000,
  "l2_hit_rate": 0.95,
  "skip_count": 127,
  "skip_rate": 0.85,
  "prefill_chunks_found": 192,
  "prefill_chunks_needed": 192
}
```

### Implementation Tasks

- [ ] **Task 2.1.1**: Add Prometheus client to ClawGate
  - File: `clawgate-integration/metrics.py`
  - Metrics: force_prefill, overlap, latency
  - Endpoint: `http://localhost:9090/metrics`

- [ ] **Task 2.1.2**: Extend ThunderLLAMA `/lmcache/stats`
  - File: `tools/server/server-context.cpp`
  - Add: `skip_count`, `skip_rate` counters
  - Implement: Real ThunderChunkStorage stats access

- [ ] **Task 2.1.3**: Set up Prometheus scraping
  - Config: `prometheus.yml`
  - Targets: ClawGate:9090, ThunderLLAMA:30000
  - Interval: 15s

**Estimated Time**: 3-4 hours

---

## Phase 2.2: Grafana Dashboard

### Dashboard Panels

**Panel 1: Performance Overview**
- Line chart: Request latency over time
- Grouped by: force_prefill vs allow_cache
- Metric: `clawgate_request_duration_seconds`

**Panel 2: Decision Breakdown**
- Pie chart: force_prefill vs allow_cache distribution
- Metric: `clawgate_force_prefill_total`, `clawgate_allow_cache_total`

**Panel 3: Cache Hit Rate**
- Line chart: LMCache hit rate over time
- Metric: `clawgate_lmcache_hit_rate`
- Alert: < 0.7 (low hit rate warning)

**Panel 4: Skip Logic Activity**
- Counter: Total skip count
- Gauge: Current skip rate
- Metric: From `/lmcache/stats`

**Panel 5: Overlap Distribution**
- Histogram: Prefix overlap distribution
- Metric: `clawgate_prefix_overlap_bucket`

### Implementation Tasks

- [ ] **Task 2.2.1**: Create Grafana dashboard JSON
  - File: `clawgate-integration/grafana-dashboard.json`
  - Import to Grafana

- [ ] **Task 2.2.2**: Configure alerts
  - Low hit rate: < 0.7
  - High latency: > 1s
  - Low skip rate: < 0.5 (when overlap > 0.8)

**Estimated Time**: 2-3 hours

---

## Phase 2.3: Decision Threshold Tuning

### Current Thresholds (Phase 1)

```python
if overlap > 0.8 and cache_hit_rate > 0.9:
    decision = "force_prefill"
elif overlap < 0.3:
    decision = "allow_cache"
else:
    decision = "allow_cache"  # Medium overlap
```

### Data Collection

**Goal**: Collect 1000+ real requests to analyze:
- Overlap distribution
- Hit rate vs overlap correlation
- Latency vs decision type
- Skip rate vs overlap

**Collection Method**:
```python
# Log to CSV for analysis
# File: ~/.openclaw/decision_log.csv
timestamp, overlap, hit_rate, decision, latency_ms, skip_triggered
```

### Analysis Questions

1. **Overlap threshold**: Is 0.8 optimal or should it be 0.75/0.85?
2. **Hit rate threshold**: Is 0.9 too strict? Try 0.8?
3. **Medium overlap**: Should we force_prefill for 0.5-0.8 range?
4. **Adaptive thresholds**: Should thresholds change based on time-of-day?

### Implementation Tasks

- [ ] **Task 2.3.1**: Add decision logging
  - File: `clawgate-integration/context_optimizer.py`
  - Log: CSV format with all decision factors

- [ ] **Task 2.3.2**: Create analysis notebook
  - File: `clawgate-integration/analyze_decisions.ipynb`
  - Visualize: Overlap vs latency, hit_rate vs skip_rate

- [ ] **Task 2.3.3**: Tune thresholds based on data
  - Update: `_should_force_prefill()` logic
  - A/B test: Old vs new thresholds

**Estimated Time**: 4-5 hours

---

## Phase 2.4: ThunderChunkStorage Integration

### Current Issue

`/lmcache/stats` returns **optimistic mock data**:
```json
{
  "l2_hit_rate": 0.95,  // Always 0.95, not real
  "note": "Optimistic implementation"
}
```

### Goal

Return **real statistics** from ThunderChunkStorage:
```json
{
  "l2_chunks": 336,          // Real count
  "l2_usage_bytes": 86016000, // Real usage
  "l2_hit_rate": 0.87,       // Calculated from hits/misses
  "l3_chunks": 0,
  "l3_usage_bytes": 0,
  "skip_count": 127,         // New
  "total_prefills": 150      // New
}
```

### Implementation Tasks

- [ ] **Task 2.4.1**: Add stats tracking to ThunderChunkStorage
  - File: `src/thunder-chunk-storage.cpp`
  - Add: `total_hits`, `total_misses`, `skip_count` counters

- [ ] **Task 2.4.2**: Expose stats via accessor
  - Method: `ThunderChunkStorage::get_stats()`
  - Return: `{chunks, usage_bytes, hit_rate, skip_count}`

- [ ] **Task 2.4.3**: Update `/lmcache/stats` handler
  - File: `tools/server/server-context.cpp`
  - Replace mock with real stats
  - Remove "Optimistic implementation" note

**Estimated Time**: 2-3 hours

---

## Success Criteria

### Phase 2 Complete When:

- ✅ Prometheus metrics exposed from ClawGate and ThunderLLAMA
- ✅ Grafana dashboard shows real-time performance
- ✅ Collected 1000+ decision logs for analysis
- ✅ Thresholds tuned based on data (if needed)
- ✅ ThunderChunkStorage returns real stats (not mock)

### Performance Targets:

- **Monitoring overhead**: < 5ms per request
- **Dashboard refresh**: < 1s
- **Metrics storage**: < 100MB for 1 week data

---

## Timeline

| Phase | Tasks | Estimated Time | Status |
|-------|-------|----------------|--------|
| 2.1 | Prometheus integration | 3-4 hours | ⏳ Pending |
| 2.2 | Grafana dashboard | 2-3 hours | ⏳ Pending |
| 2.3 | Threshold tuning | 4-5 hours | ⏳ Pending |
| 2.4 | ThunderChunkStorage stats | 2-3 hours | ⏳ Pending |
| **Total** | | **11-15 hours** | |

---

## Next Actions

### Immediate (Start Phase 2.1)

```bash
# 1. Install Prometheus client
cd /Users/lisihao/ThunderLLAMA/clawgate-integration
pip install prometheus-client

# 2. Create metrics.py
cat > metrics.py << 'EOF'
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Decision metrics
force_prefill_total = Counter('clawgate_force_prefill_total', 'Total force prefill decisions', ['reason'])
allow_cache_total = Counter('clawgate_allow_cache_total', 'Total allow cache decisions', ['reason'])

# Performance metrics
request_duration = Histogram('clawgate_request_duration_seconds', 'Request duration', ['decision'])

# Overlap distribution
overlap_gauge = Gauge('clawgate_prefix_overlap', 'Current prefix overlap')

# Cache hit rate
hit_rate_gauge = Gauge('clawgate_lmcache_hit_rate', 'LMCache hit rate')
EOF

# 3. Add /metrics endpoint to ClawGate
# Edit: clawgate-integration/server.py (if exists) or context_optimizer.py
```

### Documentation

Create monitoring guide:
```bash
# File: clawgate-integration/MONITORING.md
- How to start Prometheus
- How to access Grafana
- How to interpret metrics
- Troubleshooting guide
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Prometheus overhead | High latency | Use async metrics, batch updates |
| Storage growth | Disk full | Retention: 7 days, scrape interval: 15s |
| Grafana complexity | Hard to maintain | Start simple, iterate based on needs |
| Data quality | Wrong thresholds | Validate logs, cross-check with manual tests |

---

**Phase 2 Owner**: Claude Sonnet 4.5
**Start Date**: 2026-03-12
**Target Completion**: 2026-03-13

---

**Related Documents**:
- `PHASE1_INVESTIGATION_REPORT.md` - Phase 1 results
- `THREE_LAYER_OPTIMIZATION.md` - Overall architecture
- `UNIFIED_CONFIG.md` - Configuration management
