"""
Prometheus Metrics for ClawGate

Exposes metrics for monitoring cache-aware routing decisions and performance.

Author: Claude (Anthropic AI)
Date: 2026-03-12
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from typing import Dict, Any
import time


# ============================================================================
# Decision Metrics
# ============================================================================

force_prefill_total = Counter(
    'clawgate_force_prefill_total',
    'Total number of force prefill decisions',
    ['reason']
)

allow_cache_total = Counter(
    'clawgate_allow_cache_total',
    'Total number of allow cache decisions',
    ['reason']
)

# ============================================================================
# Performance Metrics
# ============================================================================

request_duration = Histogram(
    'clawgate_request_duration_seconds',
    'Request processing duration in seconds',
    ['decision', 'agent_type'],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# ============================================================================
# Overlap Metrics
# ============================================================================

prefix_overlap = Histogram(
    'clawgate_prefix_overlap',
    'Prefix overlap ratio distribution',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

current_overlap = Gauge(
    'clawgate_current_prefix_overlap',
    'Most recent prefix overlap ratio'
)

# ============================================================================
# Cache Metrics
# ============================================================================

lmcache_hit_rate = Gauge(
    'clawgate_lmcache_hit_rate',
    'Current LMCache hit rate from /lmcache/stats'
)

skip_triggered_total = Counter(
    'clawgate_skip_triggered_total',
    'Total number of times skip logic was triggered'
)

# ============================================================================
# System Info
# ============================================================================

clawgate_info = Info(
    'clawgate_build',
    'ClawGate build information'
)

clawgate_info.info({
    'version': '1.0.0',
    'component': 'cache_aware_routing',
    'phase': '2'
})

# ============================================================================
# Helper Functions
# ============================================================================

class MetricsRecorder:
    """
    Helper class for recording metrics with context management
    """

    def __init__(self):
        self.start_time = None
        self.decision = None
        self.agent_type = None

    def start_request(self, agent_type: str):
        """Start timing a request"""
        self.start_time = time.time()
        self.agent_type = agent_type
        return self

    def record_decision(
        self,
        decision: str,
        reason: str,
        overlap: float,
        hit_rate: float,
        skip_triggered: bool = False
    ):
        """
        Record a routing decision

        Args:
            decision: "force_prefill" or "allow_cache"
            reason: Decision reason (e.g., "high_overlap", "low_overlap")
            overlap: Prefix overlap ratio (0.0-1.0)
            hit_rate: LMCache hit rate (0.0-1.0)
            skip_triggered: Whether skip logic was triggered
        """
        self.decision = decision

        # Decision counters
        if decision == "force_prefill":
            force_prefill_total.labels(reason=reason).inc()
        else:
            allow_cache_total.labels(reason=reason).inc()

        # Overlap metrics
        prefix_overlap.observe(overlap)
        current_overlap.set(overlap)

        # Cache metrics
        lmcache_hit_rate.set(hit_rate)

        if skip_triggered:
            skip_triggered_total.inc()

    def finish_request(self):
        """Finish timing a request and record duration"""
        if self.start_time is None:
            return

        duration = time.time() - self.start_time

        if self.decision and self.agent_type:
            request_duration.labels(
                decision=self.decision,
                agent_type=self.agent_type
            ).observe(duration)

        # Reset state
        self.start_time = None
        self.decision = None
        self.agent_type = None


def get_metrics() -> bytes:
    """
    Get current metrics in Prometheus exposition format

    Returns:
        bytes: Metrics in Prometheus text format
    """
    return generate_latest(REGISTRY)


def reset_metrics():
    """Reset all metrics (for testing)"""
    # Note: Prometheus client doesn't support resetting counters
    # This is intentional - counters should be monotonically increasing
    # Only gauges can be reset
    current_overlap.set(0)
    lmcache_hit_rate.set(0)


# ============================================================================
# Convenience Functions
# ============================================================================

def record_force_prefill(reason: str, overlap: float, hit_rate: float):
    """Record a force prefill decision"""
    force_prefill_total.labels(reason=reason).inc()
    prefix_overlap.observe(overlap)
    current_overlap.set(overlap)
    lmcache_hit_rate.set(hit_rate)


def record_allow_cache(reason: str, overlap: float, hit_rate: float):
    """Record an allow cache decision"""
    allow_cache_total.labels(reason=reason).inc()
    prefix_overlap.observe(overlap)
    current_overlap.set(overlap)
    lmcache_hit_rate.set(hit_rate)


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of current metrics (for debugging)

    Returns:
        dict: Summary of key metrics
    """
    # Note: This is a simplified view for debugging
    # For production monitoring, use the Prometheus endpoint
    return {
        "current_overlap": current_overlap._value.get(),
        "lmcache_hit_rate": lmcache_hit_rate._value.get(),
        # Counters don't expose _value in the same way
        # Use Prometheus scraping for accurate counter values
    }
