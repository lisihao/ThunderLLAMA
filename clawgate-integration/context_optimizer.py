"""
Clawgate Context Optimizer - ContextPilot Integration

Bridges ContextPilot, Clawgate, and ThunderLLAMA for optimal context management
in OpenClaw multi-agent architecture.

Author: Claude (Anthropic AI)
Date: 2026-03-12
"""

import asyncio
import contextpilot as cp
from typing import List, Dict, Any, Optional, Tuple
import httpx
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import json
from lmcache_stats_client import LMCacheStatsClient

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Per-agent cache statistics"""
    agent_type: str
    total_requests: int
    cache_hits: int
    cache_misses: int
    tokens_saved: int
    avg_latency_ms: float

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ClawgateContextOptimizer:
    """
    Context optimization layer for Clawgate

    Features:
    - ContextPilot integration (reorder, deduplicate)
    - LMCache eviction sync
    - Per-agent statistics
    - Online and offline modes
    """

    def __init__(
        self,
        thunderllama_url: str = "http://localhost:30000",
        contextpilot_url: Optional[str] = None,
        lmcache_enabled: bool = True,
        enable_eviction_sync: bool = True,
        use_gpu: bool = False,
    ):
        """
        Initialize Clawgate Context Optimizer

        Args:
            thunderllama_url: ThunderLLAMA inference endpoint
            contextpilot_url: ContextPilot server URL (None = embedded mode)
            lmcache_enabled: Enable LMCache integration
            enable_eviction_sync: Sync evictions to ContextPilot
            use_gpu: Use GPU for ContextPilot operations
        """
        self.thunderllama_url = thunderllama_url
        self.lmcache_enabled = lmcache_enabled

        # ContextPilot instance
        if contextpilot_url:
            # Server mode: ContextPilot runs as separate service
            self.cp = None
            self.cp_url = contextpilot_url
            self.cp_mode = "server"
        else:
            # Embedded mode: ContextPilot runs in-process
            self.cp = cp.ContextPilot(use_gpu=use_gpu)
            self.cp_url = None
            self.cp_mode = "embedded"

        # HTTP client for async requests
        self.http_client = httpx.AsyncClient(timeout=60.0)

        # LMCache stats client for cache-aware routing
        self.lmcache_client = LMCacheStatsClient(base_url=thunderllama_url)

        # Track recent requests for prefix overlap calculation
        self.recent_requests = deque(maxlen=20)  # Keep last 20 prompts

        # Statistics tracking (per agent)
        self.stats: Dict[str, CacheStats] = defaultdict(
            lambda: CacheStats("", 0, 0, 0, 0, 0.0)
        )

        # Eviction sync
        if enable_eviction_sync and lmcache_enabled:
            self._setup_eviction_sync()

        logger.info(f"ClawgateContextOptimizer initialized")
        logger.info(f"  Mode: {self.cp_mode}")
        logger.info(f"  ThunderLLAMA: {thunderllama_url}")
        logger.info(f"  LMCache: {'enabled' if lmcache_enabled else 'disabled'}")
        logger.info(f"  Eviction Sync: {'enabled' if enable_eviction_sync else 'disabled'}")

    def _setup_eviction_sync(self):
        """
        Setup eviction sync between LMCache and ContextPilot

        When ThunderLLAMA's LMCache evicts chunks, ContextPilot's index
        must be updated to avoid stale cache assumptions.

        Two approaches:
        1. Webhook: ThunderLLAMA calls ContextPilot /evict endpoint (推荐)
        2. Polling: Clawgate polls LMCache stats and syncs (备选)
        """
        if self.cp_mode == "server":
            # Server mode: use ContextPilot's built-in eviction endpoint
            from contextpilot_hook import enable_eviction_sync
            evict_url = f"{self.cp_url}/evict"
            enable_eviction_sync(
                index_url=evict_url,
                backend_url=self.thunderllama_url
            )
            logger.info(f"Eviction sync enabled: {evict_url}")
        else:
            # Embedded mode: start polling task
            asyncio.create_task(self._eviction_sync_polling())
            logger.info("Eviction sync polling started")

    async def _eviction_sync_polling(self):
        """
        Poll LMCache stats and sync evictions to ContextPilot
        (Fallback mechanism when webhook is not available)
        """
        last_eviction_count = 0

        while True:
            try:
                # Query LMCache stats from ThunderLLAMA
                stats_url = f"{self.thunderllama_url}/lmcache/stats"
                response = await self.http_client.get(stats_url)

                if response.status_code == 200:
                    stats = response.json()
                    current_evictions = stats.get("total_evictions", 0)

                    # Detect new evictions
                    if current_evictions > last_eviction_count:
                        evicted_hashes = stats.get("recently_evicted", [])
                        if evicted_hashes and self.cp:
                            # Notify ContextPilot to evict from index
                            self.cp.evict(evicted_hashes)
                            logger.debug(f"Synced {len(evicted_hashes)} evictions to ContextPilot")

                        last_eviction_count = current_evictions

            except Exception as e:
                logger.warning(f"Eviction sync polling error: {e}")

            # Poll every 5 seconds
            await asyncio.sleep(5)

    async def optimize_agent_request(
        self,
        agent_type: str,
        contexts: List[str],
        query: str,
        model: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize and execute a single agent request (online mode)

        Workflow:
        1. ContextPilot reorders contexts to maximize prefix reuse
        2. Send optimized request to ThunderLLAMA
        3. Track statistics

        Args:
            agent_type: Agent identifier (e.g., "reviewer", "architect")
            contexts: List of context blocks (tools, roles, etc.)
            query: User query
            model: Model name
            **kwargs: Additional generation parameters

        Returns:
            Response from ThunderLLAMA
        """
        import time
        start_time = time.time()

        # Step 1: Optimize with ContextPilot
        if self.cp_mode == "embedded":
            # Embedded mode: call ContextPilot directly
            optimized_messages = self.cp.optimize(contexts, query)
        else:
            # Server mode: HTTP call to ContextPilot
            optimized_messages = await self._call_contextpilot_server(
                contexts, query, mode="online"
            )

        # Convert messages to prompt string for analysis
        optimized_prompt = self._messages_to_prompt(optimized_messages)

        # Step 2: **NEW** Cache-aware routing decision
        should_force, reason = await self._should_force_prefill(
            optimized_prompt, agent_type
        )

        # Step 3: Send to ThunderLLAMA (with cache_prompt parameter)
        response = await self._call_thunderllama(
            messages=optimized_messages,
            model=model,
            cache_prompt=not should_force,  # False → force prefill
            **kwargs
        )

        # Track cache_prompt decision
        response["cache_prompt_used"] = not should_force
        response["skip_triggered"] = response.get("skip_count", 0) > 0
        response["overlap_reason"] = reason

        # Add to recent requests history
        self.recent_requests.append(optimized_prompt)

        # Step 3: Update statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_stats(
            agent_type=agent_type,
            cache_hit=response.get("cache_hit", False),
            tokens_saved=response.get("cached_tokens", 0),
            latency_ms=elapsed_ms
        )

        logger.debug(f"Agent {agent_type}: {elapsed_ms:.1f}ms, cache_hit={response.get('cache_hit')}")

        return response

    async def optimize_batch_requests(
        self,
        agent_requests: List[Dict[str, Any]],
        model: str = "default",
    ) -> List[Dict[str, Any]]:
        """
        Optimize and execute a batch of agent requests (offline mode)

        Workflow:
        1. ContextPilot globally reorders and schedules all requests
        2. Execute in optimized order for maximum cache reuse
        3. Return results in original order

        Args:
            agent_requests: List of request dicts with keys:
                - agent_type: str
                - contexts: List[str]
                - query: str
            model: Model name

        Returns:
            List of responses in original order
        """
        import time
        start_time = time.time()

        # Extract contexts and queries
        all_contexts = [r["contexts"] for r in agent_requests]
        all_queries = [r["query"] for r in agent_requests]

        # Step 1: ContextPilot batch optimization
        if self.cp_mode == "embedded":
            messages_batch, order = self.cp.optimize_batch(all_contexts, all_queries)
        else:
            messages_batch, order = await self._call_contextpilot_batch_server(
                all_contexts, all_queries
            )

        logger.info(f"ContextPilot scheduled {len(agent_requests)} requests")
        logger.debug(f"Execution order: {order}")

        # Step 2: Execute in optimized order
        results = []
        for i, messages in enumerate(messages_batch):
            response = await self._call_thunderllama(
                messages=messages,
                model=model
            )
            results.append(response)

            # Update stats for the agent at original index
            original_idx = order[i]
            agent_type = agent_requests[original_idx]["agent_type"]
            self._update_stats(
                agent_type=agent_type,
                cache_hit=response.get("cache_hit", False),
                tokens_saved=response.get("cached_tokens", 0),
                latency_ms=(time.time() - start_time) * 1000
            )

        # Step 3: Reorder results to match original input order
        reordered_results = [None] * len(results)
        for i, orig_idx in enumerate(order):
            reordered_results[orig_idx] = results[i]

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Batch completed in {total_time:.1f}ms ({len(agent_requests)} requests)")

        return reordered_results

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI message format to plain text prompt for prefix analysis

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            str: Concatenated prompt text

        Example:
            messages = [
                {"role": "system", "content": "You are a reviewer"},
                {"role": "user", "content": "Review this code"}
            ]
            → "You are a reviewer Review this code"
        """
        # Simple concatenation of all message contents
        # Role prefixes are ignored for overlap calculation
        return " ".join(msg.get("content", "") for msg in messages)

    async def _call_thunderllama(
        self,
        messages: List[Dict[str, str]],
        model: str,
        cache_prompt: bool = True,  # **NEW** parameter
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call ThunderLLAMA inference endpoint

        Args:
            messages: OpenAI-compatible message list
            model: Model name
            cache_prompt: Allow session cache (default: True)
                         False → force full prefill, trigger skip logic
            **kwargs: Additional parameters

        Returns:
            Response with cache statistics
        """
        url = f"{self.thunderllama_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "cache_prompt": cache_prompt,  # **NEW** Pass to ThunderLLAMA
            **kwargs
        }

        try:
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()

            result = response.json()

            # Extract cache info from response headers or body
            # (ThunderLLAMA should expose this)
            cache_hit = response.headers.get("X-LMCache-Hit", "false") == "true"
            cached_tokens = int(response.headers.get("X-LMCache-Cached-Tokens", "0"))

            result["cache_hit"] = cache_hit
            result["cached_tokens"] = cached_tokens

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"ThunderLLAMA error: {e.response.status_code} {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"ThunderLLAMA call failed: {e}")
            raise

    async def _call_contextpilot_server(
        self,
        contexts: List[str],
        query: str,
        mode: str = "online"
    ) -> List[Dict[str, str]]:
        """
        Call ContextPilot server (when running in server mode)

        Args:
            contexts: Context blocks
            query: User query
            mode: "online" or "offline"

        Returns:
            Optimized messages
        """
        url = f"{self.cp_url}/optimize"
        payload = {
            "contexts": contexts,
            "query": query,
            "mode": mode
        }

        response = await self.http_client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["messages"]

    async def _call_contextpilot_batch_server(
        self,
        all_contexts: List[List[str]],
        all_queries: List[str]
    ) -> Tuple[List[List[Dict[str, str]]], List[int]]:
        """
        Call ContextPilot server for batch optimization

        Returns:
            (messages_batch, execution_order)
        """
        url = f"{self.cp_url}/optimize_batch"
        payload = {
            "contexts_batch": all_contexts,
            "queries": all_queries
        }

        response = await self.http_client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["messages_batch"], result["order"]

    def _update_stats(
        self,
        agent_type: str,
        cache_hit: bool,
        tokens_saved: int,
        latency_ms: float
    ):
        """
        Update per-agent statistics

        Args:
            agent_type: Agent identifier
            cache_hit: Whether cache was hit
            tokens_saved: Number of tokens saved by cache
            latency_ms: Request latency in milliseconds
        """
        stats = self.stats[agent_type]

        if stats.agent_type == "":
            stats.agent_type = agent_type

        stats.total_requests += 1

        if cache_hit:
            stats.cache_hits += 1
        else:
            stats.cache_misses += 1

        stats.tokens_saved += tokens_saved

        # Update average latency (exponential moving average)
        alpha = 0.2
        stats.avg_latency_ms = (
            alpha * latency_ms + (1 - alpha) * stats.avg_latency_ms
        )

    def get_stats(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics

        Args:
            agent_type: Specific agent (None = all agents)

        Returns:
            Statistics dict
        """
        if agent_type:
            stats = self.stats[agent_type]
            return {
                "agent_type": stats.agent_type,
                "total_requests": stats.total_requests,
                "cache_hit_rate": stats.hit_rate,
                "tokens_saved": stats.tokens_saved,
                "avg_latency_ms": stats.avg_latency_ms
            }
        else:
            # Aggregate stats for all agents
            return {
                agent: {
                    "total_requests": stats.total_requests,
                    "cache_hit_rate": stats.hit_rate,
                    "tokens_saved": stats.tokens_saved,
                    "avg_latency_ms": stats.avg_latency_ms
                }
                for agent, stats in self.stats.items()
            }

    def print_stats(self):
        """
        Pretty print statistics
        """
        print("\n" + "="*70)
        print("  CLAWGATE CONTEXT OPTIMIZER STATISTICS")
        print("="*70)

        for agent, stats in sorted(self.stats.items()):
            if stats.total_requests == 0:
                continue

            print(f"\nAgent: {agent}")
            print(f"  Total Requests:  {stats.total_requests}")
            print(f"  Cache Hit Rate:  {stats.hit_rate*100:.1f}%")
            print(f"  Tokens Saved:    {stats.tokens_saved:,}")
            print(f"  Avg Latency:     {stats.avg_latency_ms:.1f} ms")

        print("\n" + "="*70)

    # ========================================================================
    # Phase 1: Cache-Aware Routing (2026-03-12)
    # ========================================================================

    def get_prefix_overlap(
        self,
        prompt: str,
        window_size: int = 10
    ) -> float:
        """
        Calculate prefix overlap between current prompt and recent requests

        Args:
            prompt: Current prompt text
            window_size: Number of recent requests to compare (default: 10)

        Returns:
            float: Average prefix overlap ratio (0.0 - 1.0)
                  0.0 = no overlap
                  1.0 = complete overlap

        Note:
            Uses simple token-level prefix matching.
            Efficient enough for routing decisions.
        """
        if not self.recent_requests:
            return 0.0

        # Get recent N prompts
        recent = list(self.recent_requests)[-window_size:]

        # Calculate overlap with each
        overlaps = []
        for prev_prompt in recent:
            overlap = self._calculate_prefix_overlap(prompt, prev_prompt)
            overlaps.append(overlap)

        # Return average
        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    def _calculate_prefix_overlap(
        self,
        prompt1: str,
        prompt2: str
    ) -> float:
        """
        Calculate prefix overlap between two prompts

        Algorithm:
        1. Simple whitespace tokenization
        2. Find longest common prefix
        3. Return overlap ratio

        Args:
            prompt1: First prompt
            prompt2: Second prompt

        Returns:
            float: Overlap ratio (0.0 - 1.0)
        """
        # Simple tokenization (split by whitespace)
        tokens1 = prompt1.split()
        tokens2 = prompt2.split()

        # Find longest common prefix
        common_len = 0
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2:
                common_len += 1
            else:
                break

        # Calculate overlap ratio (relative to shorter prompt)
        min_len = min(len(tokens1), len(tokens2))
        return common_len / min_len if min_len > 0 else 0.0

    async def _should_force_prefill(
        self,
        prompt: str,
        agent_type: str
    ) -> Tuple[bool, str]:
        """
        Decide whether to force full prefill (disable session cache)

        Decision logic:
        - High overlap (>80%) + High LMCache hit (>90%) → Force prefill
        - Low overlap (<30%) → Allow session cache
        - Medium → Allow session cache

        Args:
            prompt: Optimized prompt after ContextPilot
            agent_type: Agent type for logging

        Returns:
            Tuple[bool, str]: (should_force, reason)
                - should_force: True → set cache_prompt=false
                - reason: Decision reason for logging
        """
        # 1. Calculate prefix overlap
        overlap = self.get_prefix_overlap(prompt, window_size=10)

        # 2. Query LMCache hit rate
        cache_hit_rate = await self.lmcache_client.get_estimated_hit_rate()

        # 3. Decision rules
        if overlap > 0.8 and cache_hit_rate > 0.9:
            # High overlap + high hit → force prefill to trigger skip logic
            reason = f"high_overlap={overlap:.2f},hit_rate={cache_hit_rate:.2f}"
            logger.info(f"[{agent_type}] Force prefill: {reason}")
            return True, reason

        elif overlap < 0.3:
            # Low overlap → allow session cache
            reason = f"low_overlap={overlap:.2f}"
            logger.debug(f"[{agent_type}] Allow session cache: {reason}")
            return False, reason

        else:
            # Medium overlap → use hybrid hashing, allow session cache
            reason = f"medium_overlap={overlap:.2f}"
            logger.debug(f"[{agent_type}] Medium overlap: {reason}")
            return False, reason

    async def close(self):
        """
        Cleanup resources
        """
        await self.http_client.aclose()
        await self.lmcache_client.close()
        logger.info("ClawgateContextOptimizer closed")


# ============================================================================
# Helper Functions
# ============================================================================

def create_optimizer_from_env() -> ClawgateContextOptimizer:
    """
    Create optimizer from environment variables

    Environment variables:
        THUNDERLLAMA_URL: ThunderLLAMA endpoint (default: http://localhost:30000)
        CONTEXTPILOT_URL: ContextPilot server URL (default: None = embedded)
        LMCACHE_ENABLED: Enable LMCache (default: true)
        EVICTION_SYNC_ENABLED: Enable eviction sync (default: true)
        CONTEXTPILOT_GPU: Use GPU for ContextPilot (default: false)
    """
    import os

    return ClawgateContextOptimizer(
        thunderllama_url=os.getenv("THUNDERLLAMA_URL", "http://localhost:30000"),
        contextpilot_url=os.getenv("CONTEXTPILOT_URL"),
        lmcache_enabled=os.getenv("LMCACHE_ENABLED", "true").lower() == "true",
        enable_eviction_sync=os.getenv("EVICTION_SYNC_ENABLED", "true").lower() == "true",
        use_gpu=os.getenv("CONTEXTPILOT_GPU", "false").lower() == "true",
    )
