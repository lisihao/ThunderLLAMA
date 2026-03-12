#!/usr/bin/env python3
"""
Multi-Agent Performance Benchmark

Benchmark ContextPilot + LMCache performance improvements
for OpenClaw multi-agent scenarios.

Scenarios:
1. Single agent repeated calls (memory effect)
2. Multiple agents with shared tools (prefix sharing)
3. Batch scheduling optimization
4. Large-scale agent orchestration

Author: Claude (Anthropic AI)
Date: 2026-03-12
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
import json

from context_optimizer import ClawgateContextOptimizer
from openclaw_prompt_builder import OpenClawPromptBuilder, ToolCategory


@dataclass
class BenchmarkResult:
    """Benchmark result metrics"""
    scenario: str
    num_requests: int
    total_time_ms: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float
    tokens_saved: int
    throughput_rps: float


class MultiAgentBenchmark:
    """Performance benchmark suite"""

    def __init__(self, thunderllama_url: str = "http://localhost:30000"):
        self.thunderllama_url = thunderllama_url
        self.optimizer = None
        self.prompt_builder = OpenClawPromptBuilder()
        self.results: List[BenchmarkResult] = []

    async def setup(self):
        """Initialize benchmark environment"""
        print("\n" + "="*70)
        print("  MULTI-AGENT PERFORMANCE BENCHMARK")
        print("="*70)
        print(f"  ThunderLLAMA: {self.thunderllama_url}")
        print(f"  LMCache: Enabled")
        print(f"  ContextPilot: Enabled")
        print("="*70)

        self.optimizer = ClawgateContextOptimizer(
            thunderllama_url=self.thunderllama_url,
            lmcache_enabled=True,
            enable_eviction_sync=True
        )

    async def teardown(self):
        """Cleanup"""
        if self.optimizer:
            await self.optimizer.close()

    # ========================================================================
    # Benchmark 1: Single Agent Repeated Calls (Memory Effect)
    # ========================================================================

    async def benchmark_single_agent_repeated(self, num_calls: int = 100):
        """
        Benchmark: Same agent, same context, different queries

        Expected: High cache hit rate after first call
        """
        print("\n" + "="*70)
        print(f"  BENCHMARK 1: Single Agent Repeated ({num_calls} calls)")
        print("="*70)

        agent_type = "reviewer"
        contexts = self.prompt_builder.build(
            agent_type=agent_type,
            task="Review code for bugs"
        )

        latencies = []
        cache_hits = 0
        tokens_saved_total = 0

        start_time = time.time()

        for i in range(num_calls):
            req_start = time.time()

            response = await self.optimizer.optimize_agent_request(
                agent_type=agent_type,
                contexts=contexts,
                query=f"Review iteration {i}",
                model="default"
            )

            latency = (time.time() - req_start) * 1000
            latencies.append(latency)

            if response.get('cache_hit', False):
                cache_hits += 1
            tokens_saved_total += response.get('cached_tokens', 0)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_calls} calls")

        total_time = (time.time() - start_time) * 1000

        result = BenchmarkResult(
            scenario="Single Agent Repeated",
            num_requests=num_calls,
            total_time_ms=total_time,
            avg_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
            cache_hit_rate=cache_hits / num_calls,
            tokens_saved=tokens_saved_total,
            throughput_rps=num_calls / (total_time / 1000)
        )

        self._print_result(result)
        self.results.append(result)

        return result

    # ========================================================================
    # Benchmark 2: Multiple Agents with Shared Tools (Prefix Sharing)
    # ========================================================================

    async def benchmark_multi_agent_shared_tools(self, num_agents: int = 10):
        """
        Benchmark: Different agents, same tool set, different tasks

        Expected: High prefix cache hit rate
        """
        print("\n" + "="*70)
        print(f"  BENCHMARK 2: Multi-Agent Shared Tools ({num_agents} agents)")
        print("="*70)

        agent_types = ["reviewer", "architect", "coder", "tester", "documenter", "ops"]
        tool_categories = [ToolCategory.FILE_OPS, ToolCategory.CODE_ANALYSIS]

        agent_requests = []
        for i in range(num_agents):
            agent_type = agent_types[i % len(agent_types)]
            agent_requests.append({
                "agent_type": agent_type,
                "contexts": self.prompt_builder.build(
                    agent_type=agent_type,
                    task=f"Task {i} for {agent_type}",
                    tool_categories=tool_categories
                ),
                "query": f"Execute task {i}"
            })

        latencies = []
        cache_hits = 0
        tokens_saved_total = 0

        start_time = time.time()

        # Execute sequentially to measure individual latencies
        for i, req in enumerate(agent_requests):
            req_start = time.time()

            response = await self.optimizer.optimize_agent_request(
                agent_type=req["agent_type"],
                contexts=req["contexts"],
                query=req["query"],
                model="default"
            )

            latency = (time.time() - req_start) * 1000
            latencies.append(latency)

            if response.get('cache_hit', False):
                cache_hits += 1
            tokens_saved_total += response.get('cached_tokens', 0)

        total_time = (time.time() - start_time) * 1000

        result = BenchmarkResult(
            scenario=f"Multi-Agent Shared Tools ({num_agents} agents)",
            num_requests=num_agents,
            total_time_ms=total_time,
            avg_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
            cache_hit_rate=cache_hits / num_agents,
            tokens_saved=tokens_saved_total,
            throughput_rps=num_agents / (total_time / 1000)
        )

        self._print_result(result)
        self.results.append(result)

        return result

    # ========================================================================
    # Benchmark 3: Batch Scheduling Optimization
    # ========================================================================

    async def benchmark_batch_scheduling(self, num_agents: int = 20):
        """
        Benchmark: Batch optimization with ContextPilot scheduling

        Expected: Better cache reuse with optimized ordering
        """
        print("\n" + "="*70)
        print(f"  BENCHMARK 3: Batch Scheduling ({num_agents} agents)")
        print("="*70)

        agent_types = ["reviewer", "architect", "coder"]
        tool_categories = [ToolCategory.FILE_OPS, ToolCategory.CODE_ANALYSIS]

        agent_requests = []
        for i in range(num_agents):
            agent_type = agent_types[i % len(agent_types)]
            agent_requests.append({
                "agent_type": agent_type,
                "contexts": self.prompt_builder.build(
                    agent_type=agent_type,
                    task=f"Batch task {i}",
                    tool_categories=tool_categories
                ),
                "query": f"Batch query {i}"
            })

        start_time = time.time()

        results = await self.optimizer.optimize_batch_requests(
            agent_requests,
            model="default"
        )

        total_time = (time.time() - start_time) * 1000

        cache_hits = sum(1 for r in results if r.get('cache_hit', False))
        tokens_saved_total = sum(r.get('cached_tokens', 0) for r in results)

        result = BenchmarkResult(
            scenario=f"Batch Scheduling ({num_agents} agents)",
            num_requests=num_agents,
            total_time_ms=total_time,
            avg_latency_ms=total_time / num_agents,
            median_latency_ms=total_time / num_agents,
            p95_latency_ms=total_time / num_agents,
            cache_hit_rate=cache_hits / num_agents,
            tokens_saved=tokens_saved_total,
            throughput_rps=num_agents / (total_time / 1000)
        )

        self._print_result(result)
        self.results.append(result)

        return result

    # ========================================================================
    # Benchmark 4: Large-Scale Orchestration (OpenClaw Realistic)
    # ========================================================================

    async def benchmark_large_scale(self, num_rounds: int = 5, agents_per_round: int = 10):
        """
        Benchmark: Simulate OpenClaw realistic workload

        Multiple rounds of multi-agent execution
        """
        print("\n" + "="*70)
        print(f"  BENCHMARK 4: Large-Scale Orchestration")
        print(f"  ({num_rounds} rounds × {agents_per_round} agents = {num_rounds * agents_per_round} total)")
        print("="*70)

        total_requests = 0
        total_cache_hits = 0
        total_tokens_saved = 0
        all_latencies = []

        overall_start = time.time()

        for round_num in range(num_rounds):
            print(f"\n  Round {round_num + 1}/{num_rounds}...")

            agent_types = ["reviewer", "architect", "coder", "tester"]
            agent_requests = []

            for i in range(agents_per_round):
                agent_type = agent_types[i % len(agent_types)]
                agent_requests.append({
                    "agent_type": agent_type,
                    "contexts": self.prompt_builder.build(
                        agent_type=agent_type,
                        task=f"Round {round_num} task {i}"
                    ),
                    "query": f"R{round_num} Q{i}"
                })

            round_start = time.time()
            results = await self.optimizer.optimize_batch_requests(
                agent_requests,
                model="default"
            )
            round_time = (time.time() - round_start) * 1000

            # Aggregate stats
            for r in results:
                all_latencies.append(round_time / len(results))
                if r.get('cache_hit', False):
                    total_cache_hits += 1
                total_tokens_saved += r.get('cached_tokens', 0)

            total_requests += len(results)

            print(f"    Time: {round_time:.1f} ms")
            print(f"    Cache hit rate: {sum(1 for r in results if r.get('cache_hit')) / len(results) * 100:.1f}%")

        total_time = (time.time() - overall_start) * 1000

        result = BenchmarkResult(
            scenario=f"Large-Scale ({num_rounds}×{agents_per_round})",
            num_requests=total_requests,
            total_time_ms=total_time,
            avg_latency_ms=statistics.mean(all_latencies),
            median_latency_ms=statistics.median(all_latencies),
            p95_latency_ms=statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 20 else max(all_latencies),
            cache_hit_rate=total_cache_hits / total_requests,
            tokens_saved=total_tokens_saved,
            throughput_rps=total_requests / (total_time / 1000)
        )

        self._print_result(result)
        self.results.append(result)

        return result

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _print_result(self, result: BenchmarkResult):
        """Pretty print benchmark result"""
        print("\n  Results:")
        print(f"    Total Time:     {result.total_time_ms:.1f} ms")
        print(f"    Avg Latency:    {result.avg_latency_ms:.1f} ms")
        print(f"    Median Latency: {result.median_latency_ms:.1f} ms")
        print(f"    P95 Latency:    {result.p95_latency_ms:.1f} ms")
        print(f"    Cache Hit Rate: {result.cache_hit_rate * 100:.1f}%")
        print(f"    Tokens Saved:   {result.tokens_saved:,}")
        print(f"    Throughput:     {result.throughput_rps:.2f} req/s")

    def print_summary(self):
        """Print benchmark summary table"""
        print("\n" + "="*70)
        print("  BENCHMARK SUMMARY")
        print("="*70)

        print(f"\n{'Scenario':<40} {'Requests':<12} {'Avg Lat (ms)':<15} {'Hit Rate':<12} {'Throughput':<12}")
        print("-" * 100)

        for r in self.results:
            print(f"{r.scenario:<40} {r.num_requests:<12} {r.avg_latency_ms:<15.1f} "
                  f"{r.cache_hit_rate*100:<11.1f}% {r.throughput_rps:<11.2f} req/s")

        print("\n" + "="*70)
        print("  OVERALL STATISTICS")
        print("="*70)

        total_requests = sum(r.num_requests for r in self.results)
        total_cache_hits = sum(int(r.num_requests * r.cache_hit_rate) for r in self.results)
        total_tokens_saved = sum(r.tokens_saved for r in self.results)

        print(f"  Total Requests:    {total_requests:,}")
        print(f"  Total Cache Hits:  {total_cache_hits:,} ({total_cache_hits/total_requests*100:.1f}%)")
        print(f"  Total Tokens Saved: {total_tokens_saved:,}")

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file"""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "thunderllama_url": self.thunderllama_url,
            "results": [
                {
                    "scenario": r.scenario,
                    "num_requests": r.num_requests,
                    "total_time_ms": r.total_time_ms,
                    "avg_latency_ms": r.avg_latency_ms,
                    "median_latency_ms": r.median_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "cache_hit_rate": r.cache_hit_rate,
                    "tokens_saved": r.tokens_saved,
                    "throughput_rps": r.throughput_rps
                }
                for r in self.results
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results saved to {filename}")

    # ========================================================================
    # Main Runner
    # ========================================================================

    async def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        await self.setup()

        try:
            # Benchmark 1: Single agent repeated
            await self.benchmark_single_agent_repeated(num_calls=50)

            # Benchmark 2: Multi-agent shared tools
            await self.benchmark_multi_agent_shared_tools(num_agents=10)

            # Benchmark 3: Batch scheduling
            await self.benchmark_batch_scheduling(num_agents=20)

            # Benchmark 4: Large-scale
            await self.benchmark_large_scale(num_rounds=3, agents_per_round=10)

            # Print summary
            self.print_summary()

            # Show optimizer stats
            print("\n")
            self.optimizer.print_stats()

            # Save results
            self.save_results()

        finally:
            await self.teardown()


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main benchmark runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Performance Benchmark")
    parser.add_argument(
        "--thunderllama-url",
        default="http://localhost:30000",
        help="ThunderLLAMA endpoint URL"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    benchmark = MultiAgentBenchmark(thunderllama_url=args.thunderllama_url)
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
