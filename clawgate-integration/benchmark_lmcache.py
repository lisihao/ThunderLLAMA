#!/usr/bin/env python3
"""
LMCache Performance Benchmark

Compare ThunderLLAMA performance with and without LMCache
for OpenClaw multi-agent scenarios.

Author: Claude (Anthropic AI)
Date: 2026-03-12
"""

import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
import httpx
import subprocess
import os
import signal

class LMCacheBenchmark:
    """Performance benchmark for LMCache"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.base_url = "http://localhost:30000"
        self.server_process = None
        self.results = {}

    def start_server(self, enable_lmcache: bool = False):
        """Start ThunderLLAMA server"""
        print(f"\n{'='*70}")
        print(f"  Starting ThunderLLAMA (LMCache: {'ON' if enable_lmcache else 'OFF'})")
        print(f"{'='*70}")

        # Stop any existing server
        self.stop_server()

        # Prepare command
        cmd = [
            "/Users/lisihao/ThunderLLAMA/build/bin/llama-server",
            "--model", self.model_path,
            "--port", "30000",
            "--n-gpu-layers", "99",
            "--ctx-size", "8192",
            "--threads", "8",
        ]

        # Set environment variables
        env = os.environ.copy()
        if enable_lmcache:
            env["LMCACHE_ENABLED"] = "true"
            env["THUNDER_LMCACHE_DISK_PATH"] = os.path.expanduser("~/.openclaw/lmcache_bench.bin")
            env["THUNDER_LMCACHE_L2_SIZE"] = str(8 * 1024 * 1024 * 1024)  # 8GB
            print(f"  LMCache: ENABLED")
            print(f"  Cache path: ~/.openclaw/lmcache_bench.bin")
        else:
            env["LMCACHE_ENABLED"] = "false"
            print(f"  LMCache: DISABLED")

        # Start server
        print(f"  Model: {os.path.basename(self.model_path)}")
        print(f"  Starting server...")

        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to be ready
        print(f"  Waiting for server...")
        for i in range(60):
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    print(f"  ✓ Server ready in {i+1}s\n")
                    time.sleep(2)  # Extra warmup
                    return
            except:
                time.sleep(1)

        raise RuntimeError("Server failed to start")

    def stop_server(self):
        """Stop ThunderLLAMA server"""
        if self.server_process:
            print(f"  Stopping server...")
            self.server_process.send_signal(signal.SIGTERM)
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            time.sleep(2)

    async def run_inference(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Run single inference"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            start = time.time()

            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                }
            )

            latency = (time.time() - start) * 1000  # ms

            if response.status_code == 200:
                data = response.json()
                usage = data.get('usage', {})
                return {
                    'success': True,
                    'latency_ms': latency,
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                }
            else:
                return {'success': False, 'error': response.text}

    async def benchmark_scenario(self, scenario_name: str, prompts: List[str], enable_lmcache: bool):
        """Run benchmark scenario"""
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"  LMCache: {'ON' if enable_lmcache else 'OFF'}")
        print(f"{'='*70}\n")

        # Start server
        self.start_server(enable_lmcache=enable_lmcache)

        # Run inferences
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  Request {i+1}/{len(prompts)}: ", end='', flush=True)

            result = await self.run_inference(prompt, max_tokens=50)

            if result['success']:
                print(f"{result['latency_ms']:.0f}ms "
                      f"({result['prompt_tokens']} prompt tokens)")
                results.append(result)
            else:
                print(f"FAILED: {result.get('error', 'Unknown')}")

        # Calculate statistics
        if results:
            latencies = [r['latency_ms'] for r in results]
            prompt_tokens = [r['prompt_tokens'] for r in results]

            stats = {
                'scenario': scenario_name,
                'lmcache_enabled': enable_lmcache,
                'num_requests': len(results),
                'latency_first_ms': latencies[0] if latencies else 0,
                'latency_avg_ms': statistics.mean(latencies),
                'latency_median_ms': statistics.median(latencies),
                'latency_min_ms': min(latencies),
                'latency_max_ms': max(latencies),
                'speedup_2nd_vs_1st': latencies[0] / latencies[1] if len(latencies) > 1 else 1.0,
                'avg_prompt_tokens': statistics.mean(prompt_tokens),
                'total_latency_ms': sum(latencies),
            }

            # Print summary
            print(f"\n  Results:")
            print(f"    First request:  {stats['latency_first_ms']:.0f} ms (cold)")
            if len(latencies) > 1:
                print(f"    Second request: {latencies[1]:.0f} ms (warm) - {stats['speedup_2nd_vs_1st']:.1f}x faster")
                print(f"    Average (2+):   {statistics.mean(latencies[1:]):.0f} ms")
            print(f"    Total time:     {stats['total_latency_ms']/1000:.1f} s")

            return stats

        return None

    async def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("\n" + "="*70)
        print("  LMCACHE PERFORMANCE BENCHMARK")
        print("="*70)
        print(f"  Model: {os.path.basename(self.model_path)}")
        print("="*70)

        # Scenario 1: Repeated calls (same prompt)
        # Simulates single agent asking similar questions
        repeated_prompts = [
            "Explain how photosynthesis works in plants.",
            "Explain how photosynthesis works in plants.",
            "Explain how photosynthesis works in plants.",
            "Explain how photosynthesis works in plants.",
            "Explain how photosynthesis works in plants.",
        ]

        print("\n" + "="*70)
        print("  TEST 1: Repeated Identical Prompts")
        print("  (Simulates agent memory/caching)")
        print("="*70)

        without_cache_1 = await self.benchmark_scenario(
            "Repeated Prompts (No Cache)",
            repeated_prompts,
            enable_lmcache=False
        )

        with_cache_1 = await self.benchmark_scenario(
            "Repeated Prompts (With Cache)",
            repeated_prompts,
            enable_lmcache=True
        )

        # Scenario 2: Shared prefix prompts
        # Simulates multi-agent with shared system prompts
        shared_prefix_prompts = [
            "You are a helpful AI assistant.\n\nUser: What is machine learning?",
            "You are a helpful AI assistant.\n\nUser: What is deep learning?",
            "You are a helpful AI assistant.\n\nUser: What is neural network?",
            "You are a helpful AI assistant.\n\nUser: What is reinforcement learning?",
            "You are a helpful AI assistant.\n\nUser: What is computer vision?",
        ]

        print("\n" + "="*70)
        print("  TEST 2: Shared Prefix Prompts")
        print("  (Simulates multi-agent with same system prompt)")
        print("="*70)

        without_cache_2 = await self.benchmark_scenario(
            "Shared Prefix (No Cache)",
            shared_prefix_prompts,
            enable_lmcache=False
        )

        with_cache_2 = await self.benchmark_scenario(
            "Shared Prefix (With Cache)",
            shared_prefix_prompts,
            enable_lmcache=True
        )

        # Stop server
        self.stop_server()

        # Final comparison
        self.print_comparison(without_cache_1, with_cache_1, without_cache_2, with_cache_2)

        # Save results
        self.save_results({
            'test1_without_cache': without_cache_1,
            'test1_with_cache': with_cache_1,
            'test2_without_cache': without_cache_2,
            'test2_with_cache': with_cache_2,
        })

    def print_comparison(self, no_cache_1, cache_1, no_cache_2, cache_2):
        """Print final comparison"""
        print("\n" + "="*70)
        print("  PERFORMANCE COMPARISON SUMMARY")
        print("="*70)

        if no_cache_1 and cache_1:
            print("\nTest 1: Repeated Identical Prompts")
            print(f"  Without LMCache:")
            print(f"    First:  {no_cache_1['latency_first_ms']:.0f} ms")
            print(f"    Average: {no_cache_1['latency_avg_ms']:.0f} ms")
            print(f"    Total:   {no_cache_1['total_latency_ms']/1000:.1f} s")

            print(f"\n  With LMCache:")
            print(f"    First:  {cache_1['latency_first_ms']:.0f} ms")
            print(f"    Average: {cache_1['latency_avg_ms']:.0f} ms")
            print(f"    Total:   {cache_1['total_latency_ms']/1000:.1f} s")

            speedup_avg = no_cache_1['latency_avg_ms'] / cache_1['latency_avg_ms']
            speedup_total = no_cache_1['total_latency_ms'] / cache_1['total_latency_ms']

            print(f"\n  🚀 Speedup:")
            print(f"    Average latency: {speedup_avg:.2f}x faster")
            print(f"    Total time:      {speedup_total:.2f}x faster")

        if no_cache_2 and cache_2:
            print("\n" + "-"*70)
            print("\nTest 2: Shared Prefix Prompts")
            print(f"  Without LMCache:")
            print(f"    First:  {no_cache_2['latency_first_ms']:.0f} ms")
            print(f"    Average: {no_cache_2['latency_avg_ms']:.0f} ms")
            print(f"    Total:   {no_cache_2['total_latency_ms']/1000:.1f} s")

            print(f"\n  With LMCache:")
            print(f"    First:  {cache_2['latency_first_ms']:.0f} ms")
            print(f"    Average: {cache_2['latency_avg_ms']:.0f} ms")
            print(f"    Total:   {cache_2['total_latency_ms']/1000:.1f} s")

            speedup_avg = no_cache_2['latency_avg_ms'] / cache_2['latency_avg_ms']
            speedup_total = no_cache_2['total_latency_ms'] / cache_2['total_latency_ms']

            print(f"\n  🚀 Speedup:")
            print(f"    Average latency: {speedup_avg:.2f}x faster")
            print(f"    Total time:      {speedup_total:.2f}x faster")

        print("\n" + "="*70)

    def save_results(self, results: Dict):
        """Save benchmark results"""
        filename = "lmcache_benchmark_results.json"
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model': os.path.basename(self.model_path),
                'results': results
            }, f, indent=2)
        print(f"\n✓ Results saved to {filename}")


async def main():
    """Main benchmark runner"""
    import argparse

    parser = argparse.ArgumentParser(description="LMCache Performance Benchmark")
    parser.add_argument(
        "--model",
        default="/Users/lisihao/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf",
        help="Path to GGUF model file"
    )

    args = parser.parse_args()

    benchmark = LMCacheBenchmark(model_path=args.model)

    try:
        await benchmark.run_all_benchmarks()
    finally:
        benchmark.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
