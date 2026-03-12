#!/usr/bin/env python3
"""
Simple LMCache Benchmark - Assumes server is already running

Usage:
1. Start server without LMCache:
   LMCACHE_ENABLED=false ./bin/llama-server --model <model> --port 30000

2. Run this script

3. Stop server, start with LMCache:
   LMCACHE_ENABLED=true THUNDER_LMCACHE_DISK_PATH=~/.openclaw/bench.bin ./bin/llama-server --model <model> --port 30000

4. Run this script again
"""

import asyncio
import time
import statistics
import httpx

class SimpleBenchmark:
    def __init__(self):
        self.base_url = "http://localhost:30000"

    async def check_server(self):
        """Check if server is running"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False

    async def run_inference(self, prompt: str) -> dict:
        """Single inference request"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            start = time.time()

            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.7,
                }
            )

            latency = (time.time() - start) * 1000  # ms

            if response.status_code == 200:
                data = response.json()
                usage = data.get('usage', {})
                return {
                    'latency_ms': latency,
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                }
            else:
                raise Exception(f"Request failed: {response.status_code}")

    async def run_test(self, test_name: str, prompts: list):
        """Run test with list of prompts"""
        print(f"\n{'='*70}")
        print(f"  {test_name}")
        print(f"{'='*70}\n")

        results = []
        for i, prompt in enumerate(prompts):
            print(f"Request {i+1}/{len(prompts)}: ", end='', flush=True)

            result = await self.run_inference(prompt)
            results.append(result)

            print(f"{result['latency_ms']:.0f}ms ({result['prompt_tokens']} tokens)")

        # Statistics
        latencies = [r['latency_ms'] for r in results]

        print(f"\n  Summary:")
        print(f"    First (cold):  {latencies[0]:.0f} ms")
        if len(latencies) > 1:
            print(f"    Second (warm): {latencies[1]:.0f} ms - {latencies[0]/latencies[1]:.1f}x faster")
            print(f"    Average 2-5:   {statistics.mean(latencies[1:]):.0f} ms")
        print(f"    Total:         {sum(latencies)/1000:.1f} s")

        return {
            'first_ms': latencies[0],
            'second_ms': latencies[1] if len(latencies) > 1 else 0,
            'avg_warm_ms': statistics.mean(latencies[1:]) if len(latencies) > 1 else 0,
            'total_ms': sum(latencies),
            'speedup': latencies[0] / latencies[1] if len(latencies) > 1 else 1.0,
        }

    async def run_all_tests(self):
        """Run all benchmark tests"""
        print("\n" + "="*70)
        print("  LMCACHE SIMPLE BENCHMARK")
        print("="*70)

        # Check server
        if not await self.check_server():
            print("\n❌ Error: ThunderLLAMA server not running on port 30000")
            print("\nPlease start the server first:")
            print("  cd /Users/lisihao/ThunderLLAMA/build")
            print("  LMCACHE_ENABLED=false ./bin/llama-server --model <model> --port 30000")
            return

        print("✓ Server is running")

        # Test 1: Identical repeated prompts
        test1_prompts = [
            "Explain how photosynthesis works.",
            "Explain how photosynthesis works.",
            "Explain how photosynthesis works.",
            "Explain how photosynthesis works.",
            "Explain how photosynthesis works.",
        ]

        result1 = await self.run_test("TEST 1: Repeated Identical Prompts", test1_prompts)

        # Test 2: Shared prefix
        test2_prompts = [
            "You are a helpful AI.\n\nQuestion: What is machine learning?",
            "You are a helpful AI.\n\nQuestion: What is deep learning?",
            "You are a helpful AI.\n\nQuestion: What is neural network?",
            "You are a helpful AI.\n\nQuestion: What is computer vision?",
            "You are a helpful AI.\n\nQuestion: What is NLP?",
        ]

        result2 = await self.run_test("TEST 2: Shared Prefix Prompts", test2_prompts)

        # Summary
        print(f"\n{'='*70}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*70}")
        print(f"\nTest 1 (Identical Prompts):")
        print(f"  Cold:       {result1['first_ms']:.0f} ms")
        print(f"  Warm avg:   {result1['avg_warm_ms']:.0f} ms")
        print(f"  Speedup:    {result1['speedup']:.1f}x")

        print(f"\nTest 2 (Shared Prefix):")
        print(f"  Cold:       {result2['first_ms']:.0f} ms")
        print(f"  Warm avg:   {result2['avg_warm_ms']:.0f} ms")
        print(f"  Speedup:    {result2['speedup']:.1f}x")

        print(f"\n{'='*70}")
        print("\nNext: Restart server with LMCache enabled and compare!")
        print("  LMCACHE_ENABLED=true THUNDER_LMCACHE_DISK_PATH=~/.openclaw/bench.bin \\")
        print("    ./bin/llama-server --model <model> --port 30000")
        print(f"\n{'='*70}\n")


async def main():
    benchmark = SimpleBenchmark()
    await benchmark.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
