#!/usr/bin/env python3
"""
Proper LMCache Benchmark - Realistic Multi-Agent Scenario

Key differences from previous test:
1. Long prompts (2000+ tokens) - tool definitions + system prompt
2. More requests (20+ rounds)
3. Trigger L2 → L3 eviction
"""

import asyncio
import time
import statistics
import httpx

# 模拟 OpenClaw 的工具定义（~1500 tokens）
TOOL_DEFINITIONS = """
You are an advanced AI coding assistant with access to the following tools:

## File Operations Tools

### read_file
Read the contents of a file from the filesystem.
Parameters:
- file_path (string, required): Absolute path to the file
- offset (integer, optional): Line number to start reading from
- limit (integer, optional): Number of lines to read
Returns: File contents with line numbers

### write_file
Write content to a file, creating it if it doesn't exist.
Parameters:
- file_path (string, required): Absolute path to the file
- content (string, required): Content to write
Returns: Success confirmation

### edit_file
Edit an existing file by replacing exact string matches.
Parameters:
- file_path (string, required): Absolute path to the file
- old_string (string, required): Exact text to replace
- new_string (string, required): Replacement text
Returns: Edit confirmation

### glob_files
Find files matching a glob pattern.
Parameters:
- pattern (string, required): Glob pattern (e.g., "**/*.py")
- path (string, optional): Directory to search in
Returns: List of matching file paths

### grep_files
Search for text patterns in files using regex.
Parameters:
- pattern (string, required): Regex pattern to search for
- path (string, optional): Directory or file to search
- glob (string, optional): File pattern filter
Returns: List of matches with file paths and line numbers

## Code Analysis Tools

### analyze_code
Perform static analysis on code to detect issues.
Parameters:
- file_path (string, required): Path to code file
- checks (array, optional): Specific checks to run
Returns: Analysis results with issues and suggestions

### run_tests
Execute test suite and return results.
Parameters:
- test_path (string, optional): Path to test file or directory
- pattern (string, optional): Test name pattern
Returns: Test results with pass/fail status

### lint_code
Run linter on code files.
Parameters:
- file_path (string, required): Path to code file
- fix (boolean, optional): Auto-fix issues if possible
Returns: Lint results

## Execution Tools

### bash_command
Execute a bash command in the shell.
Parameters:
- command (string, required): Command to execute
- timeout (integer, optional): Timeout in milliseconds
Returns: Command output (stdout/stderr)

### git_command
Execute git operations.
Parameters:
- operation (string, required): Git operation (commit, push, pull, etc.)
- args (array, optional): Additional arguments
Returns: Git operation result

## AI Tools

### ask_llm
Query another LLM for assistance.
Parameters:
- prompt (string, required): Prompt to send to LLM
- model (string, optional): Model to use
Returns: LLM response

### generate_code
Generate code based on specifications.
Parameters:
- spec (string, required): Code specification
- language (string, required): Programming language
Returns: Generated code

You have access to the full project context and should use these tools effectively to accomplish tasks.

Current project structure:
- /src: Source code files
- /tests: Test files
- /docs: Documentation
- /config: Configuration files

Follow these guidelines:
1. Always read files before editing
2. Run tests after making changes
3. Use appropriate tools for each task
4. Provide clear explanations of your actions
"""

class ProperBenchmark:
    def __init__(self):
        self.base_url = "http://localhost:30000"

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

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                usage = data.get('usage', {})
                return {
                    'latency_ms': latency,
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'cache_hit': data.get('cache_hit', False),  # ThunderLLAMA may report this
                }
            else:
                raise Exception(f"Request failed: {response.status_code}")

    async def run_multi_agent_simulation(self, num_agents: int = 10, num_rounds: int = 3):
        """
        Simulate OpenClaw multi-agent workflow

        Each agent gets:
        - Same tool definitions (共享前缀，触发缓存)
        - Different task (不同后缀)
        - Multiple rounds (触发 L2 → L3 eviction 和回读)
        """
        print(f"\n{'='*70}")
        print(f"  MULTI-AGENT SIMULATION")
        print(f"  Agents: {num_agents} | Rounds: {num_rounds} | Total requests: {num_agents * num_rounds}")
        print(f"{'='*70}\n")

        agent_types = ["reviewer", "architect", "coder", "tester", "ops"]
        all_results = []

        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            round_results = []

            for agent_idx in range(num_agents):
                agent_type = agent_types[agent_idx % len(agent_types)]

                # 构建长提示词：工具定义 + 任务
                # 前缀相同（工具定义），后缀不同（任务）
                prompt = f"""{TOOL_DEFINITIONS}

## Your Role
You are a {agent_type} agent in a multi-agent system.

## Current Task (Round {round_num + 1}, Agent {agent_idx + 1})
Task: Analyze the impact of PR #123 from your perspective as a {agent_type}.
Focus areas: code quality, architecture, testing, deployment.

Provide a brief analysis."""

                print(f"  [{agent_type} #{agent_idx+1}] ", end='', flush=True)

                try:
                    result = await self.run_inference(prompt)
                    round_results.append(result)
                    all_results.append(result)

                    # 简洁输出
                    cache_indicator = "🔥" if result.get('cache_hit') else "❄️"
                    print(f"{cache_indicator} {result['latency_ms']:.0f}ms ({result['prompt_tokens']}t)")

                except Exception as e:
                    print(f"❌ ERROR: {e}")

            # 每轮统计
            if round_results:
                avg_latency = statistics.mean([r['latency_ms'] for r in round_results])
                cache_hits = sum(1 for r in round_results if r.get('cache_hit', False))
                print(f"\n  Round {round_num + 1} summary: Avg {avg_latency:.0f}ms, Cache hits: {cache_hits}/{len(round_results)}")

        # 总体统计
        print(f"\n{'='*70}")
        print(f"  OVERALL STATISTICS")
        print(f"{'='*70}\n")

        if all_results:
            latencies = [r['latency_ms'] for r in all_results]

            # 按轮次分组分析
            requests_per_round = num_agents
            round_1_latencies = latencies[:requests_per_round]
            round_2_latencies = latencies[requests_per_round:2*requests_per_round] if len(latencies) > requests_per_round else []
            round_3_latencies = latencies[2*requests_per_round:3*requests_per_round] if len(latencies) > 2*requests_per_round else []

            print(f"Round 1 (cold cache):")
            print(f"  Average: {statistics.mean(round_1_latencies):.0f} ms")

            if round_2_latencies:
                print(f"\nRound 2 (warm cache):")
                print(f"  Average: {statistics.mean(round_2_latencies):.0f} ms")
                print(f"  Speedup: {statistics.mean(round_1_latencies) / statistics.mean(round_2_latencies):.1f}x")

            if round_3_latencies:
                print(f"\nRound 3 (stable cache):")
                print(f"  Average: {statistics.mean(round_3_latencies):.0f} ms")
                print(f"  Speedup: {statistics.mean(round_1_latencies) / statistics.mean(round_3_latencies):.1f}x")

            print(f"\nOverall:")
            print(f"  Total requests: {len(all_results)}")
            print(f"  Average latency: {statistics.mean(latencies):.0f} ms")
            print(f"  Median latency: {statistics.median(latencies):.0f} ms")
            print(f"  Min/Max: {min(latencies):.0f} / {max(latencies):.0f} ms")

            # 缓存命中率
            cache_hits = sum(1 for r in all_results if r.get('cache_hit', False))
            if cache_hits > 0:
                print(f"  Cache hit rate: {cache_hits}/{len(all_results)} ({cache_hits/len(all_results)*100:.1f}%)")

        print(f"\n{'='*70}\n")

async def main():
    benchmark = ProperBenchmark()

    # 检查服务器
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:30000/health")
            if response.status_code != 200:
                print("❌ Server not running")
                return
    except:
        print("❌ Server not running on port 30000")
        print("\nPlease start ThunderLLAMA server first:")
        print("  cd /Users/lisihao/ThunderLLAMA/build")
        print("  LMCACHE_ENABLED=true ./bin/llama-server --model <model> --port 30000")
        return

    print("✓ Server detected")

    # 运行测试：10 agents × 3 rounds = 30 requests
    await benchmark.run_multi_agent_simulation(num_agents=10, num_rounds=3)

if __name__ == "__main__":
    asyncio.run(main())
