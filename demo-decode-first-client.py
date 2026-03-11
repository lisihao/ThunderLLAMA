#!/usr/bin/env python3
"""
ThunderLLAMA Decode-First Test Client

Sends a mixed workload (1 long prompt + N short prompts) to measure
TTFT improvement from chunked prefill scheduling.

Usage:
    python3 demo-decode-first-client.py [--port 8080] [--long-tokens 4096] [--short-requests 3] [--delay 0.5]
"""

import argparse
import json
import sys
import threading
import time

import requests


def generate_long_prompt(n_tokens: int) -> str:
    """Generate a long prompt by repeating text. Rough approximation: ~1.3 words per token."""
    base = (
        "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
        "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
        "modern AI were planted by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. This work culminated in the invention of the "
        "programmable digital computer in the 1940s, a machine based on the abstract essence of "
        "mathematical reasoning. This device and the ideas behind it inspired a handful of scientists "
        "to begin seriously discussing the possibility of building an electronic brain. "
    )
    # Repeat to get roughly n_tokens worth of text (~4 chars per token for English)
    target_chars = n_tokens * 4
    repeated = base * (target_chars // len(base) + 1)
    return repeated[:target_chars]


def stream_completion(base_url: str, prompt: str, max_tokens: int, label: str, results: dict):
    """Send a streaming completion request and measure TTFT and TPS."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    t_send = time.perf_counter()
    t_first_token = None
    token_count = 0

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")

            if content and t_first_token is None:
                t_first_token = time.perf_counter()

            if content:
                token_count += 1

    except requests.exceptions.RequestException as e:
        results[label] = {"error": str(e)}
        return

    t_end = time.perf_counter()

    if t_first_token is None:
        results[label] = {"error": "no tokens received"}
        return

    ttft = t_first_token - t_send
    total_time = t_end - t_send
    gen_time = t_end - t_first_token
    tps = token_count / gen_time if gen_time > 0 else 0

    results[label] = {
        "ttft": ttft,
        "total_time": total_time,
        "tokens": token_count,
        "tps": tps,
        "t_send": t_send,
        "t_first": t_first_token,
        "t_end": t_end,
    }


def main():
    parser = argparse.ArgumentParser(description="ThunderLLAMA Decode-First Test Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--long-tokens", type=int, default=4096, help="Approx tokens in long prompt")
    parser.add_argument("--short-requests", type=int, default=3, help="Number of short requests")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay (sec) before sending short requests")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate per request")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Verify server is running
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"❌ Server not healthy: {health.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {base_url}")
        sys.exit(1)

    long_prompt = generate_long_prompt(args.long_tokens)

    print(f"   Long prompt: ~{args.long_tokens} tokens")
    print(f"   Short requests: {args.short_requests}")
    print(f"   Delay before short: {args.delay}s")
    print(f"   Max gen tokens: {args.max_tokens}")
    print()

    results = {}
    threads = []

    # Start long request
    t_start = time.perf_counter()
    long_thread = threading.Thread(
        target=stream_completion,
        args=(base_url, long_prompt, args.max_tokens, "long_0", results),
    )
    long_thread.start()
    threads.append(long_thread)

    # Wait, then send short requests in parallel
    time.sleep(args.delay)

    for i in range(args.short_requests):
        short_prompt = f"What is {i + 1} + {i + 1}? Answer briefly."
        t = threading.Thread(
            target=stream_completion,
            args=(base_url, short_prompt, args.max_tokens, f"short_{i}", results),
        )
        t.start()
        threads.append(t)

    # Wait for all to complete
    for t in threads:
        t.join(timeout=120)

    t_total = time.perf_counter() - t_start

    # ─── Results ───
    print("   ┌──────────┬──────────┬────────┬────────┐")
    print("   │ Request  │ TTFT (s) │ Tokens │ TPS    │")
    print("   ├──────────┼──────────┼────────┼────────┤")

    short_ttfts = []
    for label in sorted(results.keys()):
        r = results[label]
        if "error" in r:
            print(f"   │ {label:8s} │ ERROR    │ {r['error']:15s} │")
        else:
            tag = "LONG" if label.startswith("long") else "short"
            print(f"   │ {tag:8s} │ {r['ttft']:8.3f} │ {r['tokens']:6d} │ {r['tps']:6.1f} │")
            if label.startswith("short"):
                short_ttfts.append(r["ttft"])

    print("   └──────────┴──────────┴────────┴────────┘")

    if short_ttfts:
        avg_ttft = sum(short_ttfts) / len(short_ttfts)
        max_ttft = max(short_ttfts)
        min_ttft = min(short_ttfts)
        print(f"   Short TTFT: avg={avg_ttft:.3f}s  min={min_ttft:.3f}s  max={max_ttft:.3f}s")

    print(f"   Total wall time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
