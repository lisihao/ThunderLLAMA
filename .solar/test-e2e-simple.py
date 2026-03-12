#!/usr/bin/env python3
"""简单的端到端性能测试"""

import requests
import time
import json
import sys

PORT = 8090
URL = f"http://localhost:{PORT}/v1/chat/completions"

def test_performance(prompt_length_factor=50, max_tokens=128, test_name="Test"):
    """
    测试端到端性能

    Args:
        prompt_length_factor: 提示词重复次数（控制长度）
        max_tokens: 生成 token 数
        test_name: 测试名称
    """
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")

    # 构造提示词
    base_prompt = "What is the capital of France? "
    user_content = base_prompt * prompt_length_factor

    payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }

    print(f"Prompt repeat factor: {prompt_length_factor}")
    print(f"Max tokens: {max_tokens}")
    print(f"Sending request...")

    # 发送请求并计时
    start_time = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=60)
        end_time = time.time()

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None

        data = response.json()

        # 提取指标
        prompt_tokens = data['usage']['prompt_tokens']
        completion_tokens = data['usage']['completion_tokens']
        total_time = end_time - start_time

        # 计算 tok/s
        tok_per_sec = completion_tokens / total_time if total_time > 0 else 0

        # 估算 TTFT
        # 假设生成是均匀的，TTFT = total_time - (completion_tokens / tok_per_sec)
        generation_time = completion_tokens / tok_per_sec if tok_per_sec > 0 else total_time
        ttft = total_time - generation_time

        # 如果 TTFT 为负（不太可能），说明生成非常快，设为一个小值
        if ttft < 0:
            ttft = 0.001

        print(f"\n📊 Results:")
        print(f"  Prompt tokens:     {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  Total time:        {total_time:.3f}s")
        print(f"  ⏱️  TTFT:            {ttft:.3f}s")
        print(f"  🚀 Tok/s:           {tok_per_sec:.2f} tokens/s")

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_time": total_time,
            "ttft": ttft,
            "tok_per_sec": tok_per_sec
        }

    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    print("\n🔥 ThunderLLAMA Baseline 端到端性能测试")
    print("="*60)
    print("配置: Baseline (无优化)")
    print("  - GPU Layers: 99")
    print("  - Flash Attention: auto")
    print("  - Paged Attention: ❌ 禁用")
    print("  - KV Cache: f16 (无量化)")
    print("  - Continuous Batching: ❌ 禁用")
    print("  - Slots: 1")

    # 测试 1: 短提示
    result1 = test_performance(
        prompt_length_factor=50,
        max_tokens=128,
        test_name="Test 1: 短提示 (~512 tokens prompt, 128 tokens generation)"
    )

    # 测试 2: 长提示
    result2 = test_performance(
        prompt_length_factor=200,
        max_tokens=256,
        test_name="Test 2: 长提示 (~2048 tokens prompt, 256 tokens generation)"
    )

    # 汇总结果
    print(f"\n{'='*60}")
    print("📊 Baseline 性能汇总")
    print(f"{'='*60}")

    if result1:
        print(f"\n短提示:")
        print(f"  TTFT: {result1['ttft']:.3f}s")
        print(f"  Tok/s: {result1['tok_per_sec']:.2f}")

    if result2:
        print(f"\n长提示:")
        print(f"  TTFT: {result2['ttft']:.3f}s")
        print(f"  Tok/s: {result2['tok_per_sec']:.2f}")

    # 保存结果
    results = {
        "config": "baseline",
        "model": "Qwen3-1.7B-Q8_0",
        "optimization": {
            "paged_attention": False,
            "flash_attention": "auto",
            "kv_cache_type": "f16",
            "continuous_batching": False,
            "slots": 1
        },
        "test_1_short_prompt": result1,
        "test_2_long_prompt": result2
    }

    output_file = "/tmp/thunderllama-baseline-e2e.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
