#!/usr/bin/env python3
"""30B 模型端到端性能测试 - Configuration A 优化版"""

import requests
import time
import json

PORT = 8090
URL = f"http://localhost:{PORT}/completion"

def test_performance(prompt, n_predict=128, test_name="Test"):
    """测试端到端性能"""
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")

    print(f"Prompt: {prompt[:50]}...")
    print(f"Tokens to generate: {n_predict}")
    print(f"Sending request...")

    start_time = time.time()
    try:
        response = requests.post(URL, json={
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "stream": False
        }, timeout=120)

        end_time = time.time()

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            return None

        data = response.json()
        content = data['content']
        tokens_predicted = data['tokens_predicted']
        total_time = end_time - start_time
        tok_per_sec = tokens_predicted / total_time if total_time > 0 else 0

        print(f"\n📊 Results:")
        print(f"  Generated tokens: {tokens_predicted}")
        print(f"  Total time:       {total_time:.3f}s")
        print(f"  🚀 Tok/s:         {tok_per_sec:.2f}")
        print(f"\n生成文本预览:")
        print(f"  {repr(content[:100])}...")

        # 检查输出质量
        if not content.strip():
            print("  ❌ 输出为空")
            return None
        elif '\ufffd' in content or content.count('�') > 3:
            print("  ❌ 有乱码")
            return None
        else:
            print(f"  ✅ 输出正常 ({len(content)} 字符)")

        return {
            "tokens_predicted": tokens_predicted,
            "total_time": total_time,
            "tok_per_sec": tok_per_sec,
            "content_preview": content[:200]
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    print("\n🔥 ThunderLLAMA Optimized 端到端性能测试 - 30B 模型")
    print("="*60)
    print("模型: Qwen3-30B-A3B-128K-Q5_K_M")
    print("Endpoint: /completion")
    print("配置: Configuration A (生产环境标配)")
    print("  - Paged Attention: ✅ 启用")
    print("  - Flash Attention: on")
    print("  - KV Cache: q8_0 (内存 ↓50%)")
    print("  - Continuous Batching: ✅ 启用")
    print("  - Slots: 8")
    print("  - Cache RAM: 4096 MB")
    print("  - Prompt Reuse: auto")

    # 测试 1: 短提示
    prompt1 = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
    result1 = test_performance(
        prompt=prompt1,
        n_predict=128,
        test_name="Test 1: 短提示 (短 prompt, 128 tokens generation)"
    )

    # 测试 2: 长提示
    prompt2 = "Explain artificial intelligence in detail. " * 50  # ~500-600 tokens
    result2 = test_performance(
        prompt=prompt2,
        n_predict=256,
        test_name="Test 2: 长提示 (长 prompt, 256 tokens generation)"
    )

    # 汇总
    print(f"\n{'='*60}")
    print("📊 Optimized 性能汇总 (30B 模型)")
    print(f"{'='*60}")

    if result1:
        print(f"\n短提示:")
        print(f"  Tok/s: {result1['tok_per_sec']:.2f}")
        print(f"  Total time: {result1['total_time']:.3f}s")

    if result2:
        print(f"\n长提示:")
        print(f"  Tok/s: {result2['tok_per_sec']:.2f}")
        print(f"  Total time: {result2['total_time']:.3f}s")

    # 保存
    results = {
        "config": "optimized",
        "model": "Qwen3-30B-A3B-128K-Q5_K_M",
        "endpoint": "completion",
        "optimizations": {
            "paged_attention": True,
            "flash_attention": "on",
            "kv_cache_type": "q8_0",
            "continuous_batching": True,
            "slots": 8,
            "cache_ram_mb": 4096,
            "prompt_reuse": "auto"
        },
        "test_1": result1,
        "test_2": result2
    }

    output_file = "/tmp/thunderllama-optimized-30b-completion.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
