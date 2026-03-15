#!/usr/bin/env python3
"""
KV Cache 量化测试结果后处理脚本
从已有结果文件中计算性能指标，并尝试从 llama-server 日志中提取精确数据
"""

import json
import re
import glob
from pathlib import Path
from statistics import mean, stdev
from datetime import datetime

RESULTS_DIR = Path("/Users/lisihao/ThunderLLAMA/.solar/kv-quant-results")
LLAMA_LOG = Path("/tmp/llama-server-30b.log")

def extract_perf_from_log(timestamp_str):
    """
    尝试从 llama-server 日志中提取性能数据
    注意：日志会被覆盖，所以只能提取最近的测试数据
    """
    if not LLAMA_LOG.exists():
        return None, None, None

    try:
        with open(LLAMA_LOG) as f:
            log_content = f.read()

        # 查找最后一次 eval 的性能数据
        # prompt eval time =      48.05 ms /     1 tokens (   48.05 ms per token,    20.81 tokens per second)
        # eval time =     161.00 ms /    10 tokens (   16.10 ms per token,    62.11 tokens per second)

        pp_match = re.search(r'prompt eval time.*?(\d+\.\d+) tokens per second', log_content)
        tg_match = re.search(r'(?<!prompt )eval time.*?(\d+\.\d+) tokens per second', log_content)
        kv_match = re.search(r'llama_kv_cache: size = (\d+\.\d+ MiB)', log_content)

        pp_speed = float(pp_match.group(1)) if pp_match else None
        tg_speed = float(tg_match.group(1)) if tg_match else None
        kv_size = kv_match.group(1) if kv_match else None

        return tg_speed, pp_speed, kv_size
    except Exception as e:
        print(f"⚠️  从日志提取失败: {e}")
        return None, None, None

def estimate_perf_from_result(result):
    """
    从结果数据估算性能（作为后备方案）
    """
    elapsed = result.get("elapsed_time", 0)
    prompt_tokens = result.get("prompt_tokens", 0)
    completion_tokens = result.get("completion_tokens", 0)

    if elapsed > 0:
        # 粗略估算（包含网络延迟等开销）
        estimated_pp = prompt_tokens / elapsed if prompt_tokens > 0 else 0
        estimated_tg = completion_tokens / elapsed if completion_tokens > 0 else 0
        return estimated_tg, estimated_pp

    return None, None

def process_results():
    """处理所有结果文件"""

    print("="*60)
    print("KV Cache 量化测试 - 结果后处理")
    print("="*60)
    print()

    results = {"f16": [], "q8_0": [], "q4_0": []}
    updated_count = 0

    # 读取所有结果文件
    for result_file in sorted(RESULTS_DIR.glob("*.json")):
        if result_file.name.startswith("summary_"):
            continue

        with open(result_file) as f:
            data = json.load(f)

        quant = data["quant_level"]
        run_num = data["run_num"]

        # 检查是否需要更新
        needs_update = False
        if not data.get("tg_speed") or not data.get("pp_speed"):
            needs_update = True

            # 尝试从日志提取（只对最新的测试有效）
            tg_log, pp_log, kv_log = extract_perf_from_log(data["timestamp"])

            # 如果日志提取失败，使用估算值
            if tg_log is None or pp_log is None:
                tg_est, pp_est = estimate_perf_from_result(data)
                data["tg_speed"] = tg_est
                data["pp_speed"] = pp_est
                data["perf_source"] = "estimated"
                print(f"📊 [{quant}] Run {run_num}: 使用估算值 (TG={tg_est:.2f}, PP={pp_est:.2f})")
            else:
                data["tg_speed"] = tg_log
                data["pp_speed"] = pp_log
                data["kv_cache_size"] = kv_log or data.get("kv_cache_size", "")
                data["perf_source"] = "log"
                print(f"✅ [{quant}] Run {run_num}: 从日志提取 (TG={tg_log:.2f}, PP={pp_log:.2f})")

            # 保存更新后的文件
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            updated_count += 1
        else:
            print(f"✓  [{quant}] Run {run_num}: 已有数据，跳过")

        results[quant].append(data)

    print()
    print(f"✅ 更新了 {updated_count} 个结果文件")
    print()

    return results

def generate_summary(results):
    """生成汇总报告"""

    print("="*60)
    print("汇总统计")
    print("="*60)
    print()

    report = {}

    for quant in ["f16", "q8_0", "q4_0"]:
        runs = results[quant]
        if not runs:
            continue

        tg_speeds = [r["tg_speed"] for r in runs if r.get("tg_speed")]
        pp_speeds = [r["pp_speed"] for r in runs if r.get("pp_speed")]

        if tg_speeds and pp_speeds:
            report[quant] = {
                "n_runs": len(runs),
                "tg_mean": mean(tg_speeds),
                "tg_std": stdev(tg_speeds) if len(tg_speeds) > 1 else 0,
                "pp_mean": mean(pp_speeds),
                "pp_std": stdev(pp_speeds) if len(pp_speeds) > 1 else 0,
                "kv_cache_size": runs[0].get("kv_cache_size", "N/A"),
                "l2_hit_rate": runs[0].get("l2_hit_rate", 0),
                "elapsed_mean": mean([r["elapsed_time"] for r in runs]),
                "sample_output": runs[0].get("output_text", "")[:200]
            }

    # 打印表格
    print(f"{'配置':<10} {'TG (tok/s)':<20} {'PP (tok/s)':<20} {'响应时间 (s)':<15} {'L2 Hit'}")
    print("-"*90)

    for quant in ["f16", "q8_0", "q4_0"]:
        if quant not in report:
            print(f"{quant:<10} {'未完成':<20}")
            continue

        r = report[quant]
        tg_str = f"{r['tg_mean']:.2f} ± {r['tg_std']:.2f}"
        pp_str = f"{r['pp_mean']:.2f} ± {r['pp_std']:.2f}"
        elapsed_str = f"{r['elapsed_mean']:.2f}"

        print(f"{quant:<10} {tg_str:<20} {pp_str:<20} {elapsed_str:<15} {r['l2_hit_rate']:.1%}")

    print()

    # 保存汇总报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = RESULTS_DIR / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 汇总报告已保存: {summary_file}")
    print()

    return report

def main():
    if not RESULTS_DIR.exists():
        print(f"❌ 结果目录不存在: {RESULTS_DIR}")
        return

    # 处理结果
    results = process_results()

    # 生成汇总
    if any(results.values()):
        generate_summary(results)
    else:
        print("⚠️  未找到任何结果文件")

if __name__ == "__main__":
    main()
