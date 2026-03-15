#!/bin/bash
# KV Cache 量化对比测试
# 对比 f16, q8_0, q4_0 三种量化级别的性能和质量

set -e

PROJECT_DIR="/Users/lisihao/ThunderLLAMA"
RESULTS_DIR="$PROJECT_DIR/.solar/kv-quant-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 测试配置
MODEL_PATH="/Users/lisihao/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf"
TEST_PROMPT='量子力学是20世纪物理学最重要的理论之一，彻底改变了我们对微观世界的认识。量子力学的核心概念包括波粒二象性、不确定性原理、量子叠加、量子纠缠和能量量子化。波粒二象性由德布罗意首先提出，后来在双缝干涉实验中得到验证。不确定性原理由海森堡于1927年提出，指出我们无法同时精确测量粒子的位置和动量。量子叠加是量子力学的基本特性，粒子可以同时处于多个状态的叠加，薛定谔的猫思想实验展示了这一概念。量子纠缠是指两个或多个粒子之间存在非局域关联，即使相距遥远，一个粒子的状态改变会瞬间影响另一个粒子。能量量子化是普朗克为解释黑体辐射问题而提出的假设，后来成为量子力学的基石。请详细解释薛定谔方程的物理意义。'

MAX_TOKENS=100
N_RUNS=5

echo "========================================"
echo "KV Cache 量化对比测试"
echo "========================================"
echo "模型: Qwen3-30B-Q5_K_M"
echo "测试维度: f16, q8_0, q4_0"
echo "重复次数: $N_RUNS"
echo "时间戳: $TIMESTAMP"
echo "========================================"
echo ""

# 测试函数
test_kv_quant() {
    local quant_level=$1
    local run_num=$2

    echo "[$quant_level] Run $run_num/$N_RUNS - 开始测试..."

    # 1. 停止服务器
    cd "$PROJECT_DIR"
    ./stop-thunderllama.sh > /dev/null 2>&1 || true
    sleep 2

    # 2. 修改配置文件
    sed -i.bak "s/^KV_CACHE_LEVEL=.*/KV_CACHE_LEVEL=\"$quant_level\"/" thunderllama.conf
    sed -i.bak "s|^MODEL_PATH=.*|MODEL_PATH=\"$MODEL_PATH\"|" thunderllama.conf

    # 3. 启动服务器
    echo "[$quant_level] Run $run_num - 启动服务器..."
    ./start-thunderllama.sh > /dev/null 2>&1
    sleep 10

    # 4. 健康检查
    local health_status=$(curl -s http://localhost:30000/health | jq -r '.status')
    if [ "$health_status" != "ok" ]; then
        echo "❌ [$quant_level] Run $run_num - 服务器启动失败"
        return 1
    fi

    # 5. Warmup（预热）
    echo "[$quant_level] Run $run_num - Warmup..."
    curl -s http://localhost:30000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":\"你好\",\"max_tokens\":10,\"temperature\":0.7}" > /dev/null
    sleep 2

    # 6. 执行测试请求
    echo "[$quant_level] Run $run_num - 执行测试..."
    local start_time=$(date +%s.%N)

    local response=$(curl -s http://localhost:30000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$TEST_PROMPT\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0.7}")

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)

    # 7. 解析结果
    local output_text=$(echo "$response" | jq -r '.choices[0].text')
    local prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens')
    local completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens')

    # 8. 获取统计
    local stats=$(curl -s http://localhost:30000/lmcache/stats)
    local l2_hit_rate=$(echo "$stats" | jq -r '.l2_hit_rate')
    local l2_chunks=$(echo "$stats" | jq -r '.l2_chunks')

    # 9. 从日志获取性能数据
    local tg_speed=$(tail -200 /tmp/llama-server-30b.log | grep "eval time" | tail -1 | grep -oE "[0-9]+\.[0-9]+ ms / [0-9]+ tokens" | awk '{print 1000/$1}')
    local pp_speed=$(tail -200 /tmp/llama-server-30b.log | grep "prompt eval time" | tail -1 | grep -oE "[0-9]+\.[0-9]+ ms / [0-9]+ tokens" | awk '{print 1000/$1}')

    # 10. 获取 KV Cache 内存占用
    local kv_size=$(tail -100 /tmp/llama-server-30b.log | grep "llama_kv_cache: size" | tail -1 | grep -oE "[0-9]+\.[0-9]+ MiB")

    # 11. 保存结果
    local result_file="$RESULTS_DIR/${quant_level}_run${run_num}_${TIMESTAMP}.json"
    cat > "$result_file" <<EOF
{
  "quant_level": "$quant_level",
  "run_num": $run_num,
  "timestamp": "$(date -Iseconds)",
  "elapsed_time": $elapsed,
  "prompt_tokens": $prompt_tokens,
  "completion_tokens": $completion_tokens,
  "tg_speed": $tg_speed,
  "pp_speed": $pp_speed,
  "kv_cache_size": "$kv_size",
  "l2_hit_rate": $l2_hit_rate,
  "l2_chunks": $l2_chunks,
  "output_text": $(echo "$output_text" | jq -Rs .)
}
EOF

    echo "[$quant_level] Run $run_num - 完成 (TG: $tg_speed tok/s, PP: $pp_speed tok/s, KV: $kv_size)"
    echo ""
}

# 主测试循环
for quant in "f16" "q8_0" "q4_0"; do
    echo "========================================"
    echo "测试 KV Cache $quant"
    echo "========================================"

    for run in $(seq 1 $N_RUNS); do
        test_kv_quant "$quant" "$run"
    done

    echo ""
done

# 停止服务器
cd "$PROJECT_DIR"
./stop-thunderllama.sh

# 生成汇总报告
echo "========================================"
echo "生成汇总报告..."
echo "========================================"

python3 - <<PYTHON
import json
import glob
from pathlib import Path
from statistics import mean, stdev

results_dir = Path("$RESULTS_DIR")
results = {"f16": [], "q8_0": [], "q4_0": []}

# 读取所有结果
for result_file in results_dir.glob("*_${TIMESTAMP}.json"):
    with open(result_file) as f:
        data = json.load(f)
        quant = data["quant_level"]
        results[quant].append(data)

# 计算统计
report = {}
for quant in ["f16", "q8_0", "q4_0"]:
    runs = results[quant]
    if not runs:
        continue

    tg_speeds = [float(r["tg_speed"]) for r in runs if r["tg_speed"]]
    pp_speeds = [float(r["pp_speed"]) for r in runs if r["pp_speed"]]

    report[quant] = {
        "n_runs": len(runs),
        "tg_mean": mean(tg_speeds) if tg_speeds else 0,
        "tg_std": stdev(tg_speeds) if len(tg_speeds) > 1 else 0,
        "pp_mean": mean(pp_speeds) if pp_speeds else 0,
        "pp_std": stdev(pp_speeds) if len(pp_speeds) > 1 else 0,
        "kv_cache_size": runs[0]["kv_cache_size"] if runs else "N/A",
        "l2_hit_rate": runs[0]["l2_hit_rate"] if runs else 0,
        "sample_output": runs[0]["output_text"][:200] if runs else ""
    }

# 保存报告
report_file = results_dir / f"summary_{TIMESTAMP}.json"
with open(report_file, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# 打印报告
print("\n" + "="*60)
print("KV Cache 量化对比测试 - 汇总报告")
print("="*60)
print(f"\n模型: Qwen3-30B-Q5_K_M")
print(f"每个配置运行次数: $N_RUNS")
print(f"\n{'配置':<10} {'TG (tok/s)':<20} {'PP (tok/s)':<20} {'KV Cache':<15} {'L2 Hit Rate'}")
print("-"*90)

for quant in ["f16", "q8_0", "q4_0"]:
    r = report[quant]
    tg_str = f"{r['tg_mean']:.2f} ± {r['tg_std']:.2f}"
    pp_str = f"{r['pp_mean']:.2f} ± {r['pp_std']:.2f}"
    print(f"{quant:<10} {tg_str:<20} {pp_str:<20} {r['kv_cache_size']:<15} {r['l2_hit_rate']:.2%}")

print("\n" + "="*60)
print(f"详细结果保存在: {results_dir}")
print(f"汇总报告: {report_file}")
print("="*60)

# 输出文本质量对比
print("\n\n" + "="*60)
print("输出文本质量对比（前 200 字符）")
print("="*60)
for quant in ["f16", "q8_0", "q4_0"]:
    print(f"\n[{quant}]")
    print(report[quant]["sample_output"])
    print()
PYTHON

echo ""
echo "✅ 测试完成！"
