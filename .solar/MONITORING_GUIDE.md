# KV Cache 量化测试 - 监控指南

## 📊 监控工具

### 1. 快速检查（推荐）
```bash
cd /Users/lisihao/ThunderLLAMA
./.solar/check-kv-test-progress.sh
```

**输出内容**：
- 测试运行状态
- 当前进度（x/15）
- 各配置完成情况（f16, q8_0, q4_0）
- 最新日志（15行）

---

### 2. 持续监控（自动刷新）
```bash
cd /Users/lisihao/ThunderLLAMA
./.solar/watch-kv-test.sh
```

**特点**：
- 每 30 秒自动刷新
- 测试完成后自动生成最终报告
- 按 Ctrl+C 停止监控

---

### 3. 实时日志
```bash
tail -f /Users/lisihao/ThunderLLAMA/.solar/kv-quant-test.log
```

**查看**：
- 详细的测试过程
- 每次运行的完整日志
- 按 Ctrl+C 退出

---

### 4. 查看结果文件
```bash
ls -lh /Users/lisihao/ThunderLLAMA/.solar/kv-quant-results/
```

**文件说明**：
- `f16_run*.json` - f16 配置的测试结果
- `q8_0_run*.json` - q8_0 配置的测试结果
- `q4_0_run*.json` - q4_0 配置的测试结果
- `summary_*.json` - 汇总报告

---

### 5. 查看汇总报告
```bash
cat /Users/lisihao/ThunderLLAMA/.solar/kv-quant-results/summary_*.json | jq .
```

或者运行后处理脚本：
```bash
python3 /Users/lisihao/ThunderLLAMA/.solar/postprocess-kv-results.py
```

---

## 🎯 快捷命令

```bash
# 进入项目目录
cd /Users/lisihao/ThunderLLAMA

# 方式 1: 一次性检查
./.solar/check-kv-test-progress.sh

# 方式 2: 持续监控（每 30 秒刷新）
./.solar/watch-kv-test.sh

# 方式 3: 实时日志
tail -f .solar/kv-quant-test.log

# 查看当前进度百分比
ls -1 .solar/kv-quant-results/*.json 2>/dev/null | wc -l | awk '{print ($1-0) "/" 15 " (" int($1*100/15) "%)"}'
```

---

## 📈 进度计算

- **总测试次数**: 15 (3 配置 × 5 次)
- **已完成**: 根据 `*.json` 文件数量
- **进度百分比**: (已完成 / 15) × 100%

---

## ⏱️ 预计时间

- **每次测试**: 约 2-3 分钟
- **总时间**: 约 30-45 分钟
- **剩余时间**: (15 - 已完成) × 2.5 分钟

---

## 🔍 故障排查

### 测试卡住不动？
```bash
# 检查测试进程
ps aux | grep "test-kv-cache-quantization.sh"

# 检查服务器状态
curl -s http://localhost:30000/health | jq .

# 查看错误日志
tail -50 .solar/kv-quant-test.log | grep -i error
```

### 测试失败？
```bash
# 查看详细日志
cat .solar/kv-quant-test.log | less

# 查看服务器日志
tail -100 /tmp/llama-server-30b.log
```

### 重新运行测试？
```bash
# 停止当前测试
pkill -f "test-kv-cache-quantization.sh"

# 清理结果
rm -rf .solar/kv-quant-results/*

# 重新运行
./.solar/test-kv-cache-quantization.sh > .solar/kv-quant-test.log 2>&1 &
```
