#!/bin/bash
# Solar 自主优化入口脚本
# 用法: ./demo-auto-optimize.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPT_FILE="$HOME/.claude/prompts/thunderllama-optimize.md"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 Solar 自主优化系统"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "项目: ThunderLLAMA"
echo "目标: M4 性能优化 >20%"
echo "时间: 15 分钟"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查提示词文件
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ 提示词文件不存在: $PROMPT_FILE"
    exit 1
fi

# 显示提示词模板
echo "📋 提示词模板:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$PROMPT_FILE" | head -20
echo "..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 提示用户输入触发指令
echo "🎯 请在 Claude Code 中输入以下指令："
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Solar，启动 ThunderLLAMA M4 优化流程"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Solar 将自动完成以下步骤:"
echo "  1. ✅ Baseline 性能测试"
echo "  2. 🔍 瓶颈深度分析（审判官）"
echo "  3. ⚡ 优化代码生成（创想家）"
echo "  4. 👁️  代码审查（稳健派）"
echo "  5. 🔨 编译和测试"
echo "  6. 📊 性能对比报告"
echo ""
echo "预计时间: 15 分钟"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
