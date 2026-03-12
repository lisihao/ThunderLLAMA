#!/usr/bin/env python3
"""Context Shift Summarizer - 两阶段 LLM 摘要压缩

集成 CPUSummarizer 到 ThunderLLAMA，替换简单的字符截断。

阶段1 (0.6B @ port 18083): 抽取 FACTS/DECISIONS/OPEN_ISSUES
阶段2 (1.7B @ port 18084): 压缩成短记忆，添加 GOAL
"""

import time
from typing import Any, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore


class ContextShiftSummarizer:
    """两阶段对话摘要器 (0.6B 抽取 + 0.6B/1.7B 压缩)"""

    def __init__(
        self,
        stage1_endpoint: str = "http://127.0.0.1:18083/completion",
        stage2_endpoint: str = "http://127.0.0.1:18084/completion",
        enabled: bool = True,
        fallback_to_simple: bool = True,
        use_dual_model: bool = True,  # True: 0.6B+1.7B, False: 0.6B+0.6B
    ):
        self.stage1_endpoint = stage1_endpoint  # 0.6B - 抽取
        self.stage2_endpoint = stage2_endpoint if use_dual_model else stage1_endpoint  # 压缩
        self.enabled = enabled
        self.fallback_to_simple = fallback_to_simple
        self.use_dual_model = use_dual_model

        # 检查 requests 是否可用
        if requests is None:
            print("⚠️  requests 库未安装，Context Shift 摘要功能禁用")
            self.enabled = False

    def summarize_messages(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int = 200
    ) -> Optional[str]:
        """
        对消息列表进行两阶段摘要

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            target_tokens: 目标摘要长度（tokens）

        Returns:
            摘要文本，如果失败返回 None
        """
        if not self.enabled:
            return None

        if not messages:
            return None

        # 构建对话文本
        conversation = "\n\n".join([
            f"{msg.get('role', 'user')}: {self._flatten_content(msg.get('content', ''))}"
            for msg in messages
        ])

        try:
            # 阶段1：抽取事实、决策、问题
            raw_extract = self._extract_raw(conversation)
            if not raw_extract or raw_extract.startswith("ERROR:"):
                if self.fallback_to_simple:
                    return None  # Fallback to simple summarization
                return raw_extract

            # 阶段2：压缩成短记忆，添加 GOAL
            final_memory = self._compress_to_memory(raw_extract)
            if not final_memory or final_memory.startswith("ERROR:"):
                # 降级：返回阶段1结果
                return raw_extract

            return final_memory

        except Exception as e:
            print(f"⚠️  Context Shift 摘要失败: {e}")
            if self.fallback_to_simple:
                return None  # Fallback
            return f"ERROR: {str(e)}"

    def _extract_raw(self, conversation: str) -> str:
        """阶段1：抽取事实、决策、问题（0.6B 模型）"""
        prompt = f"""Extract only the key information from this conversation.

Do not summarize. Just list the facts, decisions, and issues.

Format:

[FACTS]
- fact 1
- fact 2

[DECISIONS]
- decision 1
- decision 2

[OPEN_ISSUES]
- issue 1
- issue 2

<END_EXTRACT>

Conversation:
{conversation}

Extracted:
"""
        sampling_params = {
            "prompt": prompt,
            "n_predict": 150,
            "temperature": 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "stop": ["<END_EXTRACT>", "Conversation:", "User:", "Assistant:"]
        }

        try:
            response = requests.post(
                self.stage1_endpoint,
                json=sampling_params,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", "").strip()

        except requests.exceptions.RequestException as e:
            return f"ERROR: {str(e)}"

    def _compress_to_memory(self, raw_extract: str) -> str:
        """阶段2：压缩成短记忆，添加 GOAL（1.7B 模型）"""
        prompt = f"""Add a GOAL summary to this extracted information.

Keep all facts, decisions, and issues. Just add a GOAL at the beginning.

Extracted information:
{raw_extract}

Output with GOAL:

[GOAL]
- (write a brief summary of what user wants)

{raw_extract}

<END_SUMMARY>

Output:
"""
        sampling_params = {
            "prompt": prompt,
            "n_predict": 180,
            "temperature": 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "stop": ["<END_SUMMARY>", "Extracted information:"]
        }

        try:
            response = requests.post(
                self.stage2_endpoint,
                json=sampling_params,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", "").strip()

        except requests.exceptions.RequestException as e:
            return f"ERROR: {str(e)}"

    def _flatten_content(self, content: Any) -> str:
        """展平内容（处理字符串或列表）"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    txt = c.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
                elif isinstance(c, str):
                    parts.append(c)
            return "\n".join(parts)
        return ""

    def health_check(self) -> tuple[bool, str]:
        """
        检查两个摘要服务是否就绪

        Returns:
            (is_healthy, message)
        """
        if not self.enabled:
            return (False, "Context Shift 摘要功能禁用")

        try:
            # 检查阶段1服务
            resp1 = requests.get(
                self.stage1_endpoint.replace("/completion", "/health"),
                timeout=5
            )
            if resp1.status_code != 200:
                return (False, f"阶段1服务 (0.6B) 未就绪: {resp1.status_code}")

            # 检查阶段2服务
            resp2 = requests.get(
                self.stage2_endpoint.replace("/completion", "/health"),
                timeout=5
            )
            if resp2.status_code != 200:
                return (False, f"阶段2服务 (1.7B) 未就绪: {resp2.status_code}")

            return (True, "两个摘要服务都就绪")

        except Exception as e:
            return (False, f"健康检查失败: {str(e)}")


# 简单的 fallback 摘要函数（字符截断）
def simple_compact_history(messages: list[dict[str, Any]], max_lines: int = 12) -> str:
    """简单压缩历史（字符截断，与原 compact_history_tail 兼容）"""
    lines: list[str] = []
    for m in messages[-24:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            txt = content.strip()
        else:
            txt = str(content)
        if not txt:
            continue
        lines.append(f"- {role}: {txt[:140]}")
    if not lines:
        return ""
    return "History tail summary:\n" + "\n".join(lines[-max_lines:])
