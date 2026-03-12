#!/usr/bin/env python3
"""ThunderLLAMA service layer on top of llama.cpp server.

Key goals:
- Keep llama.cpp (`llama-server`) as the inference core.
- Expose a stable OpenAI-compatible proxy endpoint.
- Provide first-class request-time optimizations:
  - Tiered context layering (must-have / nice-to-have / history-tail)
  - Per-tier token caps
  - Lightweight history summarization
  - Prompt hotspot tracking
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

# 条件导入 Context Shift 摘要器（可能不存在）
try:
    from context_shift_summarizer import ContextShiftSummarizer
    CONTEXT_SHIFT_AVAILABLE = True
except ImportError:
    CONTEXT_SHIFT_AVAILABLE = False

RE_WS = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def trim_text_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    out = text[:max_chars]
    cut = max(out.rfind("\n"), out.rfind(" "))
    if cut > int(max_chars * 0.7):
        out = out[:cut]
    return out.rstrip() + "\n[truncated]"


def flatten_content(content: Any) -> str:
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


def compact_history_tail(messages: list[dict[str, Any]], max_lines: int = 12) -> str:
    lines: list[str] = []
    for m in messages[-24:]:
        role = m.get("role", "user")
        txt = RE_WS.sub(" ", flatten_content(m.get("content", "")).strip())
        if not txt:
            continue
        lines.append(f"- {role}: {txt[:140]}")
    if not lines:
        return ""
    return "History tail summary:\n" + "\n".join(lines[-max_lines:])


@dataclass
class LayeringConfig:
    enabled: bool
    must_have_cap: int
    nice_to_have_cap: int
    history_tail_cap: int
    history_recent_cap: int
    history_archive_cap: int
    preserve_last_turns: int
    dynamic_context_postfix: bool
    shared_prefix_text: str
    context_shift_enabled: bool
    context_shift_stage1_endpoint: str
    context_shift_stage2_endpoint: str
    context_shift_mode: str  # "auto", "fast", "quality", "simple"
    context_shift_auto_threshold: int  # 对话轮数阈值，超过则用质量模式


class PromptHotspotTracker:
    def __init__(self, store_path: str | None = None) -> None:
        self.counter: Counter[str] = Counter()
        self.lock = threading.Lock()
        self.store_path = Path(store_path).expanduser() if store_path else None

    def update_from_messages(self, messages: list[dict[str, Any]]) -> None:
        sigs: list[str] = []
        for m in messages:
            if m.get("role") != "user":
                continue
            txt = RE_WS.sub(" ", flatten_content(m.get("content", "")).strip())
            if not txt:
                continue
            sig = txt[:180]
            sigs.append(sig)
        if not sigs:
            return
        with self.lock:
            self.counter.update(sigs)

    def top(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            return [
                {"prompt_signature": k, "count": v}
                for k, v in self.counter.most_common(limit)
            ]

    def flush(self) -> None:
        if not self.store_path:
            return
        payload = {"updated_at": int(time.time()), "top": self.top(200)}
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class MetricsStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS request_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_unix INTEGER NOT NULL,
                    model TEXT,
                    input_messages INTEGER NOT NULL,
                    output_messages INTEGER NOT NULL,
                    input_token_estimate INTEGER,
                    output_token_estimate INTEGER,
                    must_have_cap INTEGER,
                    nice_to_have_cap INTEGER,
                    history_tail_cap INTEGER,
                    preserve_last_turns INTEGER,
                    must_have_tokens INTEGER,
                    nice_to_have_tokens INTEGER,
                    history_tail_tokens INTEGER,
                    tail_tokens INTEGER,
                    latency_ms INTEGER,
                    upstream_status INTEGER,
                    error TEXT,
                    cache_ram_mb INTEGER,
                    meta_json TEXT
                )
                """
            )
            cols = {
                row[1]
                for row in cur.execute("PRAGMA table_info(request_metrics)").fetchall()
            }
            if "cache_ram_mb" not in cols:
                cur.execute("ALTER TABLE request_metrics ADD COLUMN cache_ram_mb INTEGER")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_request_metrics_ts ON request_metrics(ts_unix)"
            )
            self.conn.commit()

    def insert(self, row: dict[str, Any]) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO request_metrics (
                    ts_unix, model, input_messages, output_messages,
                    input_token_estimate, output_token_estimate,
                    must_have_cap, nice_to_have_cap, history_tail_cap, preserve_last_turns,
                    must_have_tokens, nice_to_have_tokens, history_tail_tokens, tail_tokens,
                    latency_ms, upstream_status, error, cache_ram_mb, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row.get("ts_unix", int(time.time()))),
                    row.get("model"),
                    int(row.get("input_messages", 0)),
                    int(row.get("output_messages", 0)),
                    int(row.get("input_token_estimate", 0)),
                    int(row.get("output_token_estimate", 0)),
                    int(row.get("must_have_cap", 0)),
                    int(row.get("nice_to_have_cap", 0)),
                    int(row.get("history_tail_cap", 0)),
                    int(row.get("preserve_last_turns", 0)),
                    int(row.get("must_have_tokens", 0)),
                    int(row.get("nice_to_have_tokens", 0)),
                    int(row.get("history_tail_tokens", 0)),
                    int(row.get("tail_tokens", 0)),
                    int(row.get("latency_ms", 0)),
                    int(row.get("upstream_status", 0)),
                    row.get("error"),
                    int(row.get("cache_ram_mb", 0)),
                    json.dumps(row.get("meta_json", {}), ensure_ascii=False),
                ),
            )
            self.conn.commit()

    def summarize_recent_cache_ram(
        self,
        lookback_sec: int,
        candidates_mb: list[int],
    ) -> list[dict[str, Any]]:
        since = int(time.time()) - max(60, lookback_sec)
        if not candidates_mb:
            return []
        qmarks = ",".join(["?"] * len(candidates_mb))
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                f"""
                SELECT
                    cache_ram_mb,
                    COUNT(*) AS total,
                    AVG(latency_ms) AS avg_latency_ms,
                    SUM(CASE WHEN upstream_status >= 200 AND upstream_status < 300 THEN 1 ELSE 0 END) AS ok_count
                FROM request_metrics
                WHERE ts_unix >= ?
                  AND cache_ram_mb IN ({qmarks})
                GROUP BY cache_ram_mb
                ORDER BY cache_ram_mb
                """,
                (since, *candidates_mb),
            )
            rows = cur.fetchall()
        horizon_s = max(1, lookback_sec)
        out: list[dict[str, Any]] = []
        for cache_ram_mb, total, avg_latency_ms, ok_count in rows:
            t = int(total or 0)
            ok = int(ok_count or 0)
            out.append(
                {
                    "cache_ram_mb": int(cache_ram_mb or 0),
                    "total": t,
                    "ok_count": ok,
                    "failure_rate": 1.0 - (ok / t if t > 0 else 0.0),
                    "avg_latency_ms": float(avg_latency_ms or 0.0),
                    "throughput_rps": ok / horizon_s,
                }
            )
        return out

    def close(self) -> None:
        with self.lock:
            self.conn.close()


def layer_messages(
    messages: list[dict[str, Any]],
    cfg: LayeringConfig,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not cfg.enabled:
        out = [
            {"role": m.get("role", "user"), "content": flatten_content(m.get("content", ""))}
            for m in messages
            if flatten_content(m.get("content", "")).strip()
        ]
        return out, {"total_output_tokens": sum(estimate_tokens(m["content"]) for m in out)}

    sys_dev = [m for m in messages if m.get("role") in ("system", "developer")]
    convo = [m for m in messages if m.get("role") not in ("system", "developer")]
    preserve_n = max(1, cfg.preserve_last_turns)
    tail = convo[-preserve_n:]
    middle = convo[:-preserve_n] if len(convo) > preserve_n else []

    must_chunks: list[str] = []
    for m in sys_dev:
        txt = flatten_content(m.get("content", "")).strip()
        if txt:
            must_chunks.append(f"[{m.get('role','system')}]\n{txt}")
    must_text = trim_text_tokens("\n\n".join(must_chunks), cfg.must_have_cap)

    nice_chunks: list[str] = []
    for m in middle[-8:]:
        txt = flatten_content(m.get("content", "")).strip()
        if txt:
            nice_chunks.append(f"[{m.get('role','user')}] {txt[:240]}")
    nice_text = trim_text_tokens("\n".join(nice_chunks), cfg.nice_to_have_cap)

    # Two-level history summarization with a hard cap on merged summary tokens.
    # Try Context Shift (LLM summarization) first, fallback to simple compact if unavailable.
    hist_text = ""
    hist_recent = ""
    hist_archive = ""

    if cfg.context_shift_enabled and CONTEXT_SHIFT_AVAILABLE and middle:
        # 动态选择模式：auto / fast / quality / simple
        mode = cfg.context_shift_mode
        use_dual_model = True  # 默认使用双模型（质量模式）

        if mode == "auto":
            # 自动模式：根据对话轮数选择
            conversation_turns = len(middle)
            if conversation_turns < cfg.context_shift_auto_threshold:
                # 短对话 → 快速模式（0.6B + 0.6B）
                use_dual_model = False
                mode_label = "fast"
            else:
                # 长对话 → 质量模式（0.6B + 1.7B）
                use_dual_model = True
                mode_label = "quality"
        elif mode == "fast":
            # 快速模式：0.6B + 0.6B
            use_dual_model = False
            mode_label = "fast"
        elif mode == "quality":
            # 质量模式：0.6B + 1.7B
            use_dual_model = True
            mode_label = "quality"
        else:
            # simple 模式：跳过 LLM 摘要，直接 fallback
            mode_label = "simple"
            # 不执行下面的 LLM 摘要

        if mode != "simple":
            # 尝试使用 Context Shift 两阶段 LLM 摘要
            try:
                summarizer = ContextShiftSummarizer(
                    stage1_endpoint=cfg.context_shift_stage1_endpoint,
                    stage2_endpoint=cfg.context_shift_stage2_endpoint,
                    enabled=True,
                    fallback_to_simple=True,
                    use_dual_model=use_dual_model,
                )
                llm_summary = summarizer.summarize_messages(middle, target_tokens=cfg.history_tail_cap)
                if llm_summary:
                    hist_text = trim_text_tokens(llm_summary, cfg.history_tail_cap)
            except Exception as e:
                print(f"⚠️  Context Shift 摘要失败，fallback 到简单压缩: {e}", flush=True)

    # Fallback: 简单字符截断（兼容原逻辑）
    if not hist_text:
        recent_window = middle[-6:]
        archive_window = middle[:-6] if len(middle) > 6 else []
        hist_recent = trim_text_tokens(
            compact_history_tail(recent_window, max_lines=8),
            cfg.history_recent_cap,
        )
        hist_archive = trim_text_tokens(
            compact_history_tail(archive_window, max_lines=10),
            cfg.history_archive_cap,
        )
        hist_parts: list[str] = []
        if hist_recent:
            hist_parts.append("Recent dialogue summary:\n" + hist_recent)
        if hist_archive:
            hist_parts.append("Archived dialogue summary:\n" + hist_archive)
        hist_text = trim_text_tokens("\n\n".join(hist_parts), cfg.history_tail_cap)

    out: list[dict[str, str]] = []
    shared_prefix = cfg.shared_prefix_text.strip()
    if shared_prefix:
        # Keep this block stable to maximize prefix cache reuse.
        out.append({"role": "system", "content": shared_prefix})

    tail_budget = max(256, cfg.must_have_cap // 2)
    per_msg_cap = max(64, tail_budget // max(1, len(tail)))
    tail_trimmed = 0
    for m in tail:
        role = m.get("role", "user")
        if role not in ("user", "assistant", "tool", "system"):
            role = "user"
        txt = flatten_content(m.get("content", "")).strip()
        if not txt:
            continue
        trimmed = trim_text_tokens(txt, per_msg_cap)
        tail_trimmed += estimate_tokens(trimmed)
        out.append({"role": role, "content": trimmed})

    dynamic_blocks: list[dict[str, str]] = []
    if must_text:
        dynamic_blocks.append(
            {"role": "system", "content": "Condensed instruction context:\n" + must_text}
        )
    if nice_text:
        dynamic_blocks.append(
            {"role": "system", "content": "Condensed recent context:\n" + nice_text}
        )
    if hist_text:
        dynamic_blocks.append(
            {"role": "system", "content": "Conversation memory tail:\n" + hist_text}
        )

    if cfg.dynamic_context_postfix:
        out.extend(dynamic_blocks)
    else:
        out = dynamic_blocks + out

    if not out:
        out = [{"role": "user", "content": "Please continue based on latest user intent."}]

    stats = {
        "shared_prefix_tokens": estimate_tokens(shared_prefix) if shared_prefix else 0,
        "must_have_tokens": estimate_tokens(must_text) if must_text else 0,
        "nice_to_have_tokens": estimate_tokens(nice_text) if nice_text else 0,
        "history_recent_tokens": estimate_tokens(hist_recent) if hist_recent else 0,
        "history_archive_tokens": estimate_tokens(hist_archive) if hist_archive else 0,
        "history_tail_tokens": estimate_tokens(hist_text) if hist_text else 0,
        "tail_tokens": tail_trimmed,
        "total_output_tokens": sum(estimate_tokens(m["content"]) for m in out),
    }
    return out, stats


def detect_task_type(messages: list[dict[str, Any]]) -> str:
    latest_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user = flatten_content(m.get("content", "")).lower()
            break
    if not latest_user:
        return "chat"

    if any(k in latest_user for k in ("翻译", "translate", "译成", "translation")):
        return "translation"
    if any(k in latest_user for k in ("总结", "摘要", "tl;dr", "summar", "概括")):
        return "summary"
    if any(k in latest_user for k in ("分析", "评估", "review", "对比", "benchmark", "调参")):
        return "analysis"
    if any(
        k in latest_user
        for k in ("代码", "code", "实现", "修复", "debug", "脚本", "api", "函数", "编译")
    ):
        return "code"
    if any(k in latest_user for k in ("写作", "创作", "文案", "故事", "creative", "brainstorm")):
        return "creative"
    return "chat"


def dynamic_max_tokens_budget(cfg: ServiceConfig, task_type: str) -> int:
    mapping = {
        "chat": cfg.max_tokens_chat,
        "summary": cfg.max_tokens_summary,
        "analysis": cfg.max_tokens_analysis,
        "code": cfg.max_tokens_code,
        "creative": cfg.max_tokens_creative,
        "translation": cfg.max_tokens_translation,
    }
    return max(cfg.dynamic_max_tokens_min, int(mapping.get(task_type, cfg.max_tokens_chat)))


class CoreManager:
    def __init__(
        self,
        enabled: bool,
        core_binary: str,
        model: str | None,
        host: str,
        port: int,
        extra_args: list[str],
        env_pairs: list[str],
        watchdog_enabled: bool,
        watchdog_interval: int,
        watchdog_timeout: int,
        watchdog_failures: int,
    ) -> None:
        self.enabled = enabled
        self.core_binary = core_binary
        self.model = model
        self.host = host
        self.port = port
        self.extra_args = extra_args
        self.env_pairs = env_pairs
        self.proc: subprocess.Popen[Any] | None = None
        self.watchdog_enabled = watchdog_enabled
        self.watchdog_interval = max(2, watchdog_interval)
        self.watchdog_timeout = max(1, watchdog_timeout)
        self.watchdog_failures = max(1, watchdog_failures)
        self._watchdog_fail_count = 0
        self._watchdog_lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self.cache_ram_mb = self._extract_cache_ram_mb(self.extra_args)

    @staticmethod
    def _extract_cache_ram_mb(args: list[str]) -> int | None:
        for i, arg in enumerate(args):
            if arg.startswith("--cache-ram="):
                val = arg.split("=", 1)[1].strip()
                if val.isdigit():
                    return int(val)
            if arg == "--cache-ram" and i + 1 < len(args):
                nxt = args[i + 1].strip()
                if nxt.isdigit():
                    return int(nxt)
        return None

    @staticmethod
    def _replace_cache_ram_arg(args: list[str], cache_ram_mb: int) -> list[str]:
        out: list[str] = []
        skip_next = False
        replaced = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg.startswith("--cache-ram="):
                out.append(f"--cache-ram={cache_ram_mb}")
                replaced = True
                continue
            if arg == "--cache-ram":
                out.extend(["--cache-ram", str(cache_ram_mb)])
                replaced = True
                skip_next = (i + 1) < len(args)
                continue
            out.append(arg)
        if not replaced:
            out.append(f"--cache-ram={cache_ram_mb}")
        return out

    def _build_cmd_env(self) -> tuple[list[str], dict[str, str]]:
        cmd = [
            self.core_binary,
            "-m",
            self.model or "",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.extra_args
        env = os.environ.copy()
        for pair in self.env_pairs:
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            env[k] = v
        return cmd, env

    def _wait_ready(self, timeout_s: int = 60) -> bool:
        url = f"http://{self.host}:{self.port}/v1/models"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=2)
                if r.status_code < 500:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        return False

    def _spawn_core(self) -> None:
        cmd, env = self._build_cmd_env()
        print("[core] starting:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, env=env)
        if not self._wait_ready():
            self._terminate_core(force=True)
            raise RuntimeError("llama-server did not become ready in time")
        print(f"[core] ready at http://{self.host}:{self.port}")

    def _terminate_core(self, force: bool = False) -> None:
        if not self.proc:
            return
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5 if force else 10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _is_zombie(self) -> bool:
        if not self.proc:
            return False
        if self.proc.poll() is not None:
            return False
        pid = self.proc.pid
        if pid <= 0:
            return False
        try:
            ps = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            stat = (ps.stdout or "").strip()
            return "Z" in stat
        except Exception:
            return False

    def _probe_healthy(self) -> bool:
        url = f"http://{self.host}:{self.port}/v1/models"
        try:
            r = requests.get(url, timeout=self.watchdog_timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _restart_core(self, reason: str) -> None:
        with self._watchdog_lock:
            if self._watchdog_stop.is_set():
                return
            print(f"[watchdog] restarting core due to: {reason}")
            self._terminate_core(force=True)
            self._spawn_core()
            self._watchdog_fail_count = 0

    def _watchdog_loop(self) -> None:
        print(
            f"[watchdog] enabled interval={self.watchdog_interval}s "
            f"timeout={self.watchdog_timeout}s failures={self.watchdog_failures}"
        )
        while not self._watchdog_stop.wait(self.watchdog_interval):
            if self._watchdog_stop.is_set():
                break
            try:
                if not self.proc:
                    self._restart_core("missing_process")
                    continue
                if self.proc.poll() is not None:
                    self._restart_core("process_exited")
                    continue
                if self._is_zombie():
                    self._restart_core("process_zombie")
                    continue

                if self._probe_healthy():
                    self._watchdog_fail_count = 0
                    continue

                self._watchdog_fail_count += 1
                print(
                    f"[watchdog] health probe failed "
                    f"({self._watchdog_fail_count}/{self.watchdog_failures})"
                )
                if self._watchdog_fail_count >= self.watchdog_failures:
                    self._restart_core("health_probe_failed")
            except Exception as e:
                print(f"[watchdog] loop error: {e}")

    def start(self) -> None:
        if not self.enabled:
            return
        if not self.model:
            raise SystemExit("--core-model is required when --auto-start-core is enabled")
        try:
            self._spawn_core()
        except RuntimeError as e:
            raise SystemExit(f"[core] {e}") from e
        if self.watchdog_enabled:
            self._watchdog_stop.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="thunder-core-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

    def stop(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2)
        if not self.proc:
            return
        if self.proc.poll() is not None:
            return
        print("[core] stopping llama-server...")
        self._terminate_core(force=True)

    def status(self) -> dict[str, Any]:
        running = self.proc is not None and self.proc.poll() is None
        return {
            "enabled": self.enabled,
            "running": running,
            "pid": self.proc.pid if running and self.proc is not None else None,
            "model": self.model,
            "host": self.host,
            "port": self.port,
            "watchdog_enabled": self.watchdog_enabled,
            "watchdog_interval": self.watchdog_interval,
            "watchdog_timeout": self.watchdog_timeout,
            "watchdog_failures": self.watchdog_failures,
            "cache_ram_mb": self.cache_ram_mb,
            "extra_args": list(self.extra_args),
        }

    def switch_model(self, model: str, core_args: list[str] | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("core manager is disabled; enable --auto-start-core first")
        if not model:
            raise RuntimeError("model is required")

        with self._watchdog_lock:
            old_model = self.model
            old_args = list(self.extra_args)

            self.model = model
            if core_args is not None:
                self.extra_args = core_args

            try:
                self._terminate_core(force=True)
                self._spawn_core()
                self._watchdog_fail_count = 0
                print(f"[core] switched model -> {self.model}")
                self.cache_ram_mb = self._extract_cache_ram_mb(self.extra_args)
                return self.status()
            except Exception as e:
                print(f"[core] switch failed, rolling back to previous model: {old_model}")
                self.model = old_model
                self.extra_args = old_args
                try:
                    self._terminate_core(force=True)
                    self._spawn_core()
                    self._watchdog_fail_count = 0
                except Exception as rollback_error:
                    raise RuntimeError(
                        f"switch failed ({e}); rollback failed ({rollback_error})"
                    ) from rollback_error
                raise RuntimeError(f"switch failed: {e}") from e

    def switch_cache_ram_mb(self, cache_ram_mb: int) -> dict[str, Any]:
        if cache_ram_mb <= 0:
            raise RuntimeError("cache_ram_mb must be positive")
        if not self.enabled:
            raise RuntimeError("core manager is disabled; enable --auto-start-core first")
        with self._watchdog_lock:
            old_args = list(self.extra_args)
            self.extra_args = self._replace_cache_ram_arg(self.extra_args, cache_ram_mb)
            try:
                self._terminate_core(force=True)
                self._spawn_core()
                self._watchdog_fail_count = 0
                self.cache_ram_mb = cache_ram_mb
                print(f"[core] switched cache-ram -> {cache_ram_mb} MB")
                return self.status()
            except Exception as e:
                self.extra_args = old_args
                try:
                    self._terminate_core(force=True)
                    self._spawn_core()
                    self._watchdog_fail_count = 0
                except Exception as rollback_error:
                    raise RuntimeError(
                        f"cache-ram switch failed ({e}); rollback failed ({rollback_error})"
                    ) from rollback_error
                self.cache_ram_mb = self._extract_cache_ram_mb(self.extra_args)
                raise RuntimeError(f"cache-ram switch failed: {e}") from e


class AutoCacheRamTuner:
    def __init__(
        self,
        enabled: bool,
        core_mgr: CoreManager,
        metrics_store: MetricsStore | None,
        interval_sec: int,
        lookback_sec: int,
        min_samples: int,
        candidates_mb: list[int],
        cooldown_sec: int,
        min_improve_score: float,
    ) -> None:
        self.enabled = enabled
        self.core_mgr = core_mgr
        self.metrics_store = metrics_store
        self.interval_sec = max(30, interval_sec)
        self.lookback_sec = max(300, lookback_sec)
        self.min_samples = max(1, min_samples)
        self.candidates_mb = sorted(set(x for x in candidates_mb if x > 0))
        self.cooldown_sec = max(60, cooldown_sec)
        self.min_improve_score = max(0.0, min_improve_score)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.last_switch_at: int | None = None
        self.last_decision: dict[str, Any] = {"status": "idle"}

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo < 1e-9:
            return [0.5 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    def _choose_best(self, rows: list[dict[str, Any]]) -> tuple[int | None, list[dict[str, Any]]]:
        valid = [r for r in rows if int(r.get("total", 0)) >= self.min_samples]
        if not valid:
            return None, []
        th = self._normalize([float(r.get("throughput_rps", 0.0)) for r in valid])
        lat = self._normalize([float(r.get("avg_latency_ms", 0.0)) for r in valid])
        fail = self._normalize([float(r.get("failure_rate", 0.0)) for r in valid])
        for i, r in enumerate(valid):
            score = 0.50 * th[i] + 0.35 * (1.0 - lat[i]) + 0.15 * (1.0 - fail[i])
            r["score"] = score
        best = max(valid, key=lambda x: float(x.get("score", 0.0)))
        return int(best["cache_ram_mb"]), valid

    def _run_once(self) -> None:
        if not self.enabled:
            return
        if not self.core_mgr.enabled:
            self.last_decision = {"status": "disabled", "reason": "core_not_enabled"}
            return
        if self.metrics_store is None:
            self.last_decision = {"status": "disabled", "reason": "metrics_store_missing"}
            return
        if not self.candidates_mb:
            self.last_decision = {"status": "disabled", "reason": "no_candidates"}
            return

        rows = self.metrics_store.summarize_recent_cache_ram(
            lookback_sec=self.lookback_sec,
            candidates_mb=self.candidates_mb,
        )
        target, scored = self._choose_best(rows)
        now = int(time.time())
        current = self.core_mgr.cache_ram_mb
        decision: dict[str, Any] = {
            "status": "analyzed",
            "at": now,
            "current_cache_ram_mb": current,
            "target_cache_ram_mb": target,
            "rows": scored,
        }
        if target is None:
            decision["reason"] = "insufficient_samples"
            self.last_decision = decision
            return
        if current == target:
            decision["reason"] = "already_optimal"
            self.last_decision = decision
            return
        if self.last_switch_at and now - self.last_switch_at < self.cooldown_sec:
            decision["reason"] = "cooldown_active"
            self.last_decision = decision
            return

        score_by_cache = {int(r["cache_ram_mb"]): float(r.get("score", 0.0)) for r in scored}
        target_score = score_by_cache.get(target, 0.0)
        current_score = score_by_cache.get(current or -1, 0.0)
        if current is not None and (target_score - current_score) < self.min_improve_score:
            decision["reason"] = "improvement_too_small"
            decision["score_delta"] = target_score - current_score
            self.last_decision = decision
            return

        try:
            self.core_mgr.switch_cache_ram_mb(target)
            self.last_switch_at = now
            decision["status"] = "switched"
            decision["reason"] = "better_24h_score"
            decision["new_cache_ram_mb"] = target
        except Exception as e:
            decision["status"] = "switch_failed"
            decision["reason"] = str(e)
        self.last_decision = decision

    def _loop(self) -> None:
        print(
            f"[auto-cache-ram] enabled interval={self.interval_sec}s lookback={self.lookback_sec}s "
            f"candidates={self.candidates_mb} min_samples={self.min_samples}"
        )
        while not self._stop.wait(self.interval_sec):
            if self._stop.is_set():
                break
            with self._lock:
                try:
                    self._run_once()
                except Exception as e:
                    self.last_decision = {"status": "error", "reason": str(e), "at": int(time.time())}
                    print(f"[auto-cache-ram] loop error: {e}")

    def start(self) -> None:
        if not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="thunder-auto-cache-ram",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_sec": self.interval_sec,
            "lookback_sec": self.lookback_sec,
            "min_samples": self.min_samples,
            "candidates_mb": self.candidates_mb,
            "cooldown_sec": self.cooldown_sec,
            "last_switch_at": self.last_switch_at,
            "last_decision": self.last_decision,
        }


class TieredPromptCache:
    def __init__(
        self,
        enabled: bool,
        warm_dir: str,
        hot_max_entries: int,
        hot_ttl_sec: int,
        warm_ttl_sec: int,
        prune_interval_sec: int,
        hot_hit_threshold: int,
    ) -> None:
        self.enabled = enabled
        self.warm_dir = Path(warm_dir).expanduser()
        self.hot_max_entries = max(1, hot_max_entries)
        self.hot_ttl_sec = max(30, hot_ttl_sec)
        self.warm_ttl_sec = max(self.hot_ttl_sec, warm_ttl_sec)
        self.prune_interval_sec = max(30, prune_interval_sec)
        self.hot_hit_threshold = max(1, hot_hit_threshold)
        self.lock = threading.Lock()
        self.hot_cache: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats_data = {"hit_hot": 0, "hit_warm": 0, "miss": 0, "store": 0}
        if self.enabled:
            self.warm_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_cacheable(payload: dict[str, Any]) -> bool:
        if payload.get("stream") is True:
            return False
        if int(payload.get("n", 1) or 1) != 1:
            return False
        temp = payload.get("temperature", 0)
        try:
            return float(temp) == 0.0
        except Exception:
            return False

    @staticmethod
    def _stable_payload_for_key(payload: dict[str, Any], layered_messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": payload.get("model"),
            "messages": layered_messages,
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature", 0),
            "top_p": payload.get("top_p"),
            "presence_penalty": payload.get("presence_penalty"),
            "frequency_penalty": payload.get("frequency_penalty"),
            "stop": payload.get("stop"),
            "response_format": payload.get("response_format"),
            "tools": payload.get("tools"),
            "tool_choice": payload.get("tool_choice"),
        }

    def make_key(self, payload: dict[str, Any], layered_messages: list[dict[str, str]]) -> str:
        stable = self._stable_payload_for_key(payload, layered_messages)
        raw = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _warm_path(self, key: str) -> Path:
        return self.warm_dir / f"{key}.json"

    @staticmethod
    def _now() -> int:
        return int(time.time())

    def _is_expired(self, entry: dict[str, Any], ttl_sec: int) -> bool:
        ts = int(entry.get("last_access", entry.get("created_at", 0)) or 0)
        return (self._now() - ts) > ttl_sec

    def _promote_hot_if_needed(self, key: str, entry: dict[str, Any]) -> None:
        if int(entry.get("hit_count", 0)) < self.hot_hit_threshold:
            return
        self.hot_cache[key] = dict(entry)
        if len(self.hot_cache) > self.hot_max_entries:
            oldest_key = min(
                self.hot_cache.keys(),
                key=lambda k: int(self.hot_cache[k].get("last_access", 0)),
            )
            if oldest_key != key:
                self.hot_cache.pop(oldest_key, None)

    def get(self, key: str) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled:
            return None, None
        now = self._now()
        with self.lock:
            hot = self.hot_cache.get(key)
            if hot and not self._is_expired(hot, self.hot_ttl_sec):
                hot["last_access"] = now
                hot["hit_count"] = int(hot.get("hit_count", 0)) + 1
                self.stats_data["hit_hot"] += 1
                body = hot.get("response")
                return (json.loads(json.dumps(body)) if isinstance(body, dict) else None), "hot"
            if hot and self._is_expired(hot, self.hot_ttl_sec):
                self.hot_cache.pop(key, None)

            p = self._warm_path(key)
            if not p.exists():
                self.stats_data["miss"] += 1
                return None, None
            try:
                entry = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                p.unlink(missing_ok=True)
                self.stats_data["miss"] += 1
                return None, None
            if self._is_expired(entry, self.warm_ttl_sec):
                p.unlink(missing_ok=True)
                self.stats_data["miss"] += 1
                return None, None
            entry["last_access"] = now
            entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
            try:
                p.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            self._promote_hot_if_needed(key, entry)
            self.stats_data["hit_warm"] += 1
            body = entry.get("response")
            return (json.loads(json.dumps(body)) if isinstance(body, dict) else None), "warm"

    def put(self, key: str, response_body: dict[str, Any], force_hot: bool = False) -> None:
        if not self.enabled:
            return
        now = self._now()
        entry = {
            "created_at": now,
            "last_access": now,
            "hit_count": 1,
            "response": response_body,
        }
        with self.lock:
            p = self._warm_path(key)
            try:
                p.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            except Exception:
                return
            if force_hot:
                hot_entry = dict(entry)
                hot_entry["hit_count"] = max(self.hot_hit_threshold, int(hot_entry["hit_count"]))
                self.hot_cache[key] = hot_entry
                if len(self.hot_cache) > self.hot_max_entries:
                    oldest_key = min(
                        self.hot_cache.keys(),
                        key=lambda k: int(self.hot_cache[k].get("last_access", 0)),
                    )
                    if oldest_key != key:
                        self.hot_cache.pop(oldest_key, None)
            self.stats_data["store"] += 1

    def _prune_once(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            for k in list(self.hot_cache.keys()):
                if self._is_expired(self.hot_cache[k], self.hot_ttl_sec):
                    self.hot_cache.pop(k, None)
            if not self.warm_dir.exists():
                return
            for p in self.warm_dir.glob("*.json"):
                try:
                    entry = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    p.unlink(missing_ok=True)
                    continue
                if self._is_expired(entry, self.warm_ttl_sec):
                    p.unlink(missing_ok=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.prune_interval_sec):
            if self._stop.is_set():
                break
            try:
                self._prune_once()
            except Exception as e:
                print(f"[prompt-cache] prune error: {e}")

    def start(self) -> None:
        if not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="thunder-prompt-cache-prune",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._prune_once()

    def status(self) -> dict[str, Any]:
        warm_files = 0
        if self.enabled and self.warm_dir.exists():
            warm_files = len(list(self.warm_dir.glob("*.json")))
        return {
            "enabled": self.enabled,
            "hot_max_entries": self.hot_max_entries,
            "hot_ttl_sec": self.hot_ttl_sec,
            "warm_ttl_sec": self.warm_ttl_sec,
            "hot_hit_threshold": self.hot_hit_threshold,
            "hot_entries": len(self.hot_cache),
            "warm_entries": warm_files,
            "stats": dict(self.stats_data),
        }


class ServiceWarmupManager:
    def __init__(
        self,
        enabled: bool,
        cfg: ServiceConfig,
        timeout: int,
        warmup_requests: int,
        warmup_max_tokens: int,
        prime_enabled: bool,
        prime_top_n: int,
        hotspot_store_path: str,
        prompt_cache: TieredPromptCache,
        residency_enabled: bool,
        residency_interval_sec: int,
        residency_top_n: int,
        residency_force_hot: bool,
    ) -> None:
        self.enabled = enabled
        self.cfg = cfg
        self.timeout = max(5, timeout)
        self.warmup_requests = max(0, warmup_requests)
        self.warmup_max_tokens = max(8, warmup_max_tokens)
        self.prime_enabled = prime_enabled
        self.prime_top_n = max(0, prime_top_n)
        self.hotspot_store_path = Path(hotspot_store_path).expanduser()
        self.prompt_cache = prompt_cache
        self.residency_enabled = residency_enabled
        self.residency_interval_sec = max(30, residency_interval_sec)
        self.residency_top_n = max(0, residency_top_n)
        self.residency_force_hot = residency_force_hot
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_report: dict[str, Any] = {"status": "idle"}

    def _post_chat(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        try:
            r = requests.post(
                self.cfg.upstream.rstrip("/") + "/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return False, {"status": r.status_code, "text": r.text[:200]}
            body = r.json()
            return isinstance(body, dict), body if isinstance(body, dict) else None
        except Exception as e:
            return False, {"error": str(e)}

    def _load_templates(self, top_n: int) -> list[str]:
        templates: list[str] = []
        if top_n <= 0 or not self.hotspot_store_path.exists():
            return templates
        try:
            payload = json.loads(self.hotspot_store_path.read_text(encoding="utf-8"))
            top = payload.get("top", [])
            for item in top:
                if not isinstance(item, dict):
                    continue
                sig = str(item.get("prompt_signature", "")).strip()
                if sig:
                    templates.append(sig)
        except Exception:
            return []
        return templates[:top_n]

    def _prime_templates(self, templates: list[str], force_hot: bool) -> tuple[int, int]:
        primed = 0
        prime_fail = 0
        for t in templates:
            prime_payload = {
                "messages": [{"role": "user", "content": t}],
                "temperature": 0.0,
                "max_tokens": self.warmup_max_tokens,
            }
            layered, _ = layer_messages(prime_payload["messages"], self.cfg.layering)
            key = self.prompt_cache.make_key(prime_payload, layered)
            ok, body = self._post_chat({**prime_payload, "messages": layered})
            if ok and isinstance(body, dict):
                self.prompt_cache.put(key, body, force_hot=force_hot)
                primed += 1
            else:
                prime_fail += 1
        return primed, prime_fail

    def _residency_loop(self) -> None:
        while not self._stop.wait(self.residency_interval_sec):
            if self._stop.is_set():
                break
            try:
                templates = self._load_templates(self.residency_top_n)
                primed, failed = self._prime_templates(templates, force_hot=self.residency_force_hot)
                self.last_report = {
                    **self.last_report,
                    "residency_last_at": int(time.time()),
                    "residency_templates": len(templates),
                    "residency_primed": primed,
                    "residency_fail": failed,
                }
            except Exception as e:
                self.last_report = {
                    **self.last_report,
                    "residency_last_at": int(time.time()),
                    "residency_error": str(e),
                }

    def run(self) -> dict[str, Any]:
        if not self.enabled:
            self.last_report = {"status": "disabled"}
            return self.last_report

        report: dict[str, Any] = {
            "status": "done",
            "warmup_ok": 0,
            "warmup_fail": 0,
            "primed": 0,
            "prime_fail": 0,
            "residency_enabled": self.residency_enabled,
        }

        warm_payload = {
            "messages": [{"role": "user", "content": "Warmup request. Reply with: ready"}],
            "temperature": 0.0,
            "max_tokens": self.warmup_max_tokens,
        }
        for _ in range(self.warmup_requests):
            ok, _ = self._post_chat(warm_payload)
            if ok:
                report["warmup_ok"] += 1
            else:
                report["warmup_fail"] += 1

        if self.prime_enabled and self.prompt_cache.enabled and self.prime_top_n > 0:
            templates = self._load_templates(self.prime_top_n)
            report["primed_templates"] = len(templates)
            primed, failed = self._prime_templates(templates, force_hot=self.residency_force_hot)
            report["primed"] += primed
            report["prime_fail"] += failed

        self.last_report = report
        return report

    def start(self) -> None:
        if not self.enabled or not self.residency_enabled or not self.prompt_cache.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._residency_loop,
            name="thunder-cache-residency",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


@dataclass
class ServiceConfig:
    listen_host: str
    listen_port: int
    upstream: str
    timeout: int
    layering: LayeringConfig
    hotspot_enabled: bool
    hotspot_top_n: int
    metrics_db_enabled: bool
    dynamic_max_tokens_enabled: bool
    dynamic_max_tokens_min: int
    max_tokens_chat: int
    max_tokens_summary: int
    max_tokens_analysis: int
    max_tokens_code: int
    max_tokens_creative: int
    max_tokens_translation: int


class ThunderServiceHandler(BaseHTTPRequestHandler):
    cfg: ServiceConfig
    hotspot_tracker: PromptHotspotTracker
    metrics_store: MetricsStore | None
    core_mgr: CoreManager
    auto_cache_tuner: AutoCacheRamTuner
    prompt_cache: TieredPromptCache
    warmup_mgr: ServiceWarmupManager

    def _write_bytes(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, status: int, body: dict[str, Any]) -> None:
        self._write_bytes(status, json.dumps(body, ensure_ascii=False).encode("utf-8"))

    def _upstream_url(self, path: str) -> str:
        return self.cfg.upstream.rstrip("/") + path

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write_json(200, {"ok": True})
            return
        if self.path.startswith("/thunder/hot-prompts"):
            if not self.cfg.hotspot_enabled:
                self._write_json(200, {"enabled": False, "top": []})
                return
            self._write_json(
                200,
                {
                    "enabled": True,
                    "top": self.hotspot_tracker.top(self.cfg.hotspot_top_n),
                },
            )
            return
        if self.path == "/thunder/features":
            self._write_json(
                200,
                {
                    "layering": {
                        "enabled": self.cfg.layering.enabled,
                        "must_have_cap": self.cfg.layering.must_have_cap,
                        "nice_to_have_cap": self.cfg.layering.nice_to_have_cap,
                        "history_tail_cap": self.cfg.layering.history_tail_cap,
                        "history_recent_cap": self.cfg.layering.history_recent_cap,
                        "history_archive_cap": self.cfg.layering.history_archive_cap,
                        "preserve_last_turns": self.cfg.layering.preserve_last_turns,
                        "dynamic_context_postfix": self.cfg.layering.dynamic_context_postfix,
                    },
                    "hotspot_tracking": self.cfg.hotspot_enabled,
                    "upstream": self.cfg.upstream,
                },
            )
            return
        if self.path == "/thunder/core/status":
            self._write_json(200, self.core_mgr.status())
            return
        if self.path == "/thunder/cache-policy/status":
            self._write_json(
                200,
                {
                    "core": self.core_mgr.status(),
                    "auto_cache_ram": self.auto_cache_tuner.status(),
                },
            )
            return
        if self.path == "/thunder/prompt-cache/status":
            self._write_json(200, self.prompt_cache.status())
            return
        if self.path == "/thunder/warmup/status":
            self._write_json(200, self.warmup_mgr.last_report)
            return
        self._proxy_raw()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/thunder/core/switch-model":
            self._switch_core_model()
            return
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._proxy_chat_completions()
            return
        self._proxy_raw()

    def _switch_core_model(self) -> None:
        cl = int(self.headers.get("Content-Length", "0") or "0")
        if cl <= 0:
            self._write_json(400, {"error": "empty_body"})
            return
        raw = self.rfile.read(cl)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_json(400, {"error": "invalid_json"})
            return

        model = payload.get("model")
        core_args = payload.get("core_args")
        if core_args is not None and not isinstance(core_args, list):
            self._write_json(400, {"error": "core_args_must_be_list"})
            return
        if isinstance(core_args, list):
            core_args = [str(x) for x in core_args]

        try:
            status = self.core_mgr.switch_model(str(model or ""), core_args=core_args)
        except RuntimeError as e:
            self._write_json(409, {"error": "switch_model_failed", "detail": str(e)})
            return

        self._write_json(200, {"ok": True, "core": status})

    def _proxy_raw(self) -> None:
        method = self.command
        headers = {
            k: v for k, v in self.headers.items() if k.lower() in ("authorization", "content-type")
        }
        try:
            if method == "GET":
                r = requests.get(self._upstream_url(self.path), headers=headers, timeout=self.cfg.timeout)
            else:
                cl = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(cl) if cl > 0 else b""
                r = requests.request(
                    method,
                    self._upstream_url(self.path),
                    headers=headers,
                    data=body,
                    timeout=self.cfg.timeout,
                )
        except requests.RequestException as e:
            self._write_json(502, {"error": "upstream_unreachable", "detail": str(e)})
            return

        content_type = r.headers.get("Content-Type", "application/json; charset=utf-8")
        self._write_bytes(r.status_code, r.content, content_type)

    def _proxy_chat_completions(self) -> None:
        started_at = time.time()
        cl = int(self.headers.get("Content-Length", "0") or "0")
        if cl <= 0:
            self._write_json(400, {"error": "empty_body"})
            return
        raw = self.rfile.read(cl)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_json(400, {"error": "invalid_json"})
            return

        if payload.get("stream") is True:
            self._write_json(501, {"error": "stream_not_supported_by_thunder_service"})
            return

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            self._write_json(400, {"error": "messages_required"})
            return

        if self.cfg.hotspot_enabled:
            self.hotspot_tracker.update_from_messages(messages)

        layered_messages, layer_stats = layer_messages(messages, self.cfg.layering)

        dynamic_meta: dict[str, Any] = {
            "enabled": self.cfg.dynamic_max_tokens_enabled,
            "task_type": None,
            "budget": None,
            "input_max_tokens": payload.get("max_tokens"),
            "applied_max_tokens": payload.get("max_tokens"),
        }
        if self.cfg.dynamic_max_tokens_enabled:
            task_type = detect_task_type(messages)
            budget = dynamic_max_tokens_budget(self.cfg, task_type)
            req_max_tokens = payload.get("max_tokens")
            applied = budget
            try:
                if req_max_tokens is not None:
                    applied = min(int(req_max_tokens), budget)
            except Exception:
                applied = budget
            payload["max_tokens"] = max(self.cfg.dynamic_max_tokens_min, int(applied))
            dynamic_meta.update(
                {
                    "task_type": task_type,
                    "budget": budget,
                    "applied_max_tokens": payload["max_tokens"],
                }
            )

        cache_key: str | None = None
        cache_tier: str | None = None
        if self.prompt_cache.enabled and self.prompt_cache.is_cacheable(payload):
            cache_key = self.prompt_cache.make_key(payload, layered_messages)
            cached_body, cache_tier = self.prompt_cache.get(cache_key)
            if isinstance(cached_body, dict):
                cached_body.setdefault("thunderllama_service", {})
                cached_body["thunderllama_service"].update(
                    {
                        "input_messages": len(messages),
                        "output_messages": len(layered_messages),
                        "layering": {
                            "enabled": self.cfg.layering.enabled,
                            "must_have_cap": self.cfg.layering.must_have_cap,
                            "nice_to_have_cap": self.cfg.layering.nice_to_have_cap,
                            "history_tail_cap": self.cfg.layering.history_tail_cap,
                            "history_recent_cap": self.cfg.layering.history_recent_cap,
                            "history_archive_cap": self.cfg.layering.history_archive_cap,
                            "preserve_last_turns": self.cfg.layering.preserve_last_turns,
                            "dynamic_context_postfix": self.cfg.layering.dynamic_context_postfix,
                        },
                        "token_estimate": layer_stats,
                        "prompt_cache": {
                            "enabled": True,
                            "hit": True,
                            "tier": cache_tier,
                        },
                        "core_runtime": {
                            "cache_ram_mb": self.core_mgr.cache_ram_mb,
                        },
                        "dynamic_max_tokens": dynamic_meta,
                    }
                )
                if self.cfg.hotspot_enabled:
                    cached_body["thunderllama_service"]["hot_prompt_top"] = self.hotspot_tracker.top(5)
                self._write_json(200, cached_body)
                self._persist_metrics(
                    payload=payload,
                    layered_messages=layered_messages,
                    layer_stats=layer_stats,
                    upstream_status=200,
                    started_at=started_at,
                    error=None,
                    body=cached_body,
                    prompt_cache_hit=True,
                    prompt_cache_tier=cache_tier,
                )
                return

        fwd = dict(payload)
        fwd["messages"] = layered_messages

        headers = {
            k: v for k, v in self.headers.items() if k.lower() in ("authorization", "content-type")
        }
        headers["Content-Type"] = "application/json"

        try:
            r = requests.post(
                self._upstream_url("/chat/completions"),
                headers=headers,
                json=fwd,
                timeout=self.cfg.timeout,
            )
        except requests.RequestException as e:
            self._write_json(502, {"error": "upstream_unreachable", "detail": str(e)})
            self._persist_metrics(
                payload=payload,
                layered_messages=layered_messages,
                layer_stats=layer_stats,
                upstream_status=502,
                started_at=started_at,
                error=f"upstream_unreachable: {e}",
                prompt_cache_hit=False,
            )
            return

        try:
            body = r.json()
        except Exception:
            body = {"error": "upstream_non_json", "status": r.status_code, "text": r.text[:500]}
        cache_miss = cache_key is not None

        if isinstance(body, dict):
            upstream_body_for_cache = json.loads(json.dumps(body))
            if (
                self.prompt_cache.enabled
                and cache_key
                and r.status_code == 200
                and "error" not in body
            ):
                self.prompt_cache.put(cache_key, upstream_body_for_cache)
            body.setdefault("thunderllama_service", {})
            body["thunderllama_service"].update(
                {
                    "input_messages": len(messages),
                    "output_messages": len(layered_messages),
                    "layering": {
                        "enabled": self.cfg.layering.enabled,
                        "must_have_cap": self.cfg.layering.must_have_cap,
                        "nice_to_have_cap": self.cfg.layering.nice_to_have_cap,
                        "history_tail_cap": self.cfg.layering.history_tail_cap,
                        "history_recent_cap": self.cfg.layering.history_recent_cap,
                        "history_archive_cap": self.cfg.layering.history_archive_cap,
                        "preserve_last_turns": self.cfg.layering.preserve_last_turns,
                        "dynamic_context_postfix": self.cfg.layering.dynamic_context_postfix,
                    },
                    "token_estimate": layer_stats,
                    "prompt_cache": {
                        "enabled": self.prompt_cache.enabled,
                        "hit": False,
                        "tier": None,
                        "miss": cache_miss,
                    },
                    "core_runtime": {
                        "cache_ram_mb": self.core_mgr.cache_ram_mb,
                    },
                    "dynamic_max_tokens": dynamic_meta,
                }
            )
            if self.cfg.hotspot_enabled:
                body["thunderllama_service"]["hot_prompt_top"] = self.hotspot_tracker.top(5)

        self._write_json(r.status_code, body if isinstance(body, dict) else {"result": body})

        if self.cfg.hotspot_enabled:
            self.hotspot_tracker.flush()
        self._persist_metrics(
            payload=payload,
            layered_messages=layered_messages,
            layer_stats=layer_stats,
            upstream_status=r.status_code,
            started_at=started_at,
            error=body.get("error") if isinstance(body, dict) else None,
            body=body if isinstance(body, dict) else {},
            prompt_cache_hit=False,
            prompt_cache_tier=None,
        )

    def _persist_metrics(
        self,
        payload: dict[str, Any],
        layered_messages: list[dict[str, str]],
        layer_stats: dict[str, int],
        upstream_status: int,
        started_at: float,
        error: str | None = None,
        body: dict[str, Any] | None = None,
        prompt_cache_hit: bool = False,
        prompt_cache_tier: str | None = None,
    ) -> None:
        if not self.cfg.metrics_db_enabled or self.metrics_store is None:
            return
        messages = payload.get("messages") or []
        input_tokens = 0
        if isinstance(messages, list):
            input_tokens = sum(
                estimate_tokens(flatten_content(m.get("content", ""))) for m in messages if isinstance(m, dict)
            )
        latency_ms = int((time.time() - started_at) * 1000)
        row = {
            "ts_unix": int(time.time()),
            "model": payload.get("model"),
            "input_messages": len(messages) if isinstance(messages, list) else 0,
            "output_messages": len(layered_messages),
            "input_token_estimate": input_tokens,
            "output_token_estimate": int(layer_stats.get("total_output_tokens", 0)),
            "must_have_cap": self.cfg.layering.must_have_cap,
            "nice_to_have_cap": self.cfg.layering.nice_to_have_cap,
            "history_tail_cap": self.cfg.layering.history_tail_cap,
            "preserve_last_turns": self.cfg.layering.preserve_last_turns,
            "must_have_tokens": int(layer_stats.get("must_have_tokens", 0)),
            "nice_to_have_tokens": int(layer_stats.get("nice_to_have_tokens", 0)),
            "history_tail_tokens": int(layer_stats.get("history_tail_tokens", 0)),
            "tail_tokens": int(layer_stats.get("tail_tokens", 0)),
            "latency_ms": latency_ms,
            "upstream_status": upstream_status,
            "error": error,
            "cache_ram_mb": int(self.core_mgr.cache_ram_mb or 0),
            "meta_json": {
                "layering_enabled": self.cfg.layering.enabled,
                "dynamic_context_postfix": self.cfg.layering.dynamic_context_postfix,
                "shared_prefix_tokens": int(layer_stats.get("shared_prefix_tokens", 0)),
                "history_recent_tokens": int(layer_stats.get("history_recent_tokens", 0)),
                "history_archive_tokens": int(layer_stats.get("history_archive_tokens", 0)),
                "hotspot_enabled": self.cfg.hotspot_enabled,
                "prompt_cache_enabled": self.prompt_cache.enabled,
                "prompt_cache_hit": prompt_cache_hit,
                "prompt_cache_tier": prompt_cache_tier,
                "dynamic_max_tokens": dynamic_max_tokens_budget(self.cfg, detect_task_type(messages))
                if self.cfg.dynamic_max_tokens_enabled
                else None,
                "body_usage": body.get("usage") if isinstance(body, dict) else None,
                "body_error": body.get("error") if isinstance(body, dict) else None,
            },
        }
        try:
            self.metrics_store.insert(row)
        except Exception as e:
            print(f"[metrics-db] insert failed: {e}")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ThunderLLAMA service layer")

    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=18081)

    ap.add_argument("--upstream", default="http://127.0.0.1:18082/v1")
    ap.add_argument("--timeout", type=int, default=300)

    ap.add_argument("--enable-layering", action="store_true", default=True)
    ap.add_argument("--disable-layering", action="store_true")
    ap.add_argument("--must-have-cap", type=int, default=1536)
    ap.add_argument("--nice-to-have-cap", type=int, default=768)
    ap.add_argument("--history-tail-cap", type=int, default=512)
    ap.add_argument(
        "--history-recent-cap",
        type=int,
        default=320,
        help="Token cap for recent dialogue summary layer",
    )
    ap.add_argument(
        "--history-archive-cap",
        type=int,
        default=192,
        help="Token cap for archived dialogue summary layer",
    )
    ap.add_argument("--preserve-last-turns", type=int, default=6)
    ap.add_argument("--enable-dynamic-context-postfix", action="store_true", default=True)
    ap.add_argument("--disable-dynamic-context-postfix", action="store_true")
    ap.add_argument(
        "--shared-prefix-file",
        default=".solar/prompt-cache/openclaw-shared-prefix.txt",
        help="Stable shared prefix file to maximize prompt prefix cache reuse",
    )
    ap.add_argument(
        "--enable-context-shift",
        action="store_true",
        default=False,
        help="Enable two-stage LLM summarization (0.6B extract + 0.6B/1.7B compress)",
    )
    ap.add_argument(
        "--context-shift-stage1-endpoint",
        default="http://127.0.0.1:18083/completion",
        help="Stage 1 (extract) endpoint - 0.6B model",
    )
    ap.add_argument(
        "--context-shift-stage2-endpoint",
        default="http://127.0.0.1:18084/completion",
        help="Stage 2 (compress) endpoint - 1.7B model (quality mode)",
    )
    ap.add_argument(
        "--context-shift-mode",
        default="auto",
        choices=["auto", "fast", "quality", "simple"],
        help="Summarization mode: auto (dynamic), fast (0.6B+0.6B), quality (0.6B+1.7B), simple (char truncation)",
    )
    ap.add_argument(
        "--context-shift-auto-threshold",
        type=int,
        default=8,
        help="Auto mode threshold: conversation turns >= N use quality mode, < N use fast mode",
    )

    ap.add_argument("--enable-hotspot-tracking", action="store_true", default=True)
    ap.add_argument("--disable-hotspot-tracking", action="store_true")
    ap.add_argument("--hotspot-top-n", type=int, default=20)
    ap.add_argument("--hotspot-store-path", default=".solar/prompt-cache/hot_prompts_live.json")
    ap.add_argument("--disable-metrics-db", action="store_true")
    ap.add_argument("--metrics-db-path", default=".solar/metrics/thunder_service.db")
    ap.add_argument("--enable-auto-cache-ram", action="store_true", default=True)
    ap.add_argument("--disable-auto-cache-ram", action="store_true")
    ap.add_argument("--auto-cache-interval-sec", type=int, default=300)
    ap.add_argument("--auto-cache-lookback-sec", type=int, default=24 * 3600)
    ap.add_argument("--auto-cache-min-samples", type=int, default=20)
    ap.add_argument("--auto-cache-candidates-mb", default="2048,4096,6144,8192")
    ap.add_argument("--auto-cache-cooldown-sec", type=int, default=1800)
    ap.add_argument("--auto-cache-min-improve", type=float, default=0.03)

    ap.add_argument("--enable-tiered-prompt-cache", action="store_true", default=True)
    ap.add_argument("--disable-tiered-prompt-cache", action="store_true")
    ap.add_argument("--prompt-cache-warm-dir", default=".solar/prompt-cache/warm")
    ap.add_argument("--prompt-cache-hot-max-entries", type=int, default=256)
    ap.add_argument("--prompt-cache-hot-ttl-sec", type=int, default=3600)
    ap.add_argument("--prompt-cache-warm-ttl-sec", type=int, default=24 * 3600)
    ap.add_argument("--prompt-cache-prune-interval-sec", type=int, default=300)
    ap.add_argument("--prompt-cache-hot-hit-threshold", type=int, default=3)
    ap.add_argument("--enable-dynamic-max-tokens", action="store_true", default=True)
    ap.add_argument("--disable-dynamic-max-tokens", action="store_true")
    ap.add_argument("--dynamic-max-tokens-min", type=int, default=64)
    ap.add_argument("--max-tokens-chat", type=int, default=192)
    ap.add_argument("--max-tokens-summary", type=int, default=192)
    ap.add_argument("--max-tokens-analysis", type=int, default=384)
    ap.add_argument("--max-tokens-code", type=int, default=512)
    ap.add_argument("--max-tokens-creative", type=int, default=768)
    ap.add_argument("--max-tokens-translation", type=int, default=256)

    ap.add_argument("--enable-startup-warmup", action="store_true", default=True)
    ap.add_argument("--disable-startup-warmup", action="store_true")
    ap.add_argument("--warmup-requests", type=int, default=2)
    ap.add_argument("--warmup-max-tokens", type=int, default=32)
    ap.add_argument("--enable-cache-priming", action="store_true", default=True)
    ap.add_argument("--disable-cache-priming", action="store_true")
    ap.add_argument("--cache-prime-top-n", type=int, default=10)
    ap.add_argument("--enable-cache-residency", action="store_true", default=True)
    ap.add_argument("--disable-cache-residency", action="store_true")
    ap.add_argument("--cache-residency-interval-sec", type=int, default=300)
    ap.add_argument("--cache-residency-top-n", type=int, default=10)
    ap.add_argument("--enable-cache-residency-force-hot", action="store_true", default=True)
    ap.add_argument("--disable-cache-residency-force-hot", action="store_true")

    # Optional: let this service boot llama-server core itself.
    ap.add_argument("--auto-start-core", action="store_true")
    ap.add_argument("--core-binary", default="./build/bin/llama-server")
    ap.add_argument("--core-model")
    ap.add_argument("--core-host", default="127.0.0.1")
    ap.add_argument("--core-port", type=int, default=18082)
    ap.add_argument("--core-arg", action="append", default=[], help="repeatable extra arg for llama-server")
    ap.add_argument(
        "--core-env",
        action="append",
        default=["LLAMA_PAGED_ATTENTION=1"],
        help="repeatable env pair KEY=VALUE for llama-server",
    )
    ap.add_argument("--disable-core-watchdog", action="store_true")
    ap.add_argument("--core-watchdog-interval", type=int, default=10)
    ap.add_argument("--core-watchdog-timeout", type=int, default=3)
    ap.add_argument("--core-watchdog-failures", type=int, default=3)

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    layering_enabled = args.enable_layering and not args.disable_layering
    dynamic_context_postfix = (
        args.enable_dynamic_context_postfix and not args.disable_dynamic_context_postfix
    )
    hotspot_enabled = args.enable_hotspot_tracking and not args.disable_hotspot_tracking
    auto_cache_ram_enabled = args.enable_auto_cache_ram and not args.disable_auto_cache_ram
    tiered_prompt_cache_enabled = (
        args.enable_tiered_prompt_cache and not args.disable_tiered_prompt_cache
    )
    dynamic_max_tokens_enabled = (
        args.enable_dynamic_max_tokens and not args.disable_dynamic_max_tokens
    )
    startup_warmup_enabled = args.enable_startup_warmup and not args.disable_startup_warmup
    cache_priming_enabled = args.enable_cache_priming and not args.disable_cache_priming
    cache_residency_enabled = args.enable_cache_residency and not args.disable_cache_residency
    cache_residency_force_hot = (
        args.enable_cache_residency_force_hot and not args.disable_cache_residency_force_hot
    )

    try:
        auto_cache_candidates_mb = [
            int(x.strip())
            for x in str(args.auto_cache_candidates_mb).split(",")
            if x.strip()
        ]
    except Exception:
        auto_cache_candidates_mb = [2048, 4096, 6144, 8192]

    shared_prefix_text = ""
    try:
        sp = Path(args.shared_prefix_file).expanduser()
        if sp.exists():
            shared_prefix_text = sp.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[layering] failed to load shared prefix file: {e}")

    core_mgr = CoreManager(
        enabled=args.auto_start_core,
        core_binary=args.core_binary,
        model=args.core_model,
        host=args.core_host,
        port=args.core_port,
        extra_args=args.core_arg,
        env_pairs=args.core_env,
        watchdog_enabled=(args.auto_start_core and not args.disable_core_watchdog),
        watchdog_interval=args.core_watchdog_interval,
        watchdog_timeout=args.core_watchdog_timeout,
        watchdog_failures=args.core_watchdog_failures,
    )

    if args.auto_start_core:
        core_mgr.start()
        args.upstream = f"http://{args.core_host}:{args.core_port}/v1"

    atexit.register(core_mgr.stop)

    cfg = ServiceConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream=args.upstream,
        timeout=args.timeout,
        layering=LayeringConfig(
            enabled=layering_enabled,
            must_have_cap=args.must_have_cap,
            nice_to_have_cap=args.nice_to_have_cap,
            history_tail_cap=args.history_tail_cap,
            history_recent_cap=args.history_recent_cap,
            history_archive_cap=args.history_archive_cap,
            preserve_last_turns=args.preserve_last_turns,
            dynamic_context_postfix=dynamic_context_postfix,
            shared_prefix_text=shared_prefix_text,
            context_shift_enabled=args.enable_context_shift,
            context_shift_stage1_endpoint=args.context_shift_stage1_endpoint,
            context_shift_stage2_endpoint=args.context_shift_stage2_endpoint,
            context_shift_mode=args.context_shift_mode,
            context_shift_auto_threshold=args.context_shift_auto_threshold,
        ),
        hotspot_enabled=hotspot_enabled,
        hotspot_top_n=args.hotspot_top_n,
        metrics_db_enabled=(not args.disable_metrics_db),
        dynamic_max_tokens_enabled=dynamic_max_tokens_enabled,
        dynamic_max_tokens_min=args.dynamic_max_tokens_min,
        max_tokens_chat=args.max_tokens_chat,
        max_tokens_summary=args.max_tokens_summary,
        max_tokens_analysis=args.max_tokens_analysis,
        max_tokens_code=args.max_tokens_code,
        max_tokens_creative=args.max_tokens_creative,
        max_tokens_translation=args.max_tokens_translation,
    )

    tracker = PromptHotspotTracker(store_path=args.hotspot_store_path if hotspot_enabled else None)
    tracker.flush()
    metrics_store = MetricsStore(args.metrics_db_path) if cfg.metrics_db_enabled else None
    prompt_cache = TieredPromptCache(
        enabled=tiered_prompt_cache_enabled,
        warm_dir=args.prompt_cache_warm_dir,
        hot_max_entries=args.prompt_cache_hot_max_entries,
        hot_ttl_sec=args.prompt_cache_hot_ttl_sec,
        warm_ttl_sec=args.prompt_cache_warm_ttl_sec,
        prune_interval_sec=args.prompt_cache_prune_interval_sec,
        hot_hit_threshold=args.prompt_cache_hot_hit_threshold,
    )
    prompt_cache.start()

    warmup_mgr = ServiceWarmupManager(
        enabled=startup_warmup_enabled,
        cfg=cfg,
        timeout=args.timeout,
        warmup_requests=args.warmup_requests,
        warmup_max_tokens=args.warmup_max_tokens,
        prime_enabled=cache_priming_enabled,
        prime_top_n=args.cache_prime_top_n,
        hotspot_store_path=args.hotspot_store_path,
        prompt_cache=prompt_cache,
        residency_enabled=cache_residency_enabled,
        residency_interval_sec=args.cache_residency_interval_sec,
        residency_top_n=args.cache_residency_top_n,
        residency_force_hot=cache_residency_force_hot,
    )
    warmup_report = warmup_mgr.run()
    print(f"[warmup] {warmup_report}")
    warmup_mgr.start()

    auto_cache_tuner = AutoCacheRamTuner(
        enabled=auto_cache_ram_enabled,
        core_mgr=core_mgr,
        metrics_store=metrics_store,
        interval_sec=args.auto_cache_interval_sec,
        lookback_sec=args.auto_cache_lookback_sec,
        min_samples=args.auto_cache_min_samples,
        candidates_mb=auto_cache_candidates_mb,
        cooldown_sec=args.auto_cache_cooldown_sec,
        min_improve_score=args.auto_cache_min_improve,
    )
    auto_cache_tuner.start()

    ThunderServiceHandler.cfg = cfg
    ThunderServiceHandler.hotspot_tracker = tracker
    ThunderServiceHandler.metrics_store = metrics_store
    ThunderServiceHandler.core_mgr = core_mgr
    ThunderServiceHandler.auto_cache_tuner = auto_cache_tuner
    ThunderServiceHandler.prompt_cache = prompt_cache
    ThunderServiceHandler.warmup_mgr = warmup_mgr

    atexit.register(auto_cache_tuner.stop)
    atexit.register(prompt_cache.stop)
    atexit.register(warmup_mgr.stop)

    server = ThreadingHTTPServer((cfg.listen_host, cfg.listen_port), ThunderServiceHandler)
    print(
        f"[thunder-service] listen=http://{cfg.listen_host}:{cfg.listen_port} "
        f"upstream={cfg.upstream} layering={cfg.layering.enabled} hotspot={cfg.hotspot_enabled} "
        f"metrics_db={cfg.metrics_db_enabled} "
        f"prompt_cache={prompt_cache.enabled} auto_cache_ram={auto_cache_tuner.enabled} "
        f"dynamic_max_tokens={cfg.dynamic_max_tokens_enabled} warmup={startup_warmup_enabled} "
        f"cache_residency={cache_residency_enabled}"
    )

    def _handle_sigterm(_: int, __: Any) -> None:
        print("\n[thunder-service] shutting down...")
        server.shutdown()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    server.serve_forever()
    tracker.flush()
    auto_cache_tuner.stop()
    prompt_cache.stop()
    warmup_mgr.stop()
    if metrics_store is not None:
        metrics_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
