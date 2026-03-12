#!/usr/bin/env python3
"""
Clawgate Configuration Loader

从全局配置文件 ~/.openclaw/config.yaml 加载配置
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class OpenClawConfig:
    """OpenClaw 全局配置加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为 ~/.openclaw/config.yaml
        """
        if config_path is None:
            config_path = os.environ.get(
                'OPENCLAW_CONFIG',
                os.path.expanduser('~/.openclaw/config.yaml')
            )

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create it or set OPENCLAW_CONFIG environment variable"
            )

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        # 展开所有路径
        self._expand_paths(self._config)

    def _expand_paths(self, obj):
        """递归展开配置中的 ~ 路径"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower()):
                    obj[key] = os.path.expanduser(value)
                elif isinstance(value, (dict, list)):
                    self._expand_paths(value)
        elif isinstance(obj, list):
            for item in obj:
                self._expand_paths(item)

    def get(self, key_path: str, default=None):
        """
        获取配置值（支持点号路径）

        Args:
            key_path: 配置键路径，如 "model.path" 或 "lmcache.enabled"
            default: 默认值

        Returns:
            配置值

        Example:
            config.get('model.path')
            config.get('lmcache.enabled', False)
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def model(self) -> Dict[str, Any]:
        """模型配置"""
        return self._config.get('model', {})

    @property
    def server(self) -> Dict[str, Any]:
        """服务器配置"""
        return self._config.get('server', {})

    @property
    def lmcache(self) -> Dict[str, Any]:
        """LMCache 配置"""
        return self._config.get('lmcache', {})

    @property
    def contextpilot(self) -> Dict[str, Any]:
        """ContextPilot 配置"""
        return self._config.get('contextpilot', {})

    @property
    def clawgate(self) -> Dict[str, Any]:
        """Clawgate 配置"""
        return self._config.get('clawgate', {})

    @property
    def performance(self) -> Dict[str, Any]:
        """性能配置"""
        return self._config.get('performance', {})

    @property
    def debug(self) -> Dict[str, Any]:
        """调试配置"""
        return self._config.get('debug', {})

    @property
    def thunderllama_url(self) -> str:
        """ThunderLLAMA 服务 URL（自动生成）"""
        host = self.server.get('host', '127.0.0.1')
        port = self.server.get('port', 30000)
        return f"http://{host}:{port}"

    def to_env_vars(self) -> Dict[str, str]:
        """
        转换为环境变量字典

        Returns:
            环境变量字典
        """
        env = {}

        # LMCache 环境变量
        if self.lmcache.get('enabled'):
            env['THUNDER_LMCACHE'] = '1'
            env['THUNDER_LMCACHE_DISK_PATH'] = self.lmcache.get('disk_path', '')

        # ThunderLLAMA URL
        env['THUNDERLLAMA_URL'] = self.thunderllama_url

        # OpenMP
        if self._config.get('environment', {}).get('kmp_duplicate_lib_ok'):
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # ContextPilot
        if self.contextpilot.get('server_url'):
            env['CONTEXTPILOT_URL'] = self.contextpilot['server_url']

        if self.lmcache.get('eviction_sync_webhook'):
            env['CONTEXTPILOT_INDEX_URL'] = self.lmcache['eviction_sync_webhook']

        return env

    def apply_env_vars(self):
        """应用环境变量到当前进程"""
        for key, value in self.to_env_vars().items():
            os.environ[key] = value

    def __repr__(self):
        return f"OpenClawConfig(path={self.config_path})"

    def __str__(self):
        """格式化输出配置"""
        lines = [
            "=" * 70,
            "  OpenClaw Configuration",
            "=" * 70,
            "",
            "Model:",
            f"  Path:        {self.model.get('path')}",
            f"  GPU Layers:  {self.model.get('gpu_layers')}",
            f"  Context:     {self.model.get('context_size')}",
            f"  Batch:       {self.model.get('batch_size')}",
            "",
            "Server:",
            f"  URL:         {self.thunderllama_url}",
            f"  Parallel:    {self.server.get('n_parallel')}",
            "",
            "LMCache:",
            f"  Enabled:     {self.lmcache.get('enabled')}",
            f"  Disk Path:   {self.lmcache.get('disk_path')}",
            f"  L2 Size:     {self.lmcache.get('l2_size_gb')} GB",
            f"  L3 Size:     {self.lmcache.get('l3_size_gb')} GB",
            "",
            "ContextPilot:",
            f"  Mode:        {self.contextpilot.get('mode')}",
            f"  GPU:         {self.contextpilot.get('use_gpu')}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================

def load_config(config_path: Optional[str] = None) -> OpenClawConfig:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，默认为 ~/.openclaw/config.yaml

    Returns:
        OpenClawConfig 对象

    Example:
        from config_loader import load_config

        config = load_config()
        print(config.thunderllama_url)
        print(config.lmcache['enabled'])
    """
    return OpenClawConfig(config_path)


# ============================================================================
# CLI 工具
# ============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        config = load_config()

        if cmd == "show":
            print(config)

        elif cmd == "get":
            if len(sys.argv) < 3:
                print("Usage: config_loader.py get <key.path>")
                sys.exit(1)

            key_path = sys.argv[2]
            value = config.get(key_path)
            print(value)

        elif cmd == "env":
            for key, value in config.to_env_vars().items():
                print(f"{key}={value}")

        elif cmd == "json":
            print(json.dumps(config._config, indent=2))

        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)

    else:
        # 默认：显示配置
        config = load_config()
        print(config)
        print("\nUsage:")
        print("  python3 config_loader.py show     # Show configuration")
        print("  python3 config_loader.py get <key>  # Get specific value")
        print("  python3 config_loader.py env      # Show environment variables")
        print("  python3 config_loader.py json     # Dump as JSON")
