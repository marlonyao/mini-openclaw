"""配置加载模块 - 参考 OpenClaw 的 config.js"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """应用程序配置，可序列化为 JSON"""

    models_config_path: str = "models.json"
    session_dir: str = "./sessions"
    workspace_dir: str = "./workspace"
    session_maintenance_mode: str = "warn"
    session_maintenance_prune_after_days: int = 30
    session_maintenance_max_entries: int = 500
    max_concurrent_runs: int = 4
    agent_timeout_seconds: int = 172800
    compaction_enabled: bool = True
    compaction_keep_recent_tokens: int = 8000
    context_pruning_enabled: bool = True
    context_pruning_ttl_seconds: int = 300
    dm_scope: str = "main"
    skills_dirs: list[str] = Field(default_factory=list)
    user_timezone: str = "Asia/Shanghai"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# 环境变量前缀
ENV_PREFIX = "MINI_OPENCLAW_"

# 环境变量到配置字段的映射
ENV_MAP: dict[str, str] = {
    "MODELS_CONFIG_PATH": "models_config_path",
    "SESSION_DIR": "session_dir",
    "WORKSPACE_DIR": "workspace_dir",
    "MAX_CONCURRENT_RUNS": "max_concurrent_runs",
    "AGENT_TIMEOUT_SECONDS": "agent_timeout_seconds",
    "DM_SCOPE": "dm_scope",
    "USER_TIMEZONE": "user_timezone",
    "COMPACTION_ENABLED": "compaction_enabled",
    "CONTEXT_PRUNING_ENABLED": "context_pruning_enabled",
}


class ConfigLoader:
    """配置加载器 - 支持 JSON 文件 + 环境变量覆盖"""

    @staticmethod
    def load(path: str | Path) -> AppConfig:
        """从 JSON 文件加载配置"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return ConfigLoader._apply_env_overrides(AppConfig.from_dict(data))

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AppConfig:
        """从字典加载（主要用于测试）"""
        return ConfigLoader._apply_env_overrides(AppConfig.from_dict(data))

    @staticmethod
    def _apply_env_overrides(config: AppConfig) -> AppConfig:
        """应用环境变量覆盖"""
        updates: dict[str, Any] = {}
        for env_key, field_name in ENV_MAP.items():
            full_key = f"{ENV_PREFIX}{env_key}"
            val = os.environ.get(full_key)
            if val is not None:
                field_type = AppConfig.model_fields[field_name].annotation
                if field_type is int or field_type == int:
                    updates[field_name] = int(val)
                elif field_type is bool or field_type == bool:
                    updates[field_name] = val.lower() in ("true", "1", "yes")
                else:
                    updates[field_name] = val

        if updates:
            return config.model_copy(update=updates)
        return config


def create_default_config() -> AppConfig:
    return AppConfig()
