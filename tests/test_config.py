"""配置系统测试"""

import json
import os
import tempfile
from pathlib import Path

from mini_openclaw.config.config import (
    AppConfig,
    ConfigLoader,
    create_default_config,
    ENV_PREFIX,
)


def test_default_config():
    """默认配置应该包含合理默认值"""
    config = create_default_config()
    assert config.models_config_path == "models.json"
    assert config.session_dir == "./sessions"
    assert config.workspace_dir == "./workspace"
    assert config.compaction_enabled is True
    assert config.context_pruning_enabled is True
    assert config.dm_scope == "main"
    assert config.max_concurrent_runs == 4


def test_from_dict():
    """从字典加载配置"""
    config = ConfigLoader.from_dict({
        "models_config_path": "/tmp/models.json",
        "session_dir": "/tmp/sessions",
        "max_concurrent_runs": 8,
        "compaction_enabled": False,
    })
    assert config.models_config_path == "/tmp/models.json"
    assert config.session_dir == "/tmp/sessions"
    assert config.max_concurrent_runs == 8
    assert config.compaction_enabled is False
    # 未提供的字段应该使用默认值
    assert config.workspace_dir == "./workspace"
    assert config.context_pruning_enabled is True


def test_load_from_json():
    """从 JSON 文件加载配置"""
    data = {
        "models_config_path": "/custom/models.json",
        "max_concurrent_runs": 2,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        config = ConfigLoader.load(tmp_path)
        assert config.models_config_path == "/custom/models.json"
        assert config.max_concurrent_runs == 2
        assert config.session_dir == "./sessions"  # 默认值
    finally:
        os.unlink(tmp_path)


def test_load_file_not_found():
    """不存在的文件应该抛出异常"""
    try:
        ConfigLoader.load("/tmp/nonexistent_config_12345.json")
        assert False, "应该抛出 FileNotFoundError"
    except FileNotFoundError:
        pass


def test_env_override():
    """环境变量应该覆盖配置"""
    os.environ[f"{ENV_PREFIX}MAX_CONCURRENT_RUNS"] = "16"
    os.environ[f"{ENV_PREFIX}DM_SCOPE"] = "per-channel-peer"
    os.environ[f"{ENV_PREFIX}COMPACTION_ENABLED"] = "false"

    config = ConfigLoader.from_dict({
        "max_concurrent_runs": 4,
        "dm_scope": "main",
        "compaction_enabled": True,
    })

    assert config.max_concurrent_runs == 16
    assert config.dm_scope == "per-channel-peer"
    assert config.compaction_enabled is False

    # 清理
    del os.environ[f"{ENV_PREFIX}MAX_CONCURRENT_RUNS"]
    del os.environ[f"{ENV_PREFIX}DM_SCOPE"]
    del os.environ[f"{ENV_PREFIX}COMPACTION_ENABLED"]


def test_to_dict_roundtrip():
    """to_dict 后应该可以恢复出相同的配置"""
    original = AppConfig(
        models_config_path="/a/b.json",
        session_dir="/data/sessions",
        max_concurrent_runs=10,
    )
    d = original.to_dict()
    restored = AppConfig.from_dict(d)
    assert restored.models_config_path == original.models_config_path
    assert restored.session_dir == original.session_dir
    assert restored.max_concurrent_runs == original.max_concurrent_runs
