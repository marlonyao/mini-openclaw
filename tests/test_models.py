"""模型注册和管理测试"""

import json
import os
import tempfile

import pytest

from mini_openclaw.models.registry import (
    ModelInfo,
    ModelProvider,
    ModelRegistry,
    ProviderNotFoundError,
    ModelNotFoundError,
)


def _sample_registry() -> ModelRegistry:
    """创建一个示例 registry 用于测试"""
    registry = ModelRegistry()
    registry.register_provider(ModelProvider(
        id="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test-key",
        api_type="openai-completions",
        models=[
            ModelInfo(id="deepseek-chat", name="DeepSeek Chat", context_window=200000),
            ModelInfo(id="deepseek-reasoner", name="DeepSeek Reasoner", context_window=200000),
        ],
    ))
    registry.register_provider(ModelProvider(
        id="moonshot",
        base_url="https://api.moonshot.cn/v1",
        api_key="sk-moon-key",
        models=[
            ModelInfo(id="kimi-k2.5", name="Kimi K2.5", context_window=200000, input_types=["text", "image"]),
        ],
    ))
    return registry


def test_register_and_list_providers():
    registry = _sample_registry()
    providers = registry.list_providers()
    assert "deepseek" in providers
    assert "moonshot" in providers
    assert len(providers) == 2


def test_resolve_model_with_provider():
    registry = _sample_registry()
    provider, model = registry.resolve_model("deepseek/deepseek-chat")
    assert provider.id == "deepseek"
    assert model.id == "deepseek-chat"
    assert model.context_window == 200000


def test_resolve_model_without_provider():
    registry = _sample_registry()
    provider, model = registry.resolve_model("kimi-k2.5")
    assert provider.id == "moonshot"
    assert model.id == "kimi-k2.5"


def test_resolve_model_not_found():
    registry = _sample_registry()
    with pytest.raises(ModelNotFoundError):
        registry.resolve_model("deepseek/nonexistent-model")


def test_resolve_provider_not_found():
    registry = _sample_registry()
    with pytest.raises(ProviderNotFoundError):
        registry.resolve_model("nonexistent/model-xyz")


def test_provider_not_found():
    registry = _sample_registry()
    with pytest.raises(ProviderNotFoundError, match="unknown"):
        registry.get_provider("unknown")


def test_list_all_models():
    registry = _sample_registry()
    models = registry.list_models()
    assert len(models) == 3  # 2 + 1


def test_list_models_by_provider():
    registry = _sample_registry()
    models = registry.list_models("deepseek")
    assert len(models) == 2
    assert all(m.id.startswith("deepseek") for m in models)


def test_get_provider():
    registry = _sample_registry()
    p = registry.get_provider("deepseek")
    assert p.base_url == "https://api.deepseek.com/v1"
    assert p.api_key == "sk-test-key"


def test_from_dict_openclaw_format():
    """支持 OpenClaw 的 models.json 格式"""
    data = {
        "providers": {
            "deepseek": {
                "baseUrl": "https://api.deepseek.com/v1",
                "apiKey": "sk-secret",
                "api": "openai-completions",
                "models": [
                    {"id": "deepseek-chat", "name": "DeepSeek Chat", "contextWindow": 200000},
                ],
            },
            "anthropic": {
                "baseUrl": "https://api.anthropic.com/v1",
                "apiKey": "sk-ant",
                "api": "anthropic-messages",
                "models": [
                    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
                ],
            },
        },
    }
    registry = ModelRegistry.from_dict(data)
    assert "deepseek" in registry.list_providers()
    assert "anthropic" in registry.list_providers()

    # 验证 camelCase 字段被正确映射
    provider, model = registry.resolve_model("deepseek/deepseek-chat")
    assert provider.base_url == "https://api.deepseek.com/v1"
    assert provider.api_type == "openai-completions"


def test_from_json_file():
    data = {
        "providers": {
            "glmcode": {
                "base_url": "https://open.bigmodel.cn/api/anthropic/v1",
                "api_key": "sk-glm",
                "api_type": "anthropic-messages",
                "models": [
                    {"id": "GLM-5.1", "name": "GLM-5.1"},
                ],
            },
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        registry = ModelRegistry.from_json_file(tmp_path)
        assert "glmcode" in registry.list_providers()
        provider, model = registry.resolve_model("glmcode/GLM-5.1")
        assert model.id == "GLM-5.1"
    finally:
        os.unlink(tmp_path)


def test_empty_registry():
    registry = ModelRegistry()
    assert registry.list_providers() == []
    assert registry.list_models() == []


def test_resolve_model_duplicate_name():
    """同名模型在多个 provider 中存在时，按注册顺序返回第一个"""
    registry = ModelRegistry()
    registry.register_provider(ModelProvider(
        id="provider_a", models=[ModelInfo(id="gpt-4", name="GPT-4 A")],
    ))
    registry.register_provider(ModelProvider(
        id="provider_b", models=[ModelInfo(id="gpt-4", name="GPT-4 B")],
    ))
    provider, model = registry.resolve_model("gpt-4")
    assert provider.id == "provider_a"
    assert model.name == "GPT-4 A"
