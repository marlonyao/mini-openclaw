"""模型注册和管理 - 参考 OpenClaw 的 model-catalog.runtime.js"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    name: str = ""
    reasoning: bool = False
    input_types: list[str] = Field(default_factory=lambda: ["text"])
    context_window: int = 128000
    max_tokens: int = 4096
    cost: dict[str, float] = Field(default_factory=dict)


class ModelProvider(BaseModel):
    """LLM Provider 配置"""
    id: str = ""
    base_url: str = ""
    api_key: str = ""
    api_type: str = "openai-completions"  # "openai-completions" | "anthropic-messages"
    models: list[ModelInfo] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class ModelRegistryError(Exception):
    """模型注册中心异常"""
    pass


class ProviderNotFoundError(ModelRegistryError):
    """Provider 不存在"""
    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        super().__init__(f"Provider not found: {provider_id}")


class ModelNotFoundError(ModelRegistryError):
    """模型不存在"""
    def __init__(self, model_id: str, provider_id: str | None = None):
        self.model_id = model_id
        self.provider_id = provider_id
        if provider_id:
            super().__init__(f"Model '{model_id}' not found in provider '{provider_id}'")
        else:
            super().__init__(f"Model not found: {model_id}")


class ModelRegistry:
    """模型注册中心 - 管理多个 provider 和模型"""

    def __init__(self):
        self._providers: dict[str, ModelProvider] = {}

    def register_provider(self, provider: ModelProvider) -> None:
        """注册一个 LLM provider"""
        pid = provider.id or provider.base_url
        self._providers[pid] = provider

    def resolve_model(self, model_ref: str) -> tuple[ModelProvider, ModelInfo]:
        """
        解析模型引用字符串，返回 (provider, model_info)

        格式:
        - "provider/model_id" → 从指定 provider 查找模型
        - "model_id" → 遍历所有 provider 查找
        """
        if "/" in model_ref:
            provider_id, model_id = model_ref.split("/", 1)
            provider = self.get_provider(provider_id)
            for m in provider.models:
                if m.id == model_id:
                    return provider, m
            raise ModelNotFoundError(model_id, provider_id)
        else:
            # 遍历所有 provider
            for provider in self._providers.values():
                for m in provider.models:
                    if m.id == model_ref:
                        return provider, m
            raise ModelNotFoundError(model_ref)

    def list_providers(self) -> list[str]:
        """列出所有注册的 provider ID"""
        return list(self._providers.keys())

    def list_models(self, provider_id: str | None = None) -> list[ModelInfo]:
        """列出模型，可按 provider 过滤"""
        if provider_id:
            provider = self.get_provider(provider_id)
            return list(provider.models)
        result: list[ModelInfo] = []
        for p in self._providers.values():
            result.extend(p.models)
        return result

    def get_provider(self, provider_id: str) -> ModelProvider:
        """获取指定 provider"""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ProviderNotFoundError(provider_id)
        return provider

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ModelRegistry:
        """
        从 models.json 格式的字典加载

        OpenClaw 格式:
        {
          "providers": {
            "deepseek": {
              "baseUrl": "https://api.deepseek.com/v1",
              "apiKey": "sk-xxx",
              "api": "openai-completions",
              "models": [...]
            }
          }
        }
        """
        registry = ModelRegistry()
        providers_data = data.get("providers", data)
        for provider_id, provider_cfg in providers_data.items():
            # 适配 snake_case 和 camelCase
            base_url = provider_cfg.get("base_url") or provider_cfg.get("baseUrl", "")
            api_key = provider_cfg.get("api_key") or provider_cfg.get("apiKey", "")
            api_type = provider_cfg.get("api") or provider_cfg.get("api_type", "openai-completions")

            models_list = provider_cfg.get("models", [])
            models = [ModelInfo(**m) for m in models_list]

            provider = ModelProvider(
                id=provider_id,
                base_url=base_url,
                api_key=api_key,
                api_type=api_type,
                models=models,
            )
            registry.register_provider(provider)
        return registry

    @staticmethod
    def from_json_file(path: str | Path) -> ModelRegistry:
        """从 JSON 文件加载"""
        with open(path) as f:
            data = json.load(f)
        return ModelRegistry.from_dict(data)
