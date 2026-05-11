#!/usr/bin/env python3
"""
mini-openclaw CLI: 交互式 Agent 命令行
"""

import asyncio
import os
from pathlib import Path

from mini_openclaw.config.config import ConfigLoader
from mini_openclaw.models.registry import ModelRegistry
from mini_openclaw.models.provider import (
    OpenAiClient, AnthropicClient,
    CompletionResponse, StreamChunk,
)
from mini_openclaw.session.store import SessionStore
from mini_openclaw.session.router import SessionRouter
from mini_openclaw.agent.loop import AgentLoop, AgentConfig
from mini_openclaw.tools.tool import ToolRegistry, EchoTool, ReadTool, WriteTool
from mini_openclaw.prompt.builder import SystemPromptBuilder, load_bootstrap_files
from mini_openclaw.context.compaction import ContextAssembler
from mini_openclaw.memory.memory import MemoryStore, MemorySearcher


class MockLlmClient:
    """模拟 LLM 客户端 - 不走真实 API"""
    def __init__(self, model_id: str = "mock-model"):
        self.model_id = model_id

    async def chat_completion(self, request):
        return CompletionResponse(
            content=f"[Mock] I received your message. Model: {self.model_id}",
            finish_reason="stop",
        )

    async def chat_completion_stream(self, request):
        yield StreamChunk(content_delta=f"[Mock] Responding via {self.model_id}. ")
        yield StreamChunk(content_delta="This is a streamed mock response.")
        yield StreamChunk(finish_reason="stop")


def _resolve_provider_client(provider, use_mock: bool = False):
    if use_mock:
        return MockLlmClient(model_id=provider.id if provider else "mock")

    api_type = provider.api_type
    base_url = provider.base_url
    api_key = provider.api_key

    if not api_key or api_key.startswith("YOUR_"):
        print(f"  ⚠️  No API key for {provider.id}, using mock mode")
        return MockLlmClient()

    if api_type == "anthropic-messages":
        return AnthropicClient(base_url, api_key)
    else:
        return OpenAiClient(base_url, api_key)


async def interactive_loop(config_file: str, model_ref: str, no_api: bool = False):
    """交互式 Agent 循环"""

    # 1. 加载配置
    print(f"📋 Loading config from {config_file}...")
    if not os.path.exists(config_file):
        from mini_openclaw.config.config import AppConfig
        config = AppConfig()
    else:
        config = ConfigLoader.load(config_file)
    print(f"  Workspace: {config.workspace_dir}")

    # 2. 加载模型
    models_path = config.models_config_path
    print(f"📦 Loading models from {models_path}...")
    if os.path.exists(models_path):
        registry = ModelRegistry.from_json_file(models_path)
        try:
            provider, model_info = registry.resolve_model(model_ref)
            print(f"  Provider: {provider.id}")
            print(f"  Model: {model_info.id} ({model_info.name or model_info.id})")
        except Exception as e:
            print(f"  ❌ Failed to resolve model '{model_ref}': {e}")
            print(f"  Available models:")
            for p in registry.list_providers():
                for m in registry.list_models(p):
                    print(f"    {p}/{m.id}")
            return
    else:
        print(f"  ❌ Models file not found: {models_path}")
        return

    # 3. 创建 LLM 客户端
    print("🔌 Creating LLM client...")
    llm_client = _resolve_provider_client(provider, use_mock=no_api)

    # 4. 注册工具
    print("🛠️  Registering tools...")
    tools = ToolRegistry()
    tools.register(EchoTool())
    tools.register(ReadTool(workspace_dir=config.workspace_dir))
    tools.register(WriteTool(workspace_dir=config.workspace_dir))
    tool_names = [t.name for t in tools.list_tools()]
    print(f"  Tools: {', '.join(tool_names)}")

    # 5. 创建会话
    print("💾 Initializing session store...")
    os.makedirs(config.session_dir, exist_ok=True)
    store = SessionStore(config.session_dir)
    router = SessionRouter(dm_scope=config.dm_scope)

    session_key = router.route("cli", "interactive_user", "direct")
    session = await store.load_session(session_key)
    if session is None:
        session = store.create_session(session_key)
        print(f"  New session: {session_key}")
    else:
        print(f"  Loaded session: {session_key} ({len(session.messages)} messages)")

    # 6. 加载 bootstrap 文件
    bootstrap = load_bootstrap_files(config.workspace_dir)

    # 7. 构建系统 prompt
    prompt_builder = SystemPromptBuilder(config)
    system_prompt = prompt_builder.build(
        extra_prompt="You are a helpful coding assistant.",
        bootstrap_files=bootstrap,
    )

    # 8. 记忆系统
    memory_store = MemoryStore(config.workspace_dir)
    memory_searcher = MemorySearcher(memory_store)

    # 9. 创建 Agent Loop
    agent_config = AgentConfig(
        system_prompt=system_prompt,
        max_tool_rounds=10,
        temperature=0.0,
    )

    agent = AgentLoop(
        llm_client=llm_client,
        tool_registry=tools,
        model_id=model_info.id,
        config=agent_config,
    )

    # 10. 上下文组装器
    assembler = ContextAssembler()

    # 11. REPL
    mode_str = "MOCK" if no_api else f"LIVE ({model_ref})"
    print(f"\n{'='*60}")
    print(f"🚀 mini-openclaw REPL ready!")
    print(f"   Mode: {mode_str}")
    print(f"   Session: {session_key}")
    print(f"   Tools: {', '.join(tool_names)}")
    print(f"{'='*60}")
    print("  Type 'exit' or 'quit' to exit")
    print("  Type '/info' for session info")
    print("  Type '/save' to save session")

    while True:
        try:
            user_input = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        if user_input == "/info":
            print(f"\n📊 Session info:")
            print(f"  Key: {session.key}")
            print(f"  Messages: {len(session.messages)}")
            print(f"  Token estimate: ~{session.token_estimate()}")
            print(f"  Model used: {model_ref}")
            continue

        if user_input == "/save":
            print("  💾 Session saved to disk")
            continue

        # 执行 Agent 循环
        print("\n🤖 mini-openclaw: ", end="", flush=True)
        try:
            async for event in agent.run(session, user_input):
                if event.kind == "text":
                    print(event.data, end="", flush=True)
                elif event.kind == "tool_start":
                    name = event.data.get("name", "?")
                    print(f"  🔧 [{name}] Calling...")
                elif event.kind == "tool_end":
                    result = event.data
                    status = "❌" if result.get("is_error") else "✅"
                    content_preview = result.get("content", "")[:60]
                    print(f"  {status} [{result.get('tool_call_id', '?')[:8]}] {content_preview}")
                    print("  💬 ", end="", flush=True)
                elif event.kind == "error":
                    print(f"\n  ❌ Error: {event.data}")
                elif event.kind == "done":
                    print()

            # 保存会话
            if len(session.messages) >= 2:
                await store.save_message(session_key, session.messages[-2])
                await store.save_message(session_key, session.messages[-1])

        except Exception as e:
            print(f"\n  ❌ Agent error: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="mini-openclaw interactive agent")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--model", default="deepseek/deepseek-chat", help="Model reference (provider/model)")
    parser.add_argument("--no-api", action="store_true", help="Mock mode (no API calls)")
    args = parser.parse_args()

    asyncio.run(interactive_loop(args.config, args.model, args.no_api))


if __name__ == "__main__":
    main()
