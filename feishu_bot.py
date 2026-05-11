#!/usr/bin/env python3
"""
mini-openclaw 飞书机器人启动器

通过飞书 WebSocket 长连接接收消息，经过 Agent 处理后回复。

使用方式:
  python3 feishu_bot.py                                # 交互模式
  python3 feishu_bot.py --prompt "你好"                # 单次对话
  python3 feishu_bot.py --no-api                       # Mock 模式测试

环境变量（或编辑此文件底部）:
  FEISHU_APP_ID     飞书应用 App ID
  FEISHU_APP_SECRET 飞书应用 App Secret
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 确保能找到 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mini_openclaw.channel.feishu import FeishuChannel, FeishuMessage
from mini_openclaw.config.config import ConfigLoader
from mini_openclaw.models.registry import ModelRegistry
from mini_openclaw.models.provider import OpenAiClient, AnthropicClient, CompletionResponse, StreamChunk
from mini_openclaw.session.store import SessionStore
from mini_openclaw.session.router import SessionRouter
from mini_openclaw.agent.loop import AgentLoop, AgentConfig
from mini_openclaw.tools.tool import ToolRegistry, EchoTool, ReadTool, WriteTool
from mini_openclaw.tools.exec_tool import ExecTool
from mini_openclaw.tools.web_search_tool import WebSearchTool
from mini_openclaw.prompt.builder import SystemPromptBuilder, load_bootstrap_files
from mini_openclaw.context.compaction import ContextAssembler


class MockLlmClient:
    """模拟 LLM 客户端"""
    def __init__(self, model_id: str = "mock"):
        self.model_id = model_id
    async def chat_completion(self, request):
        return CompletionResponse(content=f"[Mock] Received: {len(request.messages)} msgs", finish_reason="stop")
    async def chat_completion_stream(self, request):
        yield StreamChunk(content_delta=f"[Mock] Received.")
        yield StreamChunk(finish_reason="stop")


def _resolve_client(provider, mock=False):
    if mock or not provider.api_key or provider.api_key.startswith("YOUR_"):
        return MockLlmClient(provider.id if provider else "mock")
    if provider.api_type == "anthropic-messages":
        return AnthropicClient(provider.base_url, provider.api_key)
    return OpenAiClient(provider.base_url, provider.api_key)


async def main():
    parser = argparse.ArgumentParser(description="mini-openclaw Feishu Bot")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--app-id", default=os.environ.get("FEISHU_APP_ID", ""))
    parser.add_argument("--app-secret", default=os.environ.get("FEISHU_APP_SECRET", ""))
    parser.add_argument("--prompt", "-p", help="发送一条消息后退出（测试用）")
    args = parser.parse_args()

    # ── 配置 ──
    if os.path.exists(args.config):
        config = ConfigLoader.load(args.config)
    else:
        from mini_openclaw.config.config import AppConfig
        config = AppConfig()

    # ── 模型 ──
    registry = ModelRegistry.from_json_file(config.models_config_path)
    provider, model_info = registry.resolve_model(args.model)
    llm_client = _resolve_client(provider, mock=args.no_api)

    # ── 工具 ──
    tools = ToolRegistry()
    tools.register(EchoTool())
    tools.register(ReadTool(config.workspace_dir))
    tools.register(WriteTool(config.workspace_dir))
    tools.register(ExecTool())
    tools.register(WebSearchTool())
    # 检查 Playwright
    try:
        import subprocess
        subprocess.run(["playwright", "--version"], capture_output=True, timeout=5)
        from mini_openclaw.tools.playwright_tool import BrowserTool
        tools.register(BrowserTool())
        print("  ✅ Browser tool available")
    except Exception:
        print("  ⚠️  Playwright not installed, no browser tool")

    # ── 会话 ──
    os.makedirs(config.session_dir, exist_ok=True)
    store = SessionStore(config.session_dir)
    router = SessionRouter(dm_scope="per-channel-peer")

    # ── 系统 Prompt ──
    os.makedirs(config.workspace_dir, exist_ok=True)
    bootstrap = load_bootstrap_files(config.workspace_dir)
    builder = SystemPromptBuilder(config)
    system_prompt = builder.build(
        extra_prompt="你是 mini-openclaw，一个 AI 助手。回答问题要简洁。",
        bootstrap_files=bootstrap,
    )

    # ── Agent ──
    agent = AgentLoop(
        llm_client, tools, model_info.id,
        AgentConfig(system_prompt=system_prompt, max_tool_rounds=10),
    )

    # ── 飞书配置 ──
    app_id = args.app_id
    app_secret = args.app_secret

    if not app_id or not app_secret:
        print("❌ FEISHU_APP_ID and FEISHU_APP_SECRET required")
        print("   Set env vars or use --app-id / --app-secret")
        sys.exit(1)

    # ── 创建飞书通道 ──
    channel = FeishuChannel(app_id, app_secret)

    # ── 消息处理函数 ──
    async def agent_runner(msg: FeishuMessage, chat_id: str, sender_id: str) -> str:
        nonlocal agent, store, router, model_info, config

        session_key = router.route("feishu", sender_id, msg.chat_type)
        session = await store.load_session(session_key)
        if session is None:
            session = store.create_session(session_key)

        print(f"\n💬 [{msg.chat_type}] {sender_id[:12]}: {msg.text[:60]}")

        # 执行 agent
        response_parts: list[str] = []
        try:
            async for event in agent.run(session, msg.text):
                if event.kind == "text":
                    response_parts.append(event.data)
                    print(event.data, end="", flush=True)
                elif event.kind == "tool_start":
                    print(f"\n  🔧 {event.data.get('name', '?')}", end="")
                elif event.kind == "tool_end":
                    r = event.data
                    status = "✓" if not r.get("is_error") else "✗"
                    print(f" {status}", end="")
                elif event.kind == "error":
                    print(f"\n  ❌ {event.data}")
            print()

            # 保存会话
            if len(session.messages) >= 2:
                await store.save_message(session_key, session.messages[-2])
                await store.save_message(session_key, session.messages[-1])

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {e}"

        return "".join(response_parts) if response_parts else "(empty response)"

    channel.set_agent_runner(agent_runner)

    # ── 测试模式：发送消息到飞书 ──
    if args.prompt:
        import httpx
        # 先执行 agent 获得回复
        test_msg = FeishuMessage(
            message_id="test_msg", chat_id="test_chat",
            sender_id="test_sender", text=args.prompt, chat_type="p2p",
        )
        response = await agent_runner(test_msg, "test_chat", "test_sender")
        print(f"\n🤖 Response: {response[:300]}")

        # 发送到飞书
        print(f"\n📤 Sending to Feishu...")
        await channel.client.send_text(
            "ou_5516922192cd1b0ef8d244a6ed66119d",
            f"🤖 mini-openclaw says:\n{response}",
            receive_id_type="open_id",
        )
        print(f"✅ Sent!")
        return

    # ── 启动 ──
    print(f"\n{'='*50}")
    print(f"🤖 mini-openclaw Feishu Bot")
    print(f"   Model: {args.model}")
    print(f"   Mode: {'MOCK' if args.no_api else 'LIVE'}")
    print(f"   Tools: {len(tools.list_tools())}")
    print(f"{'='*50}\n")

    await channel.start()


if __name__ == "__main__":
    asyncio.run(main())
