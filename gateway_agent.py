#!/usr/bin/env python3
"""
mini-openclaw Gateway 客户端 — 连接本地 OpenClaw Gateway

mini-openclaw 作为 Gateway 的 agent 后端：
1. 通过 WebSocket 连到本地的 OpenClaw Gateway
2. Gateway 把飞书/Telegram等消息转发过来
3. mini-openclaw 处理完把回复发回 Gateway
4. Gateway 负责把回复发到对应的渠道

使用方式:
  python3 gateway_agent.py                    # 启动 agent 服务
  python3 gateway_agent.py --port 18789       # 指定端口
  python3 gateway_agent.py --no-api           # Mock 模式
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import websockets

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


# ──── Mock ────

class MockLlmClient:
    def __init__(self, model_id="mock"):
        self.model_id = model_id
    async def chat_completion(self, req):
        return CompletionResponse(content=f"[Mock] {req.messages[-1].content}", finish_reason="stop")
    async def chat_completion_stream(self, req):
        yield StreamChunk(content_delta=f"[Mock processed]")
        yield StreamChunk(finish_reason="stop")


def _resolve_client(provider, mock=False):
    if mock or not provider.api_key or provider.api_key.startswith("YOUR_"):
        return MockLlmClient()
    if provider.api_type == "anthropic-messages":
        return AnthropicClient(provider.base_url, provider.api_key)
    return OpenAiClient(provider.base_url, provider.api_key)


# ──── Gateway 协议 ────

class GatewayClient:
    """OpenClaw Gateway WebSocket 客户端"""

    def __init__(self, url: str = "ws://127.0.0.1:18789", token: str = ""):
        self._url = url
        self._token = token
        self._ws = None
        self._pending: dict[str, asyncio.Future] = {}
        self._running = False
        self._listener_task = None

    async def connect(self) -> bool:
        """连接 Gateway（v3 协议）"""
        try:
            self._ws = await websockets.connect(self._url, max_size=10 * 1024 * 1024)

            challenge = json.loads(await self._ws.recv())
            nonce = challenge["payload"]["nonce"]

            import hashlib
            sig = hashlib.sha256(f"{nonce}:{self._token}".encode()).hexdigest()

            await self._ws.send(json.dumps({
                "type": "req",
                "id": "connect",
                "method": "connect",
                "params": {
                    "minProtocol": 3,
                    "maxProtocol": 3,
                    "client": {
                        "id": "cli",
                        "version": "0.1.0",
                        "platform": "linux",
                        "mode": "backend",
                    },
                    "role": "operator",
                    "scopes": ["operator.read", "operator.write"],
                    "auth": {"token": self._token},
                },
            }))

            resp = json.loads(await self._ws.recv())
            if not resp.get("ok"):
                print(f"  ❌ Gateway rejected: {resp.get('error', 'unknown')}")
                return False

            print(f"  ✅ Connected to Gateway")
            return True

        except Exception as e:
            print(f"  ⚠️  Cannot connect to Gateway: {e}")
            return False

    async def start_listener(self):
        """启动后台监听任务"""
        self._listener_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """监听 Gateway 事件循环"""
        self._running = True
        while self._running and self._ws:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=30)
                data = json.loads(raw)

                if data.get("type") == "res":
                    req_id = data.get("id")
                    if req_id in self._pending:
                        self._pending[req_id].set_result(data)
                elif data.get("type") == "event":
                    pass  # 忽略事件

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("  ⚠️  Gateway disconnected")
                break
            except Exception as e:
                if self._running:
                    pass

    async def send_agent_message(self, session_key: str, text: str) -> dict:
        """向 Gateway 发送 agent 请求"""
        req_id = uuid4().hex[:12]
        future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        await self._ws.send(json.dumps({
            "type": "req",
            "id": req_id,
            "method": "agent",
            "params": {
                "sessionKey": session_key,
                "prompt": text,
                "stream": False,
            },
        }))

        try:
            result = await asyncio.wait_for(future, timeout=120)
            return result
        except asyncio.TimeoutError:
            return {"error": "timeout"}
        finally:
            self._pending.pop(req_id, None)

    async def send_message(self, target: str, text: str, channel: str = "feishu"):
        """通过 Gateway 发送消息"""
        req_id = uuid4().hex[:12]
        await self._ws.send(json.dumps({
            "type": "req",
            "id": req_id,
            "method": "send",
            "params": {
                "target": target,
                "message": text,
                "channel": channel,
            },
        }))

    def stop(self):
        self._running = False


async def main():
    parser = argparse.ArgumentParser(description="mini-openclaw Gateway Agent")
    parser.add_argument("--port", type=int, default=18789, help="Gateway WS port")
    parser.add_argument("--url", default="", help="Full WS URL")
    parser.add_argument("--token", default="", help="Gateway auth token")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--test", help="发送测试消息并退出")
    args = parser.parse_args()

    # 配置
    if os.path.exists(args.config):
        config = ConfigLoader.load(args.config)
    else:
        from mini_openclaw.config.config import AppConfig
        config = AppConfig()

    # 模型
    registry = ModelRegistry.from_json_file(config.models_config_path)
    provider, model_info = registry.resolve_model(args.model)
    llm_client = _resolve_client(provider, mock=args.no_api)

    # 工具
    tools = ToolRegistry()
    tools.register(EchoTool())
    tools.register(ReadTool(config.workspace_dir))
    tools.register(WriteTool(config.workspace_dir))
    tools.register(ExecTool())
    tools.register(WebSearchTool())
    try:
        import subprocess
        subprocess.run(["playwright", "--version"], capture_output=True, timeout=5)
        from mini_openclaw.tools.playwright_tool import BrowserTool
        tools.register(BrowserTool())
    except Exception:
        pass

    # 会话
    os.makedirs(config.session_dir, exist_ok=True)
    store = SessionStore(config.session_dir)

    # 系统 Prompt
    os.makedirs(config.workspace_dir, exist_ok=True)
    builder = SystemPromptBuilder(config)
    system_prompt = builder.build(
        extra_prompt="你是 mini-openclaw，一个高效的 AI 助手。回答要简洁直接。",
        bootstrap_files=load_bootstrap_files(config.workspace_dir),
    )

    # Agent
    agent = AgentLoop(
        llm_client, tools, model_info.id,
        AgentConfig(system_prompt=system_prompt),
    )

    # Gateway 连接
    ws_url = args.url or f"ws://127.0.0.1:{args.port}"
    gw = GatewayClient(ws_url, args.token)

    print(f"🔌 Connecting to Gateway at {ws_url}...")
    if not await gw.connect():
        print("  ❌ Gateway not available. Start OpenClaw first.")
        print("  Or use --no-api for standalone mode.")
        sys.exit(1)

    # 启动后台监听
    await gw.start_listener()
    await asyncio.sleep(0.5)  # 等监听就绪

    # 测试模式
    if args.test:
        print(f"\n🧪 Test: sending message to Gateway...")
        result = await gw.send_agent_message("test", args.test)
        if "error" in result:
            print(f"Result: {result}")
        else:
            status = result.get("payload", {}).get("status", "ok")
            print(f"Result status: {status}")
        return

    # 消息处理
    async def handle_agent_event(data: dict):
        payload = data.get("payload", {})
        run_id = payload.get("runId", "?")
        status = payload.get("status", "")
        text = payload.get("text", "")

        if status == "completed" or status == "final":
            pass  # Gateway 自己处理了
        elif text and status in ("stream", "delta"):
            print(text, end="", flush=True)

    # 启动监听
    await gw.start_listener()

    print(f"\n{'='*50}")
    print(f"🤖 mini-openclaw Gateway Agent")
    print(f"   Model: {args.model}")
    print(f"   Gateway: {ws_url}")
    print(f"   Tools: {len(tools.list_tools())}")
    print(f"{'='*50}")
    print(f"Connected to Gateway. Agent ready to process messages.")
    print(f"Press Ctrl+C to stop.")

    try:
        # 保持运行
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        gw.stop()


if __name__ == "__main__":
    asyncio.run(main())
