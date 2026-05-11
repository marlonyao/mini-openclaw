#!/usr/bin/env python3
"""
mini-openclaw CLI — 交互式/单次 Agent 命令行工具

用法:
  # 交互模式（默认）
  python3 main.py

  # 指定模型
  python3 main.py --model deepseek/deepseek-chat

  # 单次执行（非交互）
  python3 main.py --prompt "Say hello" --no-stream

  # Mock 模式（不调用真实 API）
  python3 main.py --no-api
"""

import argparse
import asyncio
import json
import os
import sys
import time
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
from mini_openclaw.tools.playwright_tool import BrowserTool
from mini_openclaw.tools.exec_tool import ExecTool
from mini_openclaw.tools.web_search_tool import WebSearchTool
from mini_openclaw.prompt.builder import SystemPromptBuilder, load_bootstrap_files
from mini_openclaw.context.compaction import (
    ContextAssembler, compact_session, build_compaction_prompt,
)
from mini_openclaw.memory.memory import MemoryStore, MemorySearcher

# ──── 常量与着色 ────

COLOR = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "end": "\033[0m",
}


def c(code: str, text: str) -> str:
    return f"{COLOR.get(code, '')}{text}{COLOR['end']}"


def print_banner():
    banner = f"""
  {c('cyan', '╔══════════════════════════════════════════╗')}
  {c('cyan', '║')}       {c('bold', 'mini-openclaw')} — Simplified AI Agent      {c('cyan', '║')}
  {c('cyan', '╚══════════════════════════════════════════╝')}
"""
    print(banner)


# ──── Mock LLM ────


class MockLlmClient:
    """模拟 LLM 客户端"""
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


# ──── 辅助函数 ────


def _resolve_provider_client(provider, use_mock: bool = False):
    if use_mock:
        return MockLlmClient(model_id=provider.id if provider else "mock"), True

    if not provider.api_key or provider.api_key.startswith("YOUR_"):
        print(f"  {c('yellow', '⚠')} No API key for {provider.id}, using mock mode")
        return MockLlmClient(), True

    if provider.api_type == "anthropic-messages":
        return AnthropicClient(provider.base_url, provider.api_key), False
    else:
        return OpenAiClient(provider.base_url, provider.api_key), False


def register_default_tools(workspace_dir: str = ".", include_browser: bool = True) -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(EchoTool())
    tools.register(ReadTool(workspace_dir=workspace_dir))
    tools.register(WriteTool(workspace_dir=workspace_dir))
    tools.register(ExecTool())
    tools.register(WebSearchTool())
    try:
        import subprocess
        subprocess.run(["playwright", "--version"], capture_output=True, timeout=5)
        if include_browser:
            tools.register(BrowserTool())
    except (FileNotFoundError, Exception):
        pass  # Playwright not installed, skip browser tool
    return tools


# ──── 会话管理器 ────


class SessionManager:
    """管理当前会话和 Store"""

    def __init__(self, store: SessionStore, router: SessionRouter):
        self.store = store
        self.router = router
        self.current_key: str = ""
        self.current_session = None  # type: ignore

    async def create_or_load(self, key: str) -> bool:
        """加载或创建会话，返回是否是新建"""
        self.current_key = key
        session = await self.store.load_session(key)
        if session is None:
            self.current_session = self.store.create_session(key)
            return True  # new
        self.current_session = session
        return False  # loaded

    def session(self):
        return self.current_session

    async def save_current(self):
        """保存最新两条消息到磁盘"""
        sess = self.current_session
        if sess is None or len(sess.messages) < 2:
            return
        await self.store.save_message(self.current_key, sess.messages[-2])
        await self.store.save_message(self.current_key, sess.messages[-1])

    async def switch_session(self, key: str):
        await self.create_or_load(key)

    async def new_session(self):
        """创建全新的会话（不同 session_id）"""
        self.current_session = self.store.create_session(self.current_key)


# ──── 命令处理器 ────


def cmd_info(session, model_ref: str, config) -> str:
    sess = session.session()
    lines = [
        f"\n{c('bold', '📊 Session Info')}",
        f"  Key:       {session.current_key}",
        f"  ID:        {sess.id}",
        f"  Messages:  {len(sess.messages)}",
        f"  Estimate:  ~{sess.token_estimate()} tokens",
        f"  Model:     {model_ref}",
        f"  Created:   {sess.created_at.strftime('%Y-%m-%d %H:%M')}",
        f"  Updated:   {sess.updated_at.strftime('%Y-%m-%d %H:%M')}",
    ]
    # 显示最后几条消息摘要
    if sess.messages:
        lines.append(f"  {c('dim', '─' * 40)}")
        lines.append(f"  {c('bold', 'Recent messages:')}")
        for msg in sess.messages[-6:]:
            role_icon = {"user": "🧑", "assistant": "🤖", "tool": "🔧", "system": "⚙️"}
            icon = role_icon.get(msg.role, "❓")
            content = (msg.content or "")[:50]
            lines.append(f"    {icon} {c('dim', msg.role)}: {content}")
    return "\n".join(lines)


def cmd_list_sessions(store, sm: SessionManager) -> str:
    metas = store.list_sessions()
    if not metas:
        return f"  {c('yellow', 'No sessions found.')}"
    lines = [f"\n{c('bold', '📋 Sessions')}"]
    for m in sorted(metas, key=lambda x: x.updated_at, reverse=True):
        marker = c("green", "◀") if m.key == sm.current_key else " "
        lines.append(
            f"  {marker} {c('bold', m.key)}"
            f"  {c('dim', f'{m.message_count} msgs, ~{m.token_estimate} tok')}"
            f"  {c('dim', m.updated_at.strftime('%m-%d %H:%M'))}"
        )
    return "\n".join(lines)


def cmd_help() -> str:
    return f"""
{c('bold', 'Commands')}
  /help, /h        显示帮助
  /info, /i        显示当前会话信息
  /sessions, /ss   列出所有会话
  /session <key>   切换到指定会话
  /new /reset      重置/清空当前会话
  /compact         压缩当前会话历史
  /model <ref>     切换模型
  /tools           列出可用工具
  /save            保存会话到磁盘
  /save <path>     导出会话到文件
  /exit, /quit     退出
"""


async def cmd_compact(session, sm: SessionManager, llm_client) -> str:
    sess = session.session()
    if len(sess.messages) < 4:
        return f"  {c('yellow', '⚠')} Too few messages to compact."

    prompt = build_compaction_prompt(sess.messages[:-2])
    print(f"  {c('yellow', '📦')} Compacting {len(sess.messages)} messages...")

    try:
        resp = await llm_client.chat_completion(
            type("Req", (), {
                "model": "compaction",
                "messages": [type("M", (), {"role": "system", "content": prompt})],
                "tools": None,  # type: ignore
                "temperature": 0.0,
                "max_tokens": 1024,
                "stream": False,
            })()
        )
        summary = resp.content or "Conversation summary."
        compacted = compact_session(sess, summary)
        sm.current_session = compacted
        return f"  {c('green', '✅')} Compaction done. {c('dim', f'{len(sess.messages)} → {len(compacted.messages)} messages')}"
    except Exception as e:
        return f"  {c('red', '❌')} Compaction failed: {e}"


# ──── 交互式 REPL ────


async def run_repl(
    store, sm: SessionManager, agent, config, model_ref: str,
    llm_client, tools: ToolRegistry, assembler, no_api: bool,
):
    """交互式 REPL"""
    try:
        import readline
        histfile = os.path.expanduser("~/.mini_openclaw_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)
    except ImportError:
        histfile = None

    def _save_history():
        if histfile:
            try:
                readline.write_history_file(histfile)
            except Exception:
                pass

    mode_str = c("yellow", "MOCK") if no_api else c("green", model_ref)
    tool_names = [t.name for t in tools.list_tools()]

    print(f"  {c('dim', 'Model:')}    {mode_str}")
    print(f"  {c('dim', 'Session:')}  {sm.current_key} ({c('dim', 'type /help for commands')})")

    prompt_tpl = "\n" + c("cyan", "◆") + c("dim", " You: ") + c("end", "")
    ai_prefix = c("green", " ◇ ") + c("bold", "mini") + c("dim", "-openclaw: ") + c("end", "")

    while True:
        try:
            user_input = input(prompt_tpl).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # ── 退出 ──
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            _save_history()
            print(f"  {c('dim', 'Bye!')}")
            break

        # ── 斜杠命令 ──
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("help", "h"):
                print(cmd_help())

            elif cmd in ("info", "i"):
                print(cmd_info(sm, model_ref, config))

            elif cmd in ("sessions", "ss"):
                print(cmd_list_sessions(store, sm))

            elif cmd == "session":
                if arg:
                    await sm.switch_session(arg)
                    print(f"  Switched to session: {c('bold', arg)}")
                    print(cmd_info(sm, model_ref, config))
                else:
                    print(f"  {c('yellow', 'Usage:')} /session <key>")

            elif cmd in ("new", "reset"):
                await sm.new_session()
                print(f"  {c('green', '✅')} New session started")

            elif cmd == "compact":
                result = await cmd_compact(sm, sm, llm_client)
                print(result)

            elif cmd == "model":
                if arg:
                    model_ref = arg
                    print(f"  {c('green', '✅')} Model will be used on next message: {arg}")
                else:
                    print(f"  Current model: {model_ref}")

            elif cmd == "tools":
                names = [t.name for t in tools.list_tools()]
                descs = [f"  {c('bold', t.name)}: {c('dim', t.description[:60])}" for t in tools.list_tools()]
                print(f"\n{c('bold', '🔧 Tools')} ({len(names)}):")
                print("\n".join(descs))

            elif cmd == "save":
                if arg:
                    await _export_session(sm.session(), arg)
                else:
                    await sm.save_current()
                    print(f"  {c('green', '✅')} Session saved to disk")

            elif cmd == "memory":
                await _handle_memory(sm, arg)

            else:
                print(f"  {c('yellow', 'Unknown command:')} /{cmd}  ({c('dim', 'try /help')})")

            continue

        # ── 执行 Agent ──
        print(ai_prefix, end="", flush=True)
        sess = sm.session()
        try:
            async for event in agent.run(sess, user_input):
                if event.kind == "text":
                    print(c("green", event.data), end="", flush=True)
                elif event.kind == "tool_start":
                    name = event.data.get("name", "?")
                    tool = tools.get(name)
                    desc = (tool.description[:40] + "..") if tool else ""
                    print(f"\n  {c('blue', '🛠')} {c('bold', name)}: {c('dim', desc)}")
                    print(ai_prefix, end="", flush=True)
                elif event.kind == "tool_end":
                    r = event.data
                    status = c("red", "✗") if r.get("is_error") else c("green", "✓")
                    txt = r.get("content", "")[:100]
                    print(f"\n  {status} {c('dim', txt)}")
                    print(ai_prefix, end="", flush=True)
                elif event.kind == "error":
                    print(f"\n  {c('red', '✗')} {event.data}")
                elif event.kind == "done":
                    print()

            await sm.save_current()
            _save_history()

        except Exception as e:
            print(f"\n  {c('red', '✗')} Agent error: {e}")
            import traceback
            traceback.print_exc()

    _save_history()


async def _export_session(session, path: str):
    """导出会话到 JSON 文件"""
    data = {
        "key": session.key,
        "id": session.id,
        "messages": [m.to_dict() for m in session.messages],
        "metadata": session.metadata,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  {c('green', '✅')} Exported to {path}")


async def _handle_memory(sm, query: str):
    session = sm.session()
    if not query:
        print(f"  {c('yellow', 'Usage:')} /memory <query text>")
        return
    from mini_openclaw.memory.memory import keyword_search
    meta = ""
    for msg in session.messages:
        if msg.content:
            meta += msg.content + "\n"
    results = keyword_search(meta, query)
    if results:
        print(f"\n  {c('bold', 'Memory results')} ({len(results)}):")
        for r in results[:5]:
            print(f"    [{r['source'] if 'source' in r else c('dim', 'session')}:{r['line']}] {c('dim', r['content'][:80])}")
    else:
        print(f"  {c('yellow', 'No results')}")


def _register_argparse():
    parser = argparse.ArgumentParser(
        description="mini-openclaw — Simplified AI Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.json",
                        help="Config file path (default: config.json)")
    parser.add_argument("--model", default="deepseek/deepseek-chat",
                        help="Model reference, e.g. deepseek/deepseek-chat")
    parser.add_argument("--no-api", action="store_true",
                        help="Mock mode, no real API calls")
    parser.add_argument("--prompt", "-p", type=str,
                        help="One-shot prompt (non-interactive)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming for one-shot mode")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode, less verbose output")
    parser.add_argument("--session", type=str, default="main:cli",
                        help="Session key to use (default: main:cli)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    return parser


async def one_shot(
    prompt: str, agent, session, llm_client, tools: ToolRegistry,
    store, session_key: str, sm: SessionManager, stream: bool = True,
) -> int:
    """单次执行，返回退出码"""
    sess = session
    method = agent.run_stream if stream else agent.run

    try:
        content_chunks: list[str] = []
        async for event in method(sess, prompt):
            if event.kind == "text":
                content_chunks.append(event.data)
                if stream:
                    print(event.data, end="", flush=True)
            elif event.kind == "tool_start":
                name = event.data.get("name", "?")
                print(f"\n[calling {name}]", end="", flush=True)
            elif event.kind == "tool_end":
                r = event.data
                status = "✗" if r.get("is_error") else "✓"
                print(f"\n[{status} tool done]", end="", flush=True)
            elif event.kind == "error":
                print(f"\nError: {event.data}", file=sys.stderr)
                return 1
            elif event.kind == "done":
                if not stream:
                    print("".join(content_chunks))

        if stream:
            print()

        # 保存会话
        if len(sess.messages) >= 2:
            await store.save_message(session_key, sess.messages[-2])
            await store.save_message(session_key, sess.messages[-1])

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def main():
    parser = _register_argparse()
    args = parser.parse_args()

    # 配置
    if os.path.exists(args.config):
        config = ConfigLoader.load(args.config)
    else:
        from mini_openclaw.config.config import AppConfig
        config = AppConfig()

    # 模型
    models_path = config.models_config_path
    if not os.path.exists(models_path):
        print(f"Models file not found: {models_path}", file=sys.stderr)
        return 1

    registry = ModelRegistry.from_json_file(models_path)

    if args.list_models:
        print("Available models:")
        for p in registry.list_providers():
            for m in registry.list_models(p):
                print(f"  {p}/{m.id}")
        return 0

    try:
        provider, model_info = registry.resolve_model(args.model)
    except Exception as e:
        print(f"Failed to resolve model '{args.model}': {e}", file=sys.stderr)
        print("\nAvailable models:")
        for p in registry.list_providers():
            for m in registry.list_models(p):
                print(f"  {p}/{m.id}")
        return 1

    # LLM Client
    llm_client, is_mock = _resolve_provider_client(provider, use_mock=args.no_api)

    # Tools
    tools = register_default_tools(config.workspace_dir)

    # Session
    os.makedirs(config.session_dir, exist_ok=True)
    store = SessionStore(config.session_dir)
    router = SessionRouter(dm_scope=config.dm_scope)
    sm = SessionManager(store, router)

    session_key = router.route("cli", args.session, "direct")
    is_new = await sm.create_or_load(session_key)

    if not args.quiet:
        if not args.prompt:
            print_banner()
        elif is_new:
            print(f"[session: {session_key}]", file=sys.stderr)
        else:
            print(f"[session: {session_key} ({len(sm.session().messages)} msgs)]", file=sys.stderr)

    # Prompt
    ws_dir = config.workspace_dir
    os.makedirs(ws_dir, exist_ok=True)
    bootstrap = load_bootstrap_files(ws_dir)
    prompt_builder = SystemPromptBuilder(config)
    system_prompt = prompt_builder.build(
        extra_prompt="You are a helpful coding assistant. Use tools when appropriate.",
        bootstrap_files=bootstrap,
    )

    # Agent
    agent_config = AgentConfig(
        system_prompt=system_prompt,
        max_tool_rounds=10,
        temperature=0.0,
    )
    agent = AgentLoop(llm_client, tools, model_info.id, agent_config)
    assembler = ContextAssembler()

    # 模式选择
    if args.prompt:
        return await one_shot(
            prompt=args.prompt,
            agent=agent,
            session=sm.session(),
            llm_client=llm_client,
            tools=tools,
            store=store,
            session_key=session_key,
            sm=sm,
            stream=not args.no_stream,
        )
    else:
        await run_repl(
            store=store,
            sm=sm,
            agent=agent,
            config=config,
            model_ref=args.model,
            llm_client=llm_client,
            tools=tools,
            assembler=assembler,
            no_api=is_mock,
        )
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
