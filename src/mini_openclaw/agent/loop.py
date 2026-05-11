"""Agent 循环 - 参考 OpenClaw 的 agent-loop.md 和 pi-embedded-runner

核心循环：接收消息 → 组装上下文 → LLM 推理 → 工具执行 → 流式回复
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from mini_openclaw.models.provider import (
    AsyncLlmClient,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ToolCall as ProviderToolCall,
    ToolDef as ProviderToolDef,
)
from mini_openclaw.session.message import Message, ToolCall
from mini_openclaw.session.session import Session
from mini_openclaw.tools.tool import ToolRegistry, ToolCall as AgentToolCall


class AgentEvent:
    """Agent 循环中的事件"""
    def __init__(self, kind: str, data: Any = None):
        self.kind = kind  # "text", "tool_start", "tool_end", "error", "done"
        self.data = data


class AgentConfig:
    """Agent 配置"""
    def __init__(
        self,
        max_tool_rounds: int = 10,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        self.max_tool_rounds = max_tool_rounds
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens


class AgentError(Exception):
    pass


class AgentLoop:
    """Agent 主循环"""

    def __init__(
        self,
        llm_client: AsyncLlmClient,
        tool_registry: ToolRegistry,
        model_id: str,
        config: AgentConfig | None = None,
    ):
        self._llm = llm_client
        self._tools = tool_registry
        self._model_id = model_id
        self._config = config or AgentConfig()

    async def run(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """
        运行 agent 循环

        Args:
            session: 当前会话
            user_message: 用户消息文本

        Yields:
            AgentEvent: text/tool_start/tool_end/error/done 事件
        """
        # 1. 添加用户消息到会话
        msg = Message(role="user", content=user_message)
        session.add_message(msg)

        messages = self._build_messages(session)

        for round_idx in range(self._config.max_tool_rounds):
            try:
                # 2. 调用 LLM
                request = CompletionRequest(
                    model=self._model_id,
                    messages=messages,
                    tools=self._tools.to_tool_defs() if self._tools.list_tools() else None,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )

                response = await self._llm.chat_completion(request)

            except Exception as e:
                yield AgentEvent("error", f"LLM call failed: {e}")
                return

            # 3. 处理响应
            if response.content:
                yield AgentEvent("text", response.content)

            if response.tool_calls:
                # 添加助手消息到会话（包含工具调用）
                tool_calls = [
                    ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                    for tc in response.tool_calls
                ]
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=tool_calls,
                )
                session.add_message(assistant_msg)

                # 记录到 messages（用于后续请求）
                messages.append(self._to_provider_message(assistant_msg))

                # 4. 执行工具
                for tc in response.tool_calls:
                    yield AgentEvent("tool_start", {"id": tc.id, "name": tc.name})

                agent_calls = [
                    AgentToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                    for tc in response.tool_calls
                ]
                results = await self._tools.execute_batch(agent_calls)

                for result in results:
                    yield AgentEvent("tool_end", {
                        "tool_call_id": result.tool_call_id,
                        "content": result.content[:200],
                        "is_error": result.is_error,
                    })

                    # 添加工具结果到消息列表
                    tool_result_msg = ChatMessage(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    )
                    messages.append(tool_result_msg)

                    # 保存到会话
                    session.add_message(Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                    ))

                # 继续循环（让 LLM 处理工具结果）
                continue

            else:
                # 纯文本响应，没有工具调用 — 完成
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                )
                session.add_message(assistant_msg)
                yield AgentEvent("done", None)
                return

        # 超过最大工具轮数
        yield AgentEvent("error", "Max tool rounds reached")
        yield AgentEvent("done", None)

    async def run_stream(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """
        流式版本 - 实时输出 LLM 的流式响应

        简化实现：先流式输出，如果需要工具调用则回退到非流式执行工具
        """
        msg = Message(role="user", content=user_message)
        session.add_message(msg)

        messages = self._build_messages(session)
        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}

        for round_idx in range(self._config.max_tool_rounds):
            accumulated_content = ""
            accumulated_tool_calls = {}

            request = CompletionRequest(
                model=self._model_id,
                messages=messages,
                tools=self._tools.to_tool_defs() if self._tools.list_tools() else None,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            try:
                async for chunk in self._llm.chat_completion_stream(request):
                    if chunk.content_delta:
                        accumulated_content += chunk.content_delta
                        yield AgentEvent("text", chunk.content_delta)

                    if chunk.tool_call_delta:
                        # 累积工具调用
                        pass  # 简化：先缓存全文

                    if chunk.finish_reason:
                        pass

            except Exception as e:
                yield AgentEvent("error", f"LLM stream failed: {e}")
                yield AgentEvent("done", None)
                return

            # 由于流式 chunk 可能不完整，用非流式获取完整响应来判断工具调用
            # 如果有累积内容，说明 LLM 在生成文本
            # 但为了判断是否需要工具调用，我们还需要知道最终结果
            # 简化方案：如果没有文本内容，尝试做非流式调用来确认

            if not accumulated_content.strip():
                # 可能是工具调用，用非流式确认
                try:
                    response = await self._llm.chat_completion(CompletionRequest(
                        model=self._model_id,
                        messages=messages,
                        tools=self._tools.to_tool_defs() if self._tools.list_tools() else None,
                        temperature=self._config.temperature,
                        max_tokens=self._config.max_tokens,
                    ))
                except Exception as e:
                    yield AgentEvent("error", f"LLM call failed: {e}")
                    yield AgentEvent("done", None)
                    return

                if response.tool_calls:
                    tool_calls = [
                        ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                        for tc in response.tool_calls
                    ]
                    assistant_msg = Message(
                        role="assistant",
                        tool_calls=tool_calls,
                    )
                    session.add_message(assistant_msg)
                    messages.append(self._to_provider_message(assistant_msg))

                    for tc in response.tool_calls:
                        yield AgentEvent("tool_start", {"id": tc.id, "name": tc.name})

                    agent_calls = [
                        AgentToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                        for tc in response.tool_calls
                    ]
                    results = await self._tools.execute_batch(agent_calls)

                    for result in results:
                        yield AgentEvent("tool_end", {
                            "tool_call_id": result.tool_call_id,
                            "content": result.content[:200],
                            "is_error": result.is_error,
                        })
                        tool_result_msg = ChatMessage(
                            role="tool",
                            content=result.content,
                            tool_call_id=result.tool_call_id,
                        )
                        messages.append(tool_result_msg)
                        session.add_message(Message(
                            role="tool",
                            content=result.content,
                            tool_call_id=result.tool_call_id,
                        ))

                    continue

            # 纯文本或混合 — 完成
            if accumulated_content:
                session.add_message(Message(
                    role="assistant",
                    content=accumulated_content,
                ))
            yield AgentEvent("done", None)
            return

        yield AgentEvent("error", "Max tool rounds reached")
        yield AgentEvent("done", None)

    def _build_messages(self, session: Session) -> list[ChatMessage]:
        """从会话构建 LLM 消息列表"""
        messages: list[ChatMessage] = []

        # System prompt
        if self._config.system_prompt:
            messages.append(ChatMessage(
                role="system",
                content=self._config.system_prompt,
            ))

        # 历史消息
        for msg in session.get_messages():
            pm = self._to_provider_message(msg)
            if pm:
                messages.append(pm)

        return messages

    def _to_provider_message(self, msg: Message) -> ChatMessage | None:
        """将内部 Message 转换为 Provider ChatMessage"""
        kwargs: dict[str, Any] = {"role": msg.role}

        if msg.tool_calls:
            kwargs["content"] = msg.content
            kwargs["tool_calls"] = [
                ProviderToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in msg.tool_calls
            ]
        elif msg.tool_call_id:
            kwargs["content"] = msg.content
            kwargs["tool_call_id"] = msg.tool_call_id
        elif msg.role == "system":
            kwargs["content"] = msg.content
        else:
            kwargs["content"] = msg.content

        return ChatMessage(**kwargs)
