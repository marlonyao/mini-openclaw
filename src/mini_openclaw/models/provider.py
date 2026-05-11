"""LLM Provider 客户端 - 参考 OpenClaw 的 pi-ai 模块

支持 OpenAI-compatible 和 Anthropic-compatible 两种 API 格式
支持流式和非流式调用，工具调用
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel, Field


# ──── 数据模型 ────


class ToolCall(BaseModel):
    """工具调用"""
    id: str = ""
    name: str = ""
    arguments: str = "{}"  # JSON 字符串


class ToolDef(BaseModel):
    """工具定义"""
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str  # system, user, assistant, tool
    content: str | list | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class CompletionRequest(BaseModel):
    """补全请求"""
    model: str
    messages: list[ChatMessage]
    tools: list[ToolDef] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False


class CompletionResponse(BaseModel):
    """补全响应（非流式）"""
    id: str = ""
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: dict[str, object] | None = None


class StreamChunk(BaseModel):
    """流式响应块"""
    content_delta: str = ""
    tool_call_delta: ToolCall | None = None
    finish_reason: str | None = None


# ──── 抽象客户端 ────


class LlmClientError(Exception):
    """LLM 客户端异常"""
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


def _serialize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """序列化消息列表，正确处理 tool_calls 的 OpenAI 格式"""
    result: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.role == "assistant" and m.tool_calls:
            # OpenAI 格式：tool_calls 需要 function 包装
            d["content"] = m.content
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in m.tool_calls
            ]
        elif m.role == "tool":
            d["content"] = m.content
            d["tool_call_id"] = m.tool_call_id
        else:
            d["content"] = m.content
        result.append(d)
    return result


class AsyncLlmClient:
    """抽象的 LLM API 客户端基类"""

    async def chat_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        """非流式调用"""
        raise NotImplementedError

    async def chat_completion_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """流式调用"""
        raise NotImplementedError
        yield  # pragma: no cover


# ──── OpenAI-Compatible ────


class OpenAiClient(AsyncLlmClient):
    """OpenAI-compatible API 客户端"""

    def __init__(self, base_url: str, api_key: str, http_client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(self, request: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": _serialize_messages(request.messages),
            "stream": request.stream,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        return body

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        choice = data["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content")

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = []
            for tc in msg["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ))

        return CompletionResponse(
            id=data.get("id", ""),
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
        )

    def _parse_stream_line(self, line: str) -> StreamChunk | None:
        """解析 SSE 数据行"""
        if not line.startswith("data: "):
            return None
        payload = line[6:].strip()
        if payload == "[DONE]":
            return StreamChunk(finish_reason="stop")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})

        content_delta = delta.get("content", "")
        finish_reason = choices[0].get("finish_reason")

        tool_call_delta = None
        if "tool_calls" in delta:
            tc = delta["tool_calls"][0]
            func = tc.get("function", {})
            tool_call_delta = ToolCall(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=func.get("arguments", ""),
            )

        return StreamChunk(
            content_delta=content_delta or "",
            tool_call_delta=tool_call_delta,
            finish_reason=finish_reason,
        )

    async def chat_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        body = self._build_body(request)
        body["stream"] = False
        resp = await self._http.post(
            f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            json=body,
        )
        if resp.status_code != 200:
            raise LlmClientError(
                f"OpenAI API error: {resp.status_code}",
                status_code=resp.status_code,
                response_body=resp.text,
            )
        return self._parse_response(resp.json())

    async def chat_completion_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_body(request)
        body["stream"] = True
        async with self._http.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            json=body,
        ) as resp:
            if resp.status_code != 200:
                body_text = await resp.aread()
                raise LlmClientError(
                    f"OpenAI API error: {resp.status_code}",
                    status_code=resp.status_code,
                    response_body=body_text.decode(),
                )
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = self._parse_stream_line(line)
                if chunk is not None:
                    yield chunk


# ──── Anthropic-Compatible ────


class AnthropicClient(AsyncLlmClient):
    """Anthropic-compatible API 客户端"""

    def __init__(self, base_url: str, api_key: str, http_client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[list[dict], str | None]:
        """转换消息格式，抽离 system 消息"""
        system: str | None = None
        converted: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content or ""
                continue

            m: dict[str, Any] = {"role": msg.role}
            if msg.tool_calls:
                # assistant 的工具调用
                content_list: list[dict] = []
                if msg.content:
                    content_list.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    args = tc.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    content_list.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": args,
                    })
                m["content"] = content_list
            elif msg.role == "tool" and msg.tool_call_id:
                m["content"] = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }
                ]
            else:
                m["content"] = msg.content or ""

            converted.append(m)

        return converted, system

    def _build_body(self, request: CompletionRequest) -> dict[str, Any]:
        messages, system = self._convert_messages(request.messages)
        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }
        if system:
            body["system"] = system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.tools:
            body["tools"] = [self._convert_tool(t) for t in request.tools]
        return body

    def _convert_tool(self, tool: ToolDef) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        content = ""
        tool_calls: list[ToolCall] = []
        finish_reason = data.get("stop_reason")
        usage = data.get("usage")

        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=json.dumps(block["input"]) if isinstance(block.get("input"), dict) else str(block.get("input", "{}")),
                ))

        return CompletionResponse(
            id=data.get("id", ""),
            content=content or None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _parse_sse(self, resp: httpx.Response) -> AsyncIterator[StreamChunk]:
        """解析 Anthropic 流式的 SSE"""
        current_tool_call: dict[str, Any] | None = None

        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            if not line.startswith("data: "):
                continue

            payload = line[6:].strip()
            if payload == "[DONE]":
                yield StreamChunk(finish_reason="stop")
                continue

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool_call = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": "",
                    }

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield StreamChunk(content_delta=delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    if current_tool_call:
                        current_tool_call["arguments"] += partial
                        yield StreamChunk(tool_call_delta=ToolCall(
                            id=current_tool_call["id"],
                            name=current_tool_call["name"],
                            arguments=partial,
                        ))

            elif event_type == "content_block_stop":
                if current_tool_call:
                    current_tool_call = None

            elif event_type == "message_delta":
                finish = data.get("delta", {}).get("stop_reason")
                if finish:
                    yield StreamChunk(finish_reason=finish)

            elif event_type == "message_stop":
                yield StreamChunk(finish_reason="end_turn")

    async def chat_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        body = self._build_body(request)
        body["stream"] = False
        resp = await self._http.post(
            f"{self.base_url}/messages",
            headers=self._build_headers(),
            json=body,
        )
        if resp.status_code != 200:
            raise LlmClientError(
                f"Anthropic API error: {resp.status_code}",
                status_code=resp.status_code,
                response_body=resp.text,
            )
        return self._parse_response(resp.json())

    async def chat_completion_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_body(request)
        body["stream"] = True
        async with self._http.stream(
            "POST",
            f"{self.base_url}/messages",
            headers=self._build_headers(),
            json=body,
        ) as resp:
            if resp.status_code != 200:
                body_text = await resp.aread()
                raise LlmClientError(
                    f"Anthropic API error: {resp.status_code}",
                    status_code=resp.status_code,
                    response_body=body_text.decode(),
                )
            async for chunk in self._parse_sse(resp):
                yield chunk
