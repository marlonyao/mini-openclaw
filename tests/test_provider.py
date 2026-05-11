"""LLM Provider 客户端测试"""

import json

import httpx
import pytest
import respx

from mini_openclaw.models.provider import (
    ChatMessage,
    CompletionRequest,
    OpenAiClient,
    AnthropicClient,
    ToolDef,
    ToolCall,
    LlmClientError,
)


# ──── OpenAI 客户端测试 ────

@pytest.fixture
def openai_client():
    return OpenAiClient("https://api.openai.com/v1", "sk-test")


@pytest.mark.asyncio
async def test_openai_chat_completion(respx_mock, openai_client):
    respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )

    resp = await openai_client.chat_completion(CompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="Hi")],
    ))

    assert resp.content == "Hello!"
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5}


@pytest.mark.asyncio
async def test_openai_chat_completion_with_tools(respx_mock, openai_client):
    respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    resp = await openai_client.chat_completion(CompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="Weather?")],
        tools=[ToolDef(name="get_weather", description="Get weather", parameters={"type": "object"})],
    ))

    assert resp.content is None
    assert resp.tool_calls is not None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == '{"city": "Beijing"}'
    assert resp.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_openai_stream_chunks(respx_mock, openai_client):
    sse_lines = "\n\n".join([
        'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        'data: [DONE]',
    ])
    respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        text=sse_lines,
        headers={"Content-Type": "text/event-stream"},
    )

    chunks = []
    async for chunk in openai_client.chat_completion_stream(CompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="Hi")],
    )):
        chunks.append(chunk)

    contents = "".join(c.content_delta for c in chunks)
    assert contents == "Hello world"
    assert any(c.finish_reason == "stop" for c in chunks)


@pytest.mark.asyncio
async def test_openai_api_error(respx_mock, openai_client):
    respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        status_code=401, text='{"error": "unauthorized"}'
    )
    with pytest.raises(LlmClientError) as exc:
        await openai_client.chat_completion(CompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hi")],
        ))
    assert exc.value.status_code == 401


# ──── Anthropic 客户端测试 ────

@pytest.fixture
def anthropic_client():
    return AnthropicClient("https://api.anthropic.com/v1", "sk-ant-test")


@pytest.mark.asyncio
async def test_anthropic_chat_completion(respx_mock, anthropic_client):
    respx_mock.post("https://api.anthropic.com/v1/messages").respond(
        json={
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )

    resp = await anthropic_client.chat_completion(CompletionRequest(
        model="claude-sonnet-4-6",
        messages=[ChatMessage(role="user", content="Hi")],
    ))

    assert resp.content == "Hello from Claude!"
    assert resp.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_with_system_message(respx_mock, anthropic_client):
    """系统消息应该转换为 system 参数"""
    mock_route = respx_mock.post("https://api.anthropic.com/v1/messages").respond(
        json={
            "id": "msg_456",
            "content": [{"type": "text", "text": "I am helpful."}],
            "stop_reason": "end_turn",
        }
    )

    await anthropic_client.chat_completion(CompletionRequest(
        model="claude-3",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hi"),
        ],
    ))

    request_body = json.loads(mock_route.calls.last.request.content)
    assert request_body["system"] == "You are a helpful assistant."
    assert request_body["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_anthropic_with_tool_use(respx_mock, anthropic_client):
    respx_mock.post("https://api.anthropic.com/v1/messages").respond(
        json={
            "id": "msg_tool",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Beijing"},
                },
            ],
            "stop_reason": "tool_use",
        }
    )

    resp = await anthropic_client.chat_completion(CompletionRequest(
        model="claude-3",
        messages=[ChatMessage(role="user", content="Weather?")],
        tools=[ToolDef(name="get_weather", description="Get weather")],
    ))

    assert resp.content == "Let me check the weather."
    assert resp.tool_calls is not None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.finish_reason == "tool_use"


@pytest.mark.asyncio
async def test_anthropic_tool_result(respx_mock, anthropic_client):
    """工具结果消息应转换为 tool_result 格式"""
    mock_route = respx_mock.post("https://api.anthropic.com/v1/messages").respond(
        json={"id": "msg_tr", "content": [{"type": "text", "text": "Done"}], "stop_reason": "end_turn"}
    )

    await anthropic_client.chat_completion(CompletionRequest(
        model="claude-3",
        messages=[
            ChatMessage(role="user", content="Check weather"),
            ChatMessage(role="assistant", content="", tool_calls=[
                ToolCall(id="toolu_1", name="get_weather", arguments='{"city":"Beijing"}'),
            ]),
            ChatMessage(role="tool", content="Sunny", tool_call_id="toolu_1"),
        ],
    ))

    body = json.loads(mock_route.calls.last.request.content)
    # 最后一条消息应该是 tool_result
    last_msg = body["messages"][-1]["content"]
    assert last_msg[0]["type"] == "tool_result"
    assert last_msg[0]["tool_use_id"] == "toolu_1"


@pytest.mark.asyncio
async def test_anthropic_stream(respx_mock, anthropic_client):
    sse_lines = "\n\n".join([
        'event: content_block_start\ndata: {"type":"content_block_start","content_block":{"type":"text","text":""}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}',
        'event: content_block_stop\ndata: {"type":"content_block_stop"}',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
        'event: message_stop\ndata: {"type":"message_stop"}',
    ])
    respx_mock.post("https://api.anthropic.com/v1/messages").respond(
        text=sse_lines,
        headers={"Content-Type": "text/event-stream"},
    )

    chunks = []
    async for chunk in anthropic_client.chat_completion_stream(CompletionRequest(
        model="claude-3",
        messages=[ChatMessage(role="user", content="Hi")],
    )):
        chunks.append(chunk)

    content = "".join(c.content_delta for c in chunks)
    assert content == "Hello world"
    assert any(c.finish_reason in ("end_turn", "stop") for c in chunks)


@pytest.mark.asyncio
async def test_request_building_headers(respx_mock):
    """验证 OpenAI 请求头是否正确"""
    mock_route = respx_mock.post("https://custom.api.com/v1/chat/completions").respond(
        json={"id": "x", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    )
    client = OpenAiClient("https://custom.api.com/v1", "sk-custom-key")
    await client.chat_completion(CompletionRequest(
        model="my-model",
        messages=[ChatMessage(role="user", content="Hi")],
    ))

    req = mock_route.calls.last.request
    assert req.headers["Authorization"] == "Bearer sk-custom-key"
    body = json.loads(req.content)
    assert body["model"] == "my-model"
    assert body["stream"] is False


@pytest.mark.asyncio
async def test_openai_stream_tool_calls(respx_mock, openai_client):
    """流式工具调用"""
    sse_lines = "\n\n".join([
        'data: {"choices": [{"delta": {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","function":{"name":"search","arguments":""}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":"}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls":[{"index":0,"function":{"arguments":"\\"hello\\""}}]}, "finish_reason":"tool_calls"}]}',
        'data: [DONE]',
    ])
    respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        text=sse_lines, headers={"Content-Type": "text/event-stream"},
    )

    chunks = []
    async for chunk in openai_client.chat_completion_stream(CompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="search hello")],
        tools=[ToolDef(name="search", description="Search")],
    )):
        chunks.append(chunk)

    assert any(c.tool_call_delta is not None for c in chunks)
