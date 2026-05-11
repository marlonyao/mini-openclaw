"""工具系统和 Agent 循环测试"""

import pytest
import json
import tempfile
import os

from mini_openclaw.tools.tool import (
    ToolRegistry,
    EchoTool,
    ReadTool,
    WriteTool,
    ToolCall as AgentToolCall,
)
from mini_openclaw.agent.loop import AgentLoop, AgentConfig, AgentEvent
from mini_openclaw.session.session import Session
from mini_openclaw.session.message import Message


# ──── 工具注册测试 ────

class TestToolRegistry:
    def test_register_and_list(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_duplicate_registration(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(EchoTool())

    def test_get_tool(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        tool = registry.get("echo")
        assert tool is not None
        assert tool.name == "echo"

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_to_tool_defs(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        defs = registry.to_tool_defs()
        assert len(defs) == 1
        assert defs[0].name == "echo"

    @pytest.mark.asyncio
    async def test_execute_batch(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        calls = [
            AgentToolCall(id="c1", name="echo", arguments='{"message": "hello"}'),
            AgentToolCall(id="c2", name="echo", arguments='{"message": "world"}'),
        ]
        results = await registry.execute_batch(calls)
        assert len(results) == 2
        assert results[0].content == "Echo: hello"
        assert results[1].content == "Echo: world"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        call = AgentToolCall(id="c1", name="unknown", arguments="{}")
        result = await registry.execute_tool_call(call)
        assert result.is_error
        assert "not found" in result.content


@pytest.mark.asyncio
class TestTools:
    async def test_echo_tool(self):
        tool = EchoTool()
        result = await tool.execute({"message": "test"})
        assert result.content == "Echo: test"
        assert not result.is_error

    async def test_read_tool(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            tmp_path = f.name

        try:
            tool = ReadTool()
            result = await tool.execute({"path": tmp_path})
            assert result.content == "hello world"
        finally:
            os.unlink(tmp_path)

    async def test_read_not_found(self):
        tool = ReadTool()
        result = await tool.execute({"path": "/tmp/nonexistent_xyz.txt"})
        assert result.is_error
        assert "not found" in result.content

    async def test_write_tool(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            tool = WriteTool()
            result = await tool.execute({"path": path, "content": "hello"})
            assert not result.is_error
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == "hello"

    async def test_tool_call_parse_arguments(self):
        call = AgentToolCall(id="c1", name="test", arguments='{"key": "value"}')
        args = call.parse_arguments()
        assert args["key"] == "value"

    async def test_tool_call_parameter_defaults(self):
        tool = EchoTool()
        paras = tool.parameters
        assert "message" in paras["properties"]
        assert "required" in paras


# ──── AgentLoop 测试 ────

class MockLlmClient:
    """模拟 LLM 客户端用于测试"""
    def __init__(self):
        self.calls = []

    async def chat_completion(self, request):
        self.calls.append(request)
        # 默认返回一个简单的文本响应
        return self._make_text_response("Mock response")

    async def chat_completion_stream(self, request):
        self.calls.append(request)
        from mini_openclaw.models.provider import StreamChunk
        yield StreamChunk(content_delta="Mock ")
        yield StreamChunk(content_delta="response")
        yield StreamChunk(finish_reason="stop")

    def _make_text_response(self, text, finish_reason="stop"):
        from mini_openclaw.models.provider import CompletionResponse
        return CompletionResponse(content=text, finish_reason=finish_reason)

    def _make_tool_response(self, tool_calls, text=None, finish_reason="tool_use"):
        from mini_openclaw.models.provider import CompletionResponse, ToolCall
        tc = [ToolCall(id=t["id"], name=t["name"], arguments=t.get("arguments", "{}")) for t in tool_calls]
        return CompletionResponse(content=text, tool_calls=tc, finish_reason=finish_reason)


@pytest.mark.asyncio
class TestAgentLoop:
    async def test_simple_text_response(self):
        mock = MockLlmClient()
        registry = ToolRegistry()
        session = Session(key="test:main")
        loop = AgentLoop(mock, registry, "test-model")

        events = []
        async for event in loop.run(session, "Hello"):
            events.append(event)

        # 应该有 text 事件和 done 事件
        text_events = [e for e in events if e.kind == "text"]
        done_events = [e for e in events if e.kind == "done"]
        assert len(text_events) > 0
        assert len(done_events) > 0

        # 用户消息和助手消息应该已保存
        msgs = session.get_messages()
        assert len(msgs) >= 2
        assert msgs[0].role == "user"
        assert msgs[-1].role == "assistant"

    async def test_tool_call_flow(self):
        """测试工具调用流程"""
        registry = ToolRegistry()
        registry.register(EchoTool())
        session = Session(key="test:tool")
    
        # 模拟：第一次调用返回工具调用，第二次返回文本
        class ToolMockLlm(MockLlmClient):
            def __init__(self):
                super().__init__()
                self.call_count = 0
    
            async def chat_completion(self, request):
                self.calls.append(request)
                self.call_count += 1
                if self.call_count == 1:
                    return self._make_tool_response(
                        [{"id": "call_1", "name": "echo", "arguments": '{"message": "hello"}'}],
                        text="",
                    )
                return self._make_text_response("Done after tool call")
    
        loop = AgentLoop(ToolMockLlm(), registry, "test-model")
        events = []
        async for event in loop.run(session, "Use tool"):
            events.append(event)

        # 应该有 tool_start 和 tool_end 事件
        tool_starts = [e for e in events if e.kind == "tool_start"]
        tool_ends = [e for e in events if e.kind == "tool_end"]
        assert len(tool_starts) >= 1
        assert len(tool_ends) >= 1
        assert tool_starts[0].data["name"] == "echo"

    async def test_system_prompt(self):
        mock = MockLlmClient()
        registry = ToolRegistry()
        loop = AgentLoop(
            mock, registry, "test-model",
            config=AgentConfig(system_prompt="You are a helpful assistant."),
        )
        session = Session(key="test:sysprompt")

        async for event in loop.run(session, "Hello"):
            pass

        # 验证 LLM 收到 system 消息
        assert len(mock.calls) > 0
        first_call = mock.calls[0]
        messages = first_call.messages
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful assistant."

    async def test_max_tool_rounds(self):
        """超过最大工具轮数应该报错"""
        mock = MockLlmClient()
        registry = ToolRegistry()
        registry.register(EchoTool())

        # 模拟:无限返回工具调用
        class InfiniteToolMock(MockLlmClient):
            async def chat_completion(self, request):
                return self._make_tool_response(
                    [{"id": "call_x", "name": "echo", "arguments": '{"message": "loop"}'}],
                    text="",
                )

        session = Session(key="test:maxrounds")
        loop = AgentLoop(
            InfiniteToolMock(), registry, "test-model",
            config=AgentConfig(max_tool_rounds=3),
        )

        events = []
        async for event in loop.run(session, "Loop"):
            events.append(event)

        error_events = [e for e in events if e.kind == "error"]
        assert len(error_events) > 0
        assert "Max tool rounds" in error_events[-1].data

    async def test_session_history_preserved(self):
        """多次运行后会话历史应保留"""
        mock = MockLlmClient()
        registry = ToolRegistry()
        session = Session(key="test:history")
        loop = AgentLoop(mock, registry, "test-model")

        async for _ in loop.run(session, "First message"):
            pass
        async for _ in loop.run(session, "Second message"):
            pass

        msgs = session.get_messages()
        assert len(msgs) == 4  # user1, assistant1, user2, assistant2
        assert msgs[0].content == "First message"
        assert msgs[2].content == "Second message"

    async def test_empty_response(self):
        """空响应应该也能正确处理"""
        mock = MockLlmClient()
        class EmptyMock(MockLlmClient):
            async def chat_completion(self, request):
                return self._make_text_response("")

        session = Session(key="test:empty")
        loop = AgentLoop(EmptyMock(), ToolRegistry(), "test-model")

        events = []
        async for event in loop.run(session, "Say nothing"):
            events.append(event)

        done_events = [e for e in events if e.kind == "done"]
        assert len(done_events) == 1

    async def test_llm_error_handling(self):
        """LLM 报错应正确处理"""
        class ErrorMock(MockLlmClient):
            async def chat_completion(self, request):
                raise Exception("API is down")

        session = Session(key="test:error")
        loop = AgentLoop(ErrorMock(), ToolRegistry(), "test-model")

        events = []
        async for event in loop.run(session, "Hello"):
            events.append(event)

        error_events = [e for e in events if e.kind == "error"]
        assert len(error_events) > 0
        assert "API is down" in error_events[0].data
