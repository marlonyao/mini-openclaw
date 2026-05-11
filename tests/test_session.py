"""会话管理测试"""

import json
import os
import tempfile

import pytest

from mini_openclaw.session.message import Message, ToolCall
from mini_openclaw.session.session import Session
from mini_openclaw.session.store import SessionStore
from mini_openclaw.session.router import SessionRouter, SessionRouterError


# ──── 消息模型测试 ────

class TestMessage:
    def test_create_text_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.id is not None
        assert msg.tool_calls is None

    def test_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments='{"city":"Beijing"}')],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"

    def test_message_with_tool_result(self):
        msg = Message(role="tool", content="Sunny", tool_call_id="call_1")
        assert msg.tool_call_id == "call_1"

    def test_serialization_roundtrip(self):
        original = Message(
            role="assistant",
            content="Hello world",
            tool_calls=[ToolCall(name="search", arguments='{"q":"test"}')],
        )
        data = original.to_dict()
        restored = Message.from_dict(data)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.tool_calls[0].name == original.tool_calls[0].name

    def test_token_estimate(self):
        msg = Message(role="user", content="hello world")
        assert msg.token_estimate() > 0
        # "hello world" = 11 chars, /4 ≈ 3 tokens
        assert msg.token_estimate() >= 2

    def test_long_message_token_estimate(self):
        content = "A" * 400
        msg = Message(role="user", content=content)
        # 400 chars / 4 = 100 tokens
        assert msg.token_estimate() >= 100

    def test_tool_call_token_estimate(self):
        msg = Message(
            role="assistant",
            tool_calls=[ToolCall(name="long_function_name", arguments='{"very": "long argument string that takes space"}')],
        )
        assert msg.token_estimate() > 5


# ──── 会话测试 ────

class TestSession:
    def test_create_session(self):
        session = Session(key="main")
        assert session.key == "main"
        assert len(session.messages) == 0
        assert session.id is not None

    def test_add_message(self):
        session = Session(key="main")
        msg = Message(role="user", content="Hi")
        session.add_message(msg)
        assert len(session.messages) == 1
        assert session.messages[0].content == "Hi"

    def test_get_messages_limit(self):
        session = Session(key="main")
        for i in range(10):
            session.add_message(Message(role="user", content=f"msg {i}"))
        msgs = session.get_messages(limit=3)
        assert len(msgs) == 3
        assert msgs[-1].content == "msg 9"

    def test_last_n_messages(self):
        session = Session(key="main")
        for i in range(5):
            session.add_message(Message(role="user", content=f"msg {i}"))
        last = session.last_n_messages(2)
        assert len(last) == 2
        assert last[0].content == "msg 3"

    def test_token_estimate(self):
        session = Session(key="main")
        session.add_message(Message(role="user", content="Hello" * 100))
        session.add_message(Message(role="assistant", content="World" * 100))
        assert session.token_estimate() > 50

    def test_jsonl_serialization(self):
        session = Session(key="main")
        session.add_message(Message(role="user", content="Hello"))
        lines = session.to_jsonl_lines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["session_id"] == session.id


# ──── 会话存储测试 ────

@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp()
    yield SessionStore(tmpdir)
    import shutil
    shutil.rmtree(tmpdir)


@pytest.mark.asyncio
class TestSessionStore:
    async def test_create_and_load(self, store):
        session = store.create_session("test:user1")
        assert session.key == "test:user1"
        assert session.id is not None

        msg = Message(role="user", content="Hello")
        await store.save_message("test:user1", msg)

        # 新加一个消息
        msg2 = Message(role="assistant", content="Hi!")
        await store.save_message("test:user1", msg2)

        loaded = await store.load_session("test:user1")
        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Hello"
        assert loaded.messages[1].content == "Hi!"

    async def test_session_not_found(self, store):
        loaded = await store.load_session("nonexistent_key")
        assert loaded is None

    async def test_session_exists(self, store):
        store.create_session("exists:key")
        assert await store.session_exists("exists:key") is True
        assert await store.session_exists("no:exist") is False

    async def test_delete_session(self, store):
        store.create_session("del:key")
        await store.save_message("del:key", Message(role="user", content="test"))
        assert await store.session_exists("del:key") is True
        await store.delete_session("del:key")
        assert await store.session_exists("del:key") is False

    async def test_list_sessions(self, store):
        store.create_session("session:a")
        store.create_session("session:b")
        store.create_session("session:c")
        metas = store.list_sessions()
        assert len(metas) == 3
        keys = [m.key for m in metas]
        assert "session:a" in keys
        assert "session:b" in keys

    async def test_messages_persist(self, store):
        """消息持久化，加载后保留顺序"""
        session = store.create_session("test:persist")
        for i in range(5):
            await store.save_message("test:persist", Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg_{i}",
            ))

        loaded = await store.load_session("test:persist")
        assert loaded is not None
        contents = [m.content for m in loaded.messages]
        assert contents == [f"msg_{i}" for i in range(5)]

    async def test_clear_expired(self, store):
        """清理过期会话"""
        store.create_session("fresh:key")
        store.create_session("old:key")

        # 手动把 old:key 的 updated_at 改到过去
        from datetime import timedelta, timezone
        old_time = __import__("datetime").datetime.now(timezone.utc) - timedelta(hours=48)
        meta = store._index.get("old:key")
        if meta:
            meta.updated_at = old_time
        store._save_index()

        cleared = await store.clear_expired(max_idle_hours=24)
        assert cleared == 1
        assert "fresh:key" in store._index
        assert "old:key" not in store._index

    async def test_multiple_messages_with_tool_calls(self, store):
        """保存和加载带工具调用的消息"""
        store.create_session("tool:test")

        # 用户消息
        await store.save_message("tool:test", Message(role="user", content="Search something"))

        # 助手工具调用
        await store.save_message("tool:test", Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="tc_1", name="search", arguments='{"q":"hello"}')],
        ))

        # 工具结果
        await store.save_message("tool:test", Message(
            role="tool",
            content="Search results...",
            tool_call_id="tc_1",
        ))

        loaded = await store.load_session("tool:test")
        assert loaded is not None
        assert len(loaded.messages) == 3
        assert loaded.messages[1].tool_calls is not None
        assert loaded.messages[2].tool_call_id == "tc_1"


# ──── 会话路由测试 ────

class TestSessionRouter:
    def test_dm_main_mode(self):
        router = SessionRouter(dm_scope="main")
        key = router.route("feishu", "user_123", "direct")
        assert key == "main", "DM 共享模式应返回 'main'"

    def test_dm_per_peer(self):
        router = SessionRouter(dm_scope="per-peer")
        key = router.route("feishu", "user_123", "direct")
        assert key == "main:user_123"

    def test_dm_per_channel_peer(self):
        router = SessionRouter(dm_scope="per-channel-peer")
        key = router.route("telegram", "user_456", "direct")
        assert key == "main:telegram:user_456"

    def test_group_routing(self):
        router = SessionRouter()
        key = router.route("discord", "guild_789", "group")
        assert key == "group:discord:guild_789"

    def test_group_isolation(self):
        """不同群组应该有不同 session key"""
        router = SessionRouter()
        key1 = router.route("telegram", "group_a", "group")
        key2 = router.route("telegram", "group_b", "group")
        assert key1 != key2

    def test_invalid_dm_scope(self):
        with pytest.raises(SessionRouterError):
            SessionRouter(dm_scope="invalid_scope")

    def test_set_mode(self):
        router = SessionRouter(dm_scope="main")
        router.set_mode("per-channel-peer")
        key = router.route("feishu", "user_1", "direct")
        assert key == "main:feishu:user_1"

    def test_different_channels_same_peer(self):
        """per-channel-peer 模式下，不同通道同一用户应隔离"""
        router = SessionRouter(dm_scope="per-channel-peer")
        key1 = router.route("feishu", "user_1", "direct")
        key2 = router.route("telegram", "user_1", "direct")
        assert key1 != key2

    def test_default_dm_scope(self):
        router = SessionRouter()
        assert router.dm_scope == "main"

    def test_dm_main_same_for_all(self):
        """main 模式下，所有 DM 都走同一个 session"""
        router = SessionRouter(dm_scope="main")
        k1 = router.route("feishu", "user_a", "direct")
        k2 = router.route("telegram", "user_b", "direct")
        assert k1 == k2 == "main"
