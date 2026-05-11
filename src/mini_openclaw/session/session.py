"""会话模型 - 参考 OpenClaw 的 SessionManager"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from mini_openclaw.session.message import Message


class Session:
    """会话 - 消息序列 + 元数据"""

    def __init__(
        self,
        key: str,
        session_id: str | None = None,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = session_id or uuid4().hex
        self.key = key
        self.messages: list[Message] = messages or []
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.metadata: dict[str, Any] = metadata or {}

    def add_message(self, msg: Message) -> None:
        """添加消息到会话"""
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata["message_count"] = len(self.messages)

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """获取消息列表"""
        if limit is not None:
            return self.messages[-limit:]
        return list(self.messages)

    def last_n_messages(self, n: int) -> list[Message]:
        """获取最后 N 条消息"""
        return self.messages[-n:]

    def token_estimate(self) -> int:
        """整个会话的 token 估算"""
        return sum(m.token_estimate() for m in self.messages)

    def to_jsonl_lines(self) -> list[str]:
        """序列化为 JSONL 行"""
        import json
        lines: list[str] = []
        for msg in self.messages:
            record = msg.to_dict()
            record["session_id"] = self.id
            lines.append(json.dumps(record, ensure_ascii=False))
        return lines

    def __repr__(self) -> str:
        return f"Session(key={self.key!r}, messages={len(self.messages)}, tokens≈{self.token_estimate()})"
