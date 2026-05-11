"""消息模型 - 参考 OpenClaw 的 chat-message-content"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class ToolCall:
    """工具调用"""
    def __init__(
        self,
        id: str = "",
        name: str = "",
        arguments: str = "{}",
    ):
        self.id = id or f"call_{uuid4().hex[:12]}"
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ToolCall:
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", "{}"),
        )

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, name={self.name!r})"


class Message:
    """会话中的一条消息"""

    def __init__(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_call_id: str | None = None,
        msg_id: str | None = None,
        created_at: datetime | None = None,
        **kwargs: Any,
    ):
        self.id = msg_id or uuid4().hex
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.created_at = created_at or datetime.now(timezone.utc)
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Message:
        created_at = None
        if "created_at" in data:
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                created_at = None

        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        msg = Message(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            msg_id=data.get("id"),
            created_at=created_at,
        )
        if "metadata" in data:
            msg.metadata = data["metadata"]
        return msg

    def token_estimate(self) -> int:
        """字符级 token 估算：~4 chars/token"""
        total = len(self.role) + 4  # role overhead
        if self.content:
            total += len(self.content)
        if self.tool_calls:
            for tc in self.tool_calls:
                total += len(tc.name) + len(tc.arguments) + 20
        if self.tool_call_id:
            total += len(self.tool_call_id)
        # 粗略估算
        return max(1, total // 4)

    def __repr__(self) -> str:
        content_preview = (self.content or "")[:50]
        return f"Message(role={self.role!r}, content={content_preview!r})"
