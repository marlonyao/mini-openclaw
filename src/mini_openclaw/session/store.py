"""会话持久化存储 - 参考 OpenClaw 的 SessionManager (JSONL 文件)"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from mini_openclaw.session.message import Message
from mini_openclaw.session.session import Session


class SessionMeta:
    """会话元数据"""
    def __init__(
        self,
        key: str,
        session_id: str,
        created_at: datetime,
        updated_at: datetime,
        message_count: int = 0,
        token_estimate: int = 0,
    ):
        self.key = key
        self.session_id = session_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.message_count = message_count
        self.token_estimate = token_estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
            "token_estimate": self.token_estimate,
        }


class SessionStore:
    """JSONL 文件持久化存储"""

    def __init__(self, session_dir: str | Path):
        self._dir = Path(session_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "sessions_index.json"
        self._index: dict[str, SessionMeta] = {}
        self._load_index()

    def _session_path(self, key: str) -> Path:
        # 文件名为 session key 的 MD5 或安全文件名
        safe_name = key.replace("/", "_").replace(":", "_")
        return self._dir / f"{safe_name}.jsonl"

    def _load_index(self) -> None:
        """加载会话索引"""
        if self._index_path.exists():
            with open(self._index_path) as f:
                data = json.load(f)
                for key, meta_data in data.items():
                    self._index[key] = SessionMeta(
                        key=key,
                        session_id=meta_data.get("session_id", key),
                        created_at=datetime.fromisoformat(meta_data["created_at"]),
                        updated_at=datetime.fromisoformat(meta_data["updated_at"]),
                        message_count=meta_data.get("message_count", 0),
                        token_estimate=meta_data.get("token_estimate", 0),
                    )

    def _save_index(self) -> None:
        """保存会话索引"""
        data = {key: meta.to_dict() for key, meta in self._index.items()}
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_session(self, key: str) -> Session:
        """创建新会话"""
        session = Session(key=key)
        meta = SessionMeta(
            key=key,
            session_id=session.id,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        self._index[key] = meta
        # 创建空 JSONL 文件
        path = self._session_path(key)
        if not path.exists():
            path.touch()
        self._save_index()
        return session

    async def load_session(self, key: str) -> Session | None:
        """从 JSONL 文件加载会话"""
        path = self._session_path(key)
        if not path.exists():
            return None

        messages: list[Message] = []
        meta = self._index.get(key)
        session_id = meta.session_id if meta else uuid4().hex

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                msg = Message.from_dict(data)
                messages.append(msg)

        session = Session(key=key, session_id=session_id, messages=messages)
        if meta:
            session.created_at = meta.created_at
            session.updated_at = meta.updated_at
        return session

    async def save_message(self, key: str, msg: Message) -> None:
        """追加消息到 JSONL"""
        path = self._session_path(key)
        record = msg.to_dict()
        record["session_id"] = self._index.get(key, SessionMeta(key=key, session_id="", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))).session_id

        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 更新索引
        if key in self._index:
            self._index[key].updated_at = datetime.now(timezone.utc)
            self._index[key].message_count += 1
            self._index[key].token_estimate += msg.token_estimate()
        else:
            self._index[key] = SessionMeta(
                key=key,
                session_id=record["session_id"],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                message_count=1,
                token_estimate=msg.token_estimate(),
            )
        self._save_index()

    def list_sessions(self) -> list[SessionMeta]:
        """列出所有会话"""
        return list(self._index.values())

    async def session_exists(self, key: str) -> bool:
        """检查会话是否存在"""
        return key in self._index and self._session_path(key).exists()

    async def delete_session(self, key: str) -> None:
        """删除会话文件和索引"""
        path = self._session_path(key)
        if path.exists():
            os.unlink(path)
        self._index.pop(key, None)
        self._save_index()

    async def clear_expired(self, max_idle_hours: int = 24) -> int:
        """清理过期会话"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max_idle_hours)
        expired: list[str] = []
        for key, meta in self._index.items():
            if meta.updated_at < cutoff:
                expired.append(key)

        for key in expired:
            await self.delete_session(key)
        return len(expired)
