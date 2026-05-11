"""会话路由 - 参考 OpenClaw 的 Session Router

决定消息应该路由到哪个 session key。
"""

from __future__ import annotations


class SessionRouterError(Exception):
    pass


class SessionRouter:
    """消息路由决策器"""

    DM_SCOPE_MAIN = "main"              # 所有 DM 共享一个会话
    DM_SCOPE_PER_PEER = "per-peer"      # 按 sender 隔离
    DM_SCOPE_PER_CHANNEL_PEER = "per-channel-peer"  # 按 channel + sender 隔离
    DM_ISOLATION_MODES = [
        DM_SCOPE_MAIN,
        DM_SCOPE_PER_PEER,
        DM_SCOPE_PER_CHANNEL_PEER,
    ]

    def __init__(self, dm_scope: str = DM_SCOPE_MAIN):
        if dm_scope not in self.DM_ISOLATION_MODES:
            raise SessionRouterError(
                f"Invalid dm_scope: {dm_scope}. "
                f"Must be one of {self.DM_ISOLATION_MODES}"
            )
        self._dm_scope = dm_scope

    @property
    def dm_scope(self) -> str:
        return self._dm_scope

    def set_mode(self, mode: str) -> None:
        if mode not in self.DM_ISOLATION_MODES:
            raise SessionRouterError(f"Invalid mode: {mode}")
        self._dm_scope = mode

    def route(self, channel: str, peer_id: str, chat_type: str = "direct") -> str:
        """
        生成 session key。

        Args:
            channel: 消息通道 (feishu, telegram, discord 等)
            peer_id: 发送者 ID 或群组 ID
            chat_type: "direct" (私聊) 或 "group" (群组)

        Returns:
            session key 字符串
        """
        if chat_type == "group":
            # 群组按 channel + group_id 隔离
            return f"group:{channel}:{peer_id}"

        # 私聊
        if self._dm_scope == self.DM_SCOPE_MAIN:
            return "main"
        elif self._dm_scope == self.DM_SCOPE_PER_PEER:
            return f"main:{peer_id}"
        else:  # per-channel-peer
            return f"main:{channel}:{peer_id}"
