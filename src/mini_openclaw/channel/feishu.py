"""飞书消息通道 - 使用 WebSocket 长连接接收事件

不需要公网 IP / ngrok，应用需要有如下权限：
- im:message:readonly（读取消息）
- im:message（发送消息）
- im:resource（下载图片等资源）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import httpx


# ──── 数据模型 ────

@dataclass
class FeishuMessage:
    """飞书消息"""
    message_id: str
    chat_id: str
    sender_id: str
    text: str
    chat_type: str = "p2p"  # p2p = 私聊, group = 群聊
    raw: dict = field(default_factory=dict)


# ──── 飞书客户端 ────

class FeishuClientError(Exception):
    pass


class FeishuClient:
    """飞书 API 客户端"""

    BASE_URL = "https://open.feishu.cn/open-apis"

    def __init__(self, app_id: str, app_secret: str):
        self._app_id = app_id
        self._app_secret = app_secret
        self._tenant_token = ""
        self._token_expires_at = 0.0
        self._http = httpx.AsyncClient(timeout=30)

    async def get_tenant_token(self) -> str:
        """获取 tenant_access_token"""
        if time.time() < self._token_expires_at - 60:
            return self._tenant_token

        resp = await self._http.post(
            f"{self.BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        data = resp.json()
        if data.get("code") != 0:
            raise FeishuClientError(f"Auth failed: {data.get('msg', 'unknown')}")

        self._tenant_token = data["tenant_access_token"]
        self._token_expires_at = time.time() + data.get("expire", 7200)
        return self._tenant_token

    async def _get_headers(self) -> dict[str, str]:
        token = await self.get_tenant_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def get_websocket_url(self) -> str:
        """获取 WebSocket 事件订阅 URL"""
        headers = await self._get_headers()
        resp = await self._http.post(
            f"{self.BASE_URL}/ws/v1/event_subscription",
            headers=headers,
            json={},
        )
        data = resp.json()
        if data.get("code") != 0:
            raise FeishuClientError(
                f"WebSocket URL failed: {data.get('msg', 'unknown')}"
            )
        return data["data"]["url"]

    async def send_text(
        self, chat_id: str, text: str, message_id: str | None = None,
        receive_id_type: str = "open_id",
    ) -> dict:
        """发送文本消息

        Args:
            chat_id: 聊天 ID（open_id 或 chat_id）
            text: 消息文本
            message_id: 回复的消息 ID（可选）
            receive_id_type: ID 类型，默认 open_id
        """
        headers = await self._get_headers()
        content = json.dumps({"text": text}, ensure_ascii=False)

        body: dict[str, Any] = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": content,
        }

        if message_id:
            resp = await self._http.post(
                f"{self.BASE_URL}/im/v1/messages/{message_id}/reply",
                headers=headers,
                json=body,
            )
        else:
            resp = await self._http.post(
                f"{self.BASE_URL}/im/v1/messages?receive_id_type={receive_id_type}",
                headers=headers,
                json=body,
            )

        data = resp.json()
        if data.get("code") != 0:
            raise FeishuClientError(f"Send failed: {data.get('msg', 'unknown')}")
        return data

    async def get_message_text(self, message_id: str) -> str:
        """获取消息内容"""
        headers = await self._get_headers()
        resp = await self._http.get(
            f"{self.BASE_URL}/im/v1/messages/{message_id}",
            headers=headers,
        )
        data = resp.json()
        if data.get("code") != 0:
            return ""
        body = data.get("data", {}).get("items", [{}])[0].get("body", {}).get("content", "{}")
        try:
            content = json.loads(body)
            return content.get("text", "")
        except (json.JSONDecodeError, AttributeError):
            return ""


# ──── WebSocket 事件监听 ────

class FeishuEventHandler:
    """飞书事件处理器"""

    def __init__(self, client: FeishuClient):
        self._client = client
        self._running = False
        self._on_message: Callable[[FeishuMessage], Awaitable[None]] | None = None
        self._ws = None

    def set_message_handler(
        self, handler: Callable[[FeishuMessage], Awaitable[None]]
    ) -> None:
        self._on_message = handler

    async def run(self):
        """启动 WebSocket 事件监听"""
        self._running = True
        import websockets

        while self._running:
            try:
                ws_url = await self._client.get_websocket_url()
                print(f"  📡 Connecting to Feishu WebSocket...")

                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    print(f"  ✅ Feishu WebSocket connected!")

                    async for raw in ws:
                        try:
                            await self._handle_frame(raw)
                        except Exception as e:
                            print(f"  ⚠️  Handle error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  ⚠️  WS error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _handle_frame(self, raw: bytes | str):
        """处理 WebSocket 帧"""
        if isinstance(raw, bytes):
            data = json.loads(raw.decode("utf-8"))
        else:
            data = json.loads(raw)

        # Ping/Pong 心跳
        if data.get("type") == "url_verification":
            challenge = data.get("challenge", "")
            pong = json.dumps({"challenge": challenge, "type": "url_verification"})
            await self._ws.send(pong)
            return

        # 事件类型
        event_type = data.get("header", {}).get("event_type", "")
        if event_type == "im.message.receive_v1":
            await self._handle_message_event(data)

    async def _handle_message_event(self, data: dict):
        """处理消息接收事件"""
        event = data.get("event", {})
        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {}).get("open_id", "")
        message = event.get("message", {})

        chat_type = message.get("chat_type", "p2p")
        message_id = message.get("message_id", "")
        chat_id = message.get("chat_id", "")
        msg_type = message.get("msg_type", "")

        # 只处理文本消息
        if msg_type != "text":
            return

        # 获取文本内容
        content_str = message.get("content", "{}")
        try:
            content = json.loads(content_str)
            text = content.get("text", "")
        except (json.JSONDecodeError, AttributeError):
            return

        # 纯文本提取（去掉 @bot 等）
        text = re.sub(r'@_user_\d+\s*', '', text).strip()

        if not text:
            return

        feishu_msg = FeishuMessage(
            message_id=message_id,
            chat_id=chat_id,
            sender_id=sender_id,
            text=text,
            chat_type=chat_type,
            raw=data,
        )

        if self._on_message:
            await self._on_message(feishu_msg)

    def stop(self):
        self._running = False


# ──── 飞书通道集成 ────

class FeishuChannel:
    """飞书消息通道 - 将 Feishu 接入 agent"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        agent_runner: Callable[[FeishuMessage, str, str], Awaitable[str]] | None = None,
    ):
        self.client = FeishuClient(app_id, app_secret)
        self.handler = FeishuEventHandler(self.client)
        self._agent_runner = agent_runner

    def set_agent_runner(
        self, runner: Callable[[FeishuMessage, str, str], Awaitable[str]]
    ) -> None:
        """设置 agent 执行器"""
        self._agent_runner = runner
        self.handler.set_message_handler(self._on_message)

    async def _on_message(self, msg: FeishuMessage) -> None:
        """处理飞书消息"""
        if not self._agent_runner:
            await self.client.send_text(
                msg.chat_id, f"收到: {msg.text}",
                message_id=msg.message_id, receive_id_type="chat_id",
            )
            return

        try:
            response = await self._agent_runner(msg, msg.chat_id, msg.sender_id)
            if response:
                max_len = 1800
                first = True
                while len(response) > max_len:
                    chunk = response[:max_len]
                    await self.client.send_text(
                        msg.chat_id, chunk,
                        message_id=msg.message_id if first else None,
                        receive_id_type="chat_id",
                    )
                    response = response[max_len:]
                    first = False
                    await asyncio.sleep(0.5)
                await self.client.send_text(
                    msg.chat_id, response,
                    message_id=msg.message_id if first else None,
                    receive_id_type="chat_id",
                )
        except Exception as e:
            await self.client.send_text(
                msg.chat_id,
                f"❌ Agent error: {e}",
                message_id=msg.message_id,
            )

    async def start(self):
        """启动飞书通道（阻塞）"""
        print(f"  🚀 Starting Feishu channel...")
        await self.handler.run()

    def stop(self):
        self.handler.stop()
