"""上下文管理 - 参考 OpenClaw 的 compaction.md 和 session-pruning.md

包含 Compaction（对话压缩/摘要）和 Session Pruning（工具结果裁剪）
"""

from __future__ import annotations

import json
from typing import Any

from mini_openclaw.session.message import Message
from mini_openclaw.session.session import Session


def estimate_tokens(text: str) -> int:
    """字符级 token 估算"""
    return max(1, len(text) // 4)


def prune_tool_results(
    messages: list[Message],
    keep_last_n: int = 3,
    max_result_chars: int = 500,
) -> list[Message]:
    """
    裁剪旧工具结果（参考 OpenClaw 的 session-pruning.md）

    只裁剪工具结果消息，保留对话文本。
    最近 keep_last_n 个工具结果完整保留。
    """
    tool_result_indices = [
        i for i, m in enumerate(messages)
        if m.role == "tool"
    ]

    # 保留最近的几个
    if keep_last_n > 0 and len(tool_result_indices) <= keep_last_n:
        return messages

    if keep_last_n <= 0:
        prune_indices = set(tool_result_indices)
    else:
        prune_indices = set(tool_result_indices[:-keep_last_n])
    result: list[Message] = []
    for i, msg in enumerate(messages):
        if i in prune_indices:
            if msg.content and len(msg.content) > max_result_chars:
                # 软裁剪：保留头和尾
                head = msg.content[:max_result_chars // 2]
                tail = msg.content[-(max_result_chars // 4):]
                pruned = Message(
                    role=msg.role,
                    content=f"{head}\n... [truncated] ...\n{tail}",
                    tool_call_id=msg.tool_call_id,
                )
                result.append(pruned)
            else:
                # 硬清除：替换为占位符
                placeholder = Message(
                    role=msg.role,
                    content="[tool result removed - already processed by model]",
                    tool_call_id=msg.tool_call_id,
                )
                result.append(placeholder)
        else:
            result.append(msg)

    return result


def build_compaction_prompt(messages: list[Message]) -> str:
    """
    构建压缩摘要 prompt

    将历史消息转换为摘要用的提示文本。
    """
    lines: list[str] = [
        "Please summarize the following conversation, preserving important:",
        "- Decisions and conclusions made",
        "- Key facts and information discovered",
        "- User preferences and requirements",
        "- Tool outputs that contained important data",
        "",
        "Conversation history:",
    ]

    for msg in messages:
        role_label = msg.role.upper()
        if msg.role == "tool":
            content_preview = (msg.content or "")[:200]
            lines.append(f"[TOOL RESULT] {content_preview}")
        elif msg.tool_calls:
            calls = ", ".join(tc.name for tc in msg.tool_calls)
            content = msg.content or ""
            lines.append(f"[ASSISTANT (tools: {calls})] {content[:200]}")
        else:
            content = msg.content or ""
            lines.append(f"[{role_label}] {content[:500]}")

    lines.extend([
        "",
        "Summary:",
    ])
    return "\n".join(lines)


def compact_session(session: Session, summary: str) -> Session:
    """
    压缩会话：用摘要替换旧消息

    参考 OpenClaw 的 compact() 方法。
    保留最近的 N 条消息，旧的换成压缩摘要。
    """
    recent_count = min(6, len(session.messages))
    recent_messages = session.messages[-recent_count:] if recent_count > 0 else []

    # 构建压缩后的会话
    compacted = Session(key=session.key, session_id=session.id)

    # 加入摘要
    compacted.add_message(Message(
        role="system",
        content=f"[Compacted conversation summary]\n{summary}",
    ))

    # 加入最近的消息
    for msg in recent_messages:
        compacted.add_message(msg)

    return compacted


class ContextAssembler:
    """上下文组装器 - 参考 OpenClaw 的 context-engine.md"""

    def __init__(
        self,
        max_context_tokens: int = 128000,
        reserve_tokens: int = 4000,
    ):
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens

    def assemble(
        self,
        session: Session,
        system_prompt: str,
        new_message: str,
    ) -> tuple[list[Message], bool]:
        """
        组装上下文，决定哪些消息发送给 LLM

        Returns: (messages, needs_compaction)
        """
        # 估算当前上下文大小
        system_tokens = estimate_tokens(system_prompt)
        new_msg_tokens = estimate_tokens(new_message)
        total_tokens = system_tokens + session.token_estimate() + new_msg_tokens

        available = self.max_context_tokens - self.reserve_tokens
        needs_compaction = total_tokens > available

        if needs_compaction:
            # 需要压缩：先裁剪工具结果
            pruned = prune_tool_results(session.get_messages())
            # 估算裁剪后的 token
            pruned_tokens = sum(m.token_estimate() for m in pruned)
            total_tokens = system_tokens + pruned_tokens + new_msg_tokens

            if total_tokens > available:
                # 还是超，需要强制压缩
                # 只保留最后几轮对话
                recent = pruned[-10:] if len(pruned) > 10 else pruned
                return recent, True

            return pruned, False

        return session.get_messages(), False
