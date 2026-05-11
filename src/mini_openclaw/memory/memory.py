"""记忆系统 - 参考 OpenClaw 的 memory.md

长期记忆（MEMORY.md）+ 每日笔记（memory/YYYY-MM-DD.md）+ 记忆搜索
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    """文件化记忆存储"""

    def __init__(self, workspace_dir: str | Path):
        self._workspace = Path(workspace_dir)
        self._memory_path = self._workspace / "MEMORY.md"
        self._memory_dir = self._workspace / "memory"

    def get_long_term_memory(self) -> str:
        """读取 MEMORY.md"""
        if self._memory_path.exists():
            return self._memory_path.read_text()
        return ""

    def write_long_term_memory(self, content: str) -> None:
        """写入 MEMORY.md"""
        self._memory_path.write_text(content)

    def append_to_long_term_memory(self, text: str) -> None:
        """追加到 MEMORY.md"""
        existing = self.get_long_term_memory()
        if existing:
            new_content = existing.rstrip() + "\n\n" + text
        else:
            new_content = text
        self.write_long_term_memory(new_content)

    def get_daily_note(self, date: datetime | None = None) -> str:
        """读取指定日期的笔记"""
        if date is None:
            date = datetime.now(timezone.utc)
        path = self._get_daily_path(date)
        if path.exists():
            return path.read_text()
        return ""

    def write_daily_note(self, content: str, date: datetime | None = None) -> None:
        """写入每日笔记"""
        if date is None:
            date = datetime.now(timezone.utc)
        path = self._get_daily_path(date)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def append_to_daily_note(self, text: str) -> None:
        """追加到今日笔记"""
        existing = self.get_daily_note()
        if existing:
            new_content = existing.rstrip() + "\n" + text
        else:
            new_content = text
        self.write_daily_note(new_content)

    def _get_daily_path(self, date: datetime) -> Path:
        filename = date.strftime("%Y-%m-%d.md")
        return self._memory_dir / filename

    def list_daily_notes(self) -> list[str]:
        """列出所有每日笔记文件名"""
        if not self._memory_dir.exists():
            return []
        return sorted(
            f.name for f in self._memory_dir.iterdir()
            if f.suffix == ".md" and re.match(r"\d{4}-\d{2}-\d{2}", f.name)
        )

    def get_recent_notes(self, days: int = 2) -> list[tuple[str, str]]:
        """获取最近几天的笔记"""
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        results: list[tuple[str, str]] = []
        for i in range(days):
            date = now - timedelta(days=i)
            content = self.get_daily_note(date)
            if content:
                results.append((date.strftime("%Y-%m-%d"), content))
        return results


def keyword_search(text: str, query: str) -> list[dict[str, Any]]:
    """
    简单的关键词搜索

    返回匹配的行及其上下文。
    """
    if not text or not query:
        return []

    query_lower = query.lower()
    keywords = query_lower.split()
    lines = text.split("\n")
    results: list[dict[str, Any]] = []

    for i, line in enumerate(lines):
        line_lower = line.lower()
        if all(kw in line_lower for kw in keywords):
            # 获取上下文（前后各一行）
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            context = "\n".join(lines[start:end])
            results.append({
                "line": i + 1,
                "content": line.strip(),
                "context": context,
                "score": sum(line_lower.count(kw) for kw in keywords),
            })

    # 按匹配分数排序
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


class MemorySearcher:
    """记忆搜索器"""

    def __init__(self, store: MemoryStore | None = None):
        self._store = store

    def search_memory(self, query: str) -> list[dict[str, Any]]:
        """
        搜索所有记忆（MEMORY.md + 最近的每日笔记）
        """
        results: list[dict[str, Any]] = []

        if self._store:
            # 搜索长期记忆
            long_term = self._store.get_long_term_memory()
            if long_term:
                for r in keyword_search(long_term, query):
                    r["source"] = "MEMORY.md"
                    results.append(r)

            # 搜索最近的每日笔记
            for date_str, content in self._store.get_recent_notes(days=7):
                for r in keyword_search(content, query):
                    r["source"] = f"memory/{date_str}.md"
                    results.append(r)

        # 去重
        seen = set()
        unique: list[dict[str, Any]] = []
        for r in results:
            key = (r["source"], r["line"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:20]  # 最多返回20条
