"""Playwright 浏览器自动化工具 - mini-openclaw skill"""

from __future__ import annotations

import asyncio
import re
import tempfile
from typing import Any

import httpx

from mini_openclaw.tools.tool import Tool, ToolResult


def _shell_quote(s: str) -> str:
    """Shell 安全引用"""
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"


class BrowserTool(Tool):
    """浏览器自动化工具 - 使用 Playwright + httpx"""

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return "控制浏览器：打开页面并提取内容、截图、保存为PDF"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["open", "screenshot", "pdf"],
                    "description": "操作: open=打开页面展示内容, screenshot=截图, pdf=保存PDF",
                },
                "url": {
                    "type": "string",
                    "description": "目标 URL",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS 选择器（截取页面特定区域，仅 screenshot）",
                },
            },
            "required": ["action", "url"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action", "")
        url = arguments.get("url", "")

        if not url:
            return ToolResult(tool_call_id="", content="Error: URL is required", is_error=True)

        try:
            if action == "open":
                return await self._open_page(url)
            elif action == "screenshot":
                return await self._screenshot(url, arguments.get("selector"))
            elif action == "pdf":
                return await self._pdf(url)
            else:
                return ToolResult(tool_call_id="", content=f"Unknown action: {action}", is_error=True)
        except Exception as e:
            return ToolResult(tool_call_id="", content=f"Browser error: {e}", is_error=True)

    async def _open_page(self, url: str) -> ToolResult:
        """获取页面文本内容"""
        img_path = tempfile.mktemp(suffix=".png")
        quoted_url = _shell_quote(url)

        # 先截图确保页面可访问
        cmd = f"playwright screenshot --full-page {quoted_url} {_shell_quote(img_path)} 2>/dev/null"
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)

        # 用 httpx 下载页面内容
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                html = resp.text
        except Exception as e:
            html = f"<error>Failed to fetch: {e}</error>"

        text = self._extract_text(html)[:3000]

        result = (
            f"## Page: {url}\n\n"
            f"Content:\n{text[:2000]}\n\n"
            f"Screenshot: {img_path}\n"
            f"HTML length: {len(html)} chars, Text: {len(text)} chars"
        )
        return ToolResult(tool_call_id="", content=result)

    async def _screenshot(self, url: str, selector: str | None = None) -> ToolResult:
        """截图"""
        out_path = tempfile.mktemp(suffix=".png")
        quoted_url = _shell_quote(url)
        quoted_out = _shell_quote(out_path)

        if selector:
            cmd = f"playwright screenshot --selector={_shell_quote(selector)} {quoted_url} {quoted_out} 2>&1"
        else:
            cmd = f"playwright screenshot --full-page {quoted_url} {quoted_out} 2>&1"

        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        result_parts = [f"Screenshot saved to: {out_path}"]
        stdout_text = stdout.decode(errors="replace")[:300]
        if stdout_text.strip():
            result_parts.append(stdout_text.strip())
        return ToolResult(
            tool_call_id="",
            content="\n".join(result_parts),
            metadata={"screenshot_path": out_path},
        )

    async def _pdf(self, url: str) -> ToolResult:
        """保存为 PDF"""
        out_path = tempfile.mktemp(suffix=".pdf")
        cmd = f"playwright pdf {_shell_quote(url)} {_shell_quote(out_path)} 2>&1"

        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        result_parts = [f"PDF saved to: {out_path}"]
        stdout_text = stdout.decode(errors="replace")[:300]
        if stdout_text.strip():
            result_parts.append(stdout_text.strip())
        return ToolResult(
            tool_call_id="",
            content="\n".join(result_parts),
            metadata={"pdf_path": out_path},
        )

    @staticmethod
    def _extract_text(html: str) -> str:
        """从 HTML 提取可读文本"""
        # 去掉 script/style 标签及其内容
        text = re.sub(
            r'<(script|style)[^>]*>.*?</\1>', '',
            html, flags=re.DOTALL | re.IGNORECASE,
        )
        # 去掉 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 解码常见 HTML 实体
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        # 合并空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 按句子换行（简化）
        lines = re.split(r'(?<=[.!?])\s+', text)
        return '\n'.join(lines)
