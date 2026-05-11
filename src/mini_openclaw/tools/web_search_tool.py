"""Web Search 工具 - 让 agent 可以搜索互联网"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from mini_openclaw.tools.tool import Tool, ToolResult


class WebSearchTool(Tool):
    """搜索引擎工具 - 使用 DuckDuckGo（免费，无需 API key）"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "搜索互联网，返回相关网页标题和摘要。用于查找信息、最新新闻等。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "count": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query", "")
        count = min(arguments.get("count", 5), 10)

        if not query:
            return ToolResult(tool_call_id="", content="Error: query is required", is_error=True)

        try:
            results = await self._search_duckduckgo(query, count)
            if not results:
                results = await self._search_fallback(query, count)

            if not results:
                return ToolResult(
                    tool_call_id="",
                    content=f"No results found for: {query}",
                )

            lines = [f"## Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                lines.append(f"{i}. [{title}]({url})")
                if snippet:
                    lines.append(f"   {snippet}")
                lines.append("")

            return ToolResult(tool_call_id="", content="\n".join(lines).strip())

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Search error: {e}",
                is_error=True,
            )

    async def _search_duckduckgo(self, query: str, count: int) -> list[dict]:
        """通过 DuckDuckGo 搜索"""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            })
            html = resp.text

        # 提取搜索结果
        results = []
        # 匹配 DuckDuckGo 结果格式
        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        snippets = re.findall(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )

        for i, (url, title) in enumerate(blocks):
            if i >= count:
                break
            # 清理 HTML 标签
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

            results.append({
                "title": title_clean,
                "url": url,
                "snippet": snippet,
            })

        return results

    async def _search_fallback(self, query: str, count: int) -> list[dict]:
        """备用搜索：使用 Bing 基本搜索"""
        url = f"https://www.bing.com/search?q={quote_plus(query)}"

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            })
            html = resp.text

        results = []
        # 提取 Bing 搜索结果
        blocks = re.findall(
            r'<h2><a[^>]*href="([^"]*)"[^>]*>(.*?)</a></h2>',
            html,
            re.DOTALL,
        )
        snippets = re.findall(
            r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>',
            html,
            re.DOTALL,
        )

        for i, (url, title) in enumerate(blocks):
            if i >= count:
                break
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            results.append({"title": title_clean, "url": url, "snippet": snippet})

        return results
