"""工具系统 - 参考 OpenClaw 的 pi-tools 和 tool-definition-adapter

工具注册、schema 定义、执行框架
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from mini_openclaw.models.provider import ToolDef


class ToolCall(BaseModel):
    """LLM 发起的工具调用请求"""
    id: str
    name: str
    arguments: str = "{}"  # JSON string

    def parse_arguments(self) -> dict[str, Any]:
        import json
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}


class ToolResult(BaseModel):
    """工具执行结果"""
    tool_call_id: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema 参数定义"""
        return {
            "type": "object",
            "properties": {},
        }

    def to_tool_def(self) -> ToolDef:
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        ...


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_tool_defs(self) -> list[ToolDef]:
        return [t.to_tool_def() for t in self._tools.values()]

    async def execute_tool_call(self, call: ToolCall) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Error: Tool '{call.name}' not found",
                is_error=True,
            )
        try:
            args = call.parse_arguments()
            return await tool.execute(args)
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Error executing {call.name}: {e}",
                is_error=True,
            )

    async def execute_batch(
        self, calls: list[ToolCall]
    ) -> list[ToolResult]:
        """并行执行一批工具调用"""
        tasks = [self.execute_tool_call(c) for c in calls]
        return await asyncio.gather(*tasks)


# ──── 内置工具 ────


class EchoTool(Tool):
    """回声工具 - 用于测试"""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back the input message"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back",
                }
            },
            "required": ["message"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        msg = arguments.get("message", "")
        return ToolResult(tool_call_id="", content=f"Echo: {msg}")


class ReadTool(Tool):
    """读取文件内容"""

    def __init__(self, workspace_dir: str = "."):
        self._workspace = workspace_dir

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "Read contents of a file"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
            },
            "required": ["path"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path", "")
        import os.path
        if not os.path.exists(path):
            return ToolResult(tool_call_id="", content=f"File not found: {path}", is_error=True)
        with open(path) as f:
            content = f.read()
        return ToolResult(tool_call_id="", content=content)


class WriteTool(Tool):
    """写入文件"""

    def __init__(self, workspace_dir: str = "."):
        self._workspace = workspace_dir

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "Write content to a file"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return ToolResult(tool_call_id="", content=f"Written {len(content)} bytes to {path}")
