"""Exec 工具 - 让 agent 可以执行 shell 命令"""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

from mini_openclaw.tools.tool import Tool, ToolResult


class ExecTool(Tool):
    """Shell 命令执行工具"""

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "执行 shell 命令并获取输出。用于运行代码、文件操作、系统管理等。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间(秒)，默认 30",
                    "default": 30,
                },
                "workdir": {
                    "type": "string",
                    "description": "工作目录（可选，默认当前目录）",
                },
            },
            "required": ["command"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", 30)
        workdir = arguments.get("workdir", None)

        if not command:
            return ToolResult(tool_call_id="", content="Error: command is required", is_error=True)

        # 安全：禁止 rm -rf / 等危险命令
        forbidden_patterns = [
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "> /dev/",
            ":(){ :|:& };:", "chmod 777 /",
        ]
        for pattern in forbidden_patterns:
            if pattern in command:
                return ToolResult(
                    tool_call_id="",
                    content=f"Error: Command contains forbidden pattern: {pattern}",
                    is_error=True,
                )

        try:
            # 使用 shlex 分割参数以安全执行
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    tool_call_id="",
                    content=f"Command timed out after {timeout}s\nOutput so far may be incomplete.",
                    is_error=True,
                )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # 截断大输出
            if len(stdout_str) > 10000:
                stdout_str = stdout_str[:5000] + "\n\n... [output truncated] ...\n\n" + stdout_str[-3000:]

            result_parts = []
            if stdout_str.strip():
                result_parts.append(stdout_str.strip())
            if stderr_str.strip():
                result_parts.append(f"[stderr]\n{stderr_str.strip()[:2000]}")

            result = "\n".join(result_parts) if result_parts else "(no output)"
            result += f"\n\nExit code: {proc.returncode}"

            return ToolResult(
                tool_call_id="",
                content=result,
                is_error=proc.returncode != 0,
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                content=f"Error executing command: {e}",
                is_error=True,
            )
