"""系统 Prompt 构建器 - 参考 OpenClaw 的 system-prompt.md

动态组装系统 prompt，包含 workspace bootstrap 文件注入
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_openclaw.config.config import AppConfig


class SystemPromptBuilder:
    """系统 prompt 构建器"""

    def __init__(self, config: AppConfig | None = None):
        self._config = config or AppConfig()

    def build(
        self,
        extra_prompt: str = "",
        bootstrap_files: dict[str, str] | None = None,
        available_skills: list[dict[str, str]] | None = None,
        tools_description: str = "",
    ) -> str:
        """构建完整的系统 prompt"""
        sections: list[str] = []

        # 1. 身份标识
        sections.append("You are a helpful AI assistant running inside OpenClaw-compatible agent runtime.")

        # 2. 工具使用说明
        if tools_description:
            sections.append(f"\n## Tooling\n\n{tools_description}")
        else:
            sections.append("\n## Tooling\n\nYou have access to tools defined by the system.")

        # 3. 执行策略
        sections.append("""
## Execution Bias

- Act on actionable requests in the same turn
- Continue until done or genuinely blocked
- When a tool returns a weak result, try alternative approaches
- Verify before finalizing answers
- Use `exec` for commands that start now; use cron for future follow-ups""")

        # 4. Skills
        if available_skills:
            skill_lines = ["\n## Available Skills\n"]
            for skill in available_skills:
                name = skill.get("name", "unknown")
                desc = skill.get("description", "")
                location = skill.get("location", "")
                skill_lines.append(
                    f'  <skill>\n    <name>{name}</name>\n    <description>{desc}</description>\n    <location>{location}</location>\n  </skill>'
                )
            sections.append("\n".join(skill_lines))

        # 5. Bootstrap context (workspace files)
        if bootstrap_files:
            sections.append("\n## Project Context\n")
            for filename, content in bootstrap_files.items():
                sections.append(f"### /root/.openclaw/workspace/{filename}\n```\n{content}\n```")

        # 6. Runtime info
        sections.append(f"""
## Runtime

- Timezone: {self._config.user_timezone}
- Workspace: {self._config.workspace_dir}
- Session: Ready""")

        # 7. 额外的系统 prompt
        if extra_prompt:
            sections.append(f"\n{extra_prompt}")

        return "\n".join(sections)


def load_bootstrap_files(workspace_dir: str | Path) -> dict[str, str]:
    """加载 workspace 的 bootstrap 文件"""
    ws = Path(workspace_dir)
    bootstrap_files = [
        "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md",
    ]
    result: dict[str, str] = {}
    for filename in bootstrap_files:
        path = ws / filename
        if path.exists():
            content = path.read_text()
            # 截断大文件
            if len(content) > 12000:
                content = content[:6000] + "\n\n... [truncated] ...\n\n" + content[-3000:]
            result[filename] = content
        else:
            result[filename] = f"[{filename} not found]"
    return result
