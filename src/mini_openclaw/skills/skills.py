"""技能系统 - 参考 OpenClaw 的 skills.md

技能发现、加载、过滤
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import re


SKILL_METADATA_RE = re.compile(r"^---\s*$")


class Skill:
    """技能定义"""

    def __init__(
        self,
        name: str,
        description: str = "",
        location: str = "",
        metadata: dict[str, Any] | None = None,
        instructions: str = "",
    ):
        self.name = name
        self.description = description
        self.location = location
        self.metadata = metadata or {}
        self.instructions = instructions

    def to_prompt_entry(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "location": self.location,
        }


class SkillLoader:
    """技能加载器"""

    def __init__(self, skill_dirs: list[str | Path] | None = None):
        self._dirs = [Path(d) for d in (skill_dirs or [])]

    def add_dir(self, d: str | Path) -> None:
        self._dirs.append(Path(d))

    def discover_skills(self) -> list[Skill]:
        """发现所有可用技能"""
        skills: list[Skill] = []
        for d in self._dirs:
            if not d.exists():
                continue
            skills.extend(self._scan_dir(d))
        return skills

    def _scan_dir(self, d: Path) -> list[Skill]:
        skills: list[Skill] = []
        for item in d.iterdir():
            if not item.is_dir():
                continue
            skill_file = item / "SKILL.md"
            if skill_file.exists():
                skill = self._parse_skill(skill_file)
                if skill:
                    skills.append(skill)
        return skills

    def _parse_skill(self, path: Path) -> Skill | None:
        """解析 SKILL.md 文件"""
        content = path.read_text()
        parts = content.split("---", 2)

        if len(parts) < 3:
            return None

        frontmatter_text = parts[1].strip()
        instructions = parts[2].strip()

        # 解析 frontmatter
        metadata: dict[str, Any] = {}
        name = ""
        description = ""

        for line in frontmatter_text.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == "name":
                    name = value
                elif key == "description":
                    description = value
                else:
                    metadata[key] = value

        if not name:
            return None

        return Skill(
            name=name,
            description=description,
            location=str(path),
            metadata=metadata,
            instructions=instructions,
        )
