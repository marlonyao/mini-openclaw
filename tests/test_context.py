"""上下文管理、记忆系统、系统 Prompt 测试"""

import os
import tempfile

import pytest

from mini_openclaw.context.compaction import (
    estimate_tokens,
    prune_tool_results,
    compact_session,
    build_compaction_prompt,
    ContextAssembler,
)
from mini_openclaw.session.message import Message
from mini_openclaw.session.session import Session
from mini_openclaw.memory.memory import (
    MemoryStore,
    keyword_search,
    MemorySearcher,
)
from mini_openclaw.prompt.builder import SystemPromptBuilder, load_bootstrap_files
from mini_openclaw.skills.skills import SkillLoader


# ──── 上下文管理测试 ────

class TestContext:
    def test_estimate_tokens(self):
        assert estimate_tokens("hello") >= 1
        assert estimate_tokens("a" * 400) == 100

    def test_prune_tool_results_basic(self):
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="Search"),
            Message(role="tool", content="long_result_" + "x" * 1000, tool_call_id="t1"),
            Message(role="assistant", content="Result here"),
        ]
        pruned = prune_tool_results(msgs, keep_last_n=1, max_result_chars=100)
        assert len(pruned) == 5
        # 只有一条工具结果，keep_last_n=1 所以应该完整保留
        tool_msgs = [m for m in pruned if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert "long_result" in tool_msgs[0].content

    def test_prune_old_tool_results(self):
        msgs = [
            Message(role="user", content="Q1"),
            Message(role="tool", content="old_result", tool_call_id="t1"),
            Message(role="user", content="Q2"),
            Message(role="tool", content="new_result", tool_call_id="t2"),
        ]
        pruned = prune_tool_results(msgs, keep_last_n=1, max_result_chars=10)
        tool_msgs = [m for m in pruned if m.role == "tool"]
        assert len(tool_msgs) == 2
        # 最新的应完整保留（"new_result" 长度 <= 10）
        assert tool_msgs[-1].content == "new_result"
        # 旧的应被替换为占位符
        assert "removed" in tool_msgs[0].content

    def test_prune_soft_truncate(self):
        """大结果应软裁剪"""
        msgs = [
            Message(role="user", content="Q"),
            Message(role="tool", content="x" * 1000, tool_call_id="t1"),
        ]
        pruned = prune_tool_results(msgs, keep_last_n=0, max_result_chars=100)
        tool_msg = [m for m in pruned if m.role == "tool"][0]
        assert "[truncated]" in tool_msg.content
        assert len(tool_msg.content) < len("x" * 1000)

    def test_compact_session(self):
        session = Session(key="test:compact")
        for i in range(5):
            session.add_message(Message(role="user", content=f"Question {i}"))
            session.add_message(Message(role="assistant", content=f"Answer {i}"))

        compacted = compact_session(session, "Summary of old conversation")
        # 压缩后应该包含摘要 + 最近的消息
        assert compacted.messages[0].role == "system"
        assert "Compacted" in compacted.messages[0].content
        # 应该有 1 条摘要 + 最多 6 条最近消息
        assert len(compacted.messages) <= 7

    def test_build_compaction_prompt(self):
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ]
        prompt = build_compaction_prompt(msgs)
        assert "[USER]" in prompt
        assert "Hello" in prompt
        assert "[ASSISTANT]" in prompt
        assert "Summary:" in prompt

    def test_context_assembler(self):
        session = Session(key="test:assembler")
        for _ in range(3):
            session.add_message(Message(role="user", content="Hello"))
            session.add_message(Message(role="assistant", content="Hi"))

        assembler = ContextAssembler(max_context_tokens=1000000)
        msgs, needs_compaction = assembler.assemble(
            session, "You are a helpful assistant.", "New question"
        )
        assert not needs_compaction
        assert len(msgs) == 6  # 3轮对话


# ──── 记忆系统测试 ────

class TestMemory:
    def test_memory_store_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            assert store.get_long_term_memory() == ""

    def test_write_and_read_long_term(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            store.write_long_term_memory("Hello world")
            assert store.get_long_term_memory() == "Hello world"

    def test_append_to_long_term(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            store.write_long_term_memory("First line")
            store.append_to_long_term_memory("Second line")
            content = store.get_long_term_memory()
            assert "First line" in content
            assert "Second line" in content

    def test_daily_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            store.write_daily_note("Today's notes")
            assert store.get_daily_note() == "Today's notes"

    def test_append_to_daily_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            store.write_daily_note("Line 1")
            store.append_to_daily_note("Line 2")
            assert "Line 1" in store.get_daily_note()
            assert "Line 2" in store.get_daily_note()

    def test_keyword_search(self):
        text = "Python is great\nI love Rust\nTypeScript is also good"
        results = keyword_search(text, "Python")
        assert len(results) == 1
        assert results[0]["content"] == "Python is great"

    def test_keyword_search_multiple_keywords(self):
        text = "The quick brown fox\njumps over the lazy dog\nquick fox"
        results = keyword_search(text, "quick fox")
        assert len(results) >= 1

    def test_keyword_search_no_match(self):
        results = keyword_search("hello world", "xyz")
        assert len(results) == 0

    def test_memory_searcher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            store.write_long_term_memory("User prefers Python")
            searcher = MemorySearcher(store)
            results = searcher.search_memory("Python")
            assert len(results) >= 1
            assert results[0]["source"] == "MEMORY.md"

    def test_list_daily_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            store.write_daily_note("test")
            files = store.list_daily_notes()
            assert len(files) == 1


# ──── 系统 Prompt 测试 ────

class TestPrompt:
    def test_build_basic(self):
        builder = SystemPromptBuilder()
        prompt = builder.build()
        assert "You are a helpful AI assistant" in prompt
        assert "Execution Bias" in prompt

    def test_build_with_skills(self):
        builder = SystemPromptBuilder()
        prompt = builder.build(
            available_skills=[
                {"name": "weather", "description": "Get weather", "location": "/skills/weather/SKILL.md"},
            ]
        )
        assert "weather" in prompt

    def test_build_with_bootstrap(self):
        builder = SystemPromptBuilder()
        prompt = builder.build(
            bootstrap_files={"AGENTS.md": "# My Agent\nThis is who I am."}
        )
        assert "AGENTS.md" in prompt
        assert "My Agent" in prompt

    def test_build_with_extra_prompt(self):
        builder = SystemPromptBuilder()
        prompt = builder.build(extra_prompt="Remember to be concise.")
        assert "Remember to be concise" in prompt

    def test_load_bootstrap_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建一些 bootstrap 文件
            with open(os.path.join(tmpdir, "AGENTS.md"), "w") as f:
                f.write("# test agent")
            with open(os.path.join(tmpdir, "SOUL.md"), "w") as f:
                f.write("soul test")

            files = load_bootstrap_files(tmpdir)
            assert "AGENTS.md" in files
            assert files["AGENTS.md"] == "# test agent"


# ──── 技能系统测试 ────

class TestSkills:
    def test_skill_loader_no_dirs(self):
        loader = SkillLoader()
        skills = loader.discover_skills()
        assert len(skills) == 0

    def test_skill_loader_with_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建一个模拟技能
            skill_dir = os.path.join(tmpdir, "weather")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("""---
name: weather
description: Get weather information
---

Use the weather tool to get current weather.
""")

            loader = SkillLoader([tmpdir])
            skills = loader.discover_skills()
            assert len(skills) == 1
            assert skills[0].name == "weather"
            assert skills[0].description == "Get weather information"
            assert "weather tool" in skills[0].instructions

    def test_skill_loader_no_skill_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SkillLoader([tmpdir])
            skills = loader.discover_skills()
            assert len(skills) == 0

    def test_skill_to_prompt_entry(self):
        from mini_openclaw.skills.skills import Skill
        skill = Skill(name="test", description="A test skill", location="/path/to/SKILL.md")
        entry = skill.to_prompt_entry()
        assert entry["name"] == "test"
        assert entry["description"] == "A test skill"
        assert entry["location"] == "/path/to/SKILL.md"
