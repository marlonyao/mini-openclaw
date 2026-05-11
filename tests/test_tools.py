"""Exec 和 WebSearch 工具测试"""

import pytest

from mini_openclaw.tools.exec_tool import ExecTool
from mini_openclaw.tools.web_search_tool import WebSearchTool


@pytest.mark.asyncio
class TestExecTool:
    async def test_echo_command(self):
        tool = ExecTool()
        result = await tool.execute({"command": "echo hello"})
        assert not result.is_error
        assert "hello" in result.content

    async def test_exit_code(self):
        tool = ExecTool()
        result = await tool.execute({"command": "false"})
        assert result.is_error
        assert "Exit code: 1" in result.content

    async def test_with_workdir(self):
        tool = ExecTool()
        result = await tool.execute({"command": "pwd", "workdir": "/tmp"})
        assert not result.is_error
        assert "/tmp" in result.content

    async def test_forbidden_command(self):
        tool = ExecTool()
        result = await tool.execute({"command": "rm -rf /"})
        assert result.is_error
        assert "forbidden" in result.content

    async def test_timeout(self):
        tool = ExecTool()
        result = await tool.execute({"command": "sleep 10", "timeout": 1})
        assert result.is_error
        assert "timed out" in result.content

    async def test_empty_command(self):
        tool = ExecTool()
        result = await tool.execute({"command": ""})
        assert result.is_error

    async def test_multi_line_output(self):
        tool = ExecTool()
        result = await tool.execute({"command": "printf 'line1\\nline2\\nline3'"})
        assert not result.is_error
        assert "line1" in result.content
        assert "line3" in result.content

    async def test_long_output_truncation(self):
        """长输出应该被截断"""
        tool = ExecTool()
        result = await tool.execute({
            "command": "python3 -c \"print('x' * 20000)\"",
        })
        assert not result.is_error
        assert "truncated" in result.content
        assert len(result.content) < 12000


@pytest.mark.asyncio
class TestWebSearchTool:
    async def test_search_basic(self):
        """搜索应该返回结果"""
        tool = WebSearchTool()
        result = await tool.execute({"query": "Python programming language", "count": 3})
        # 不检查 is_error 因为网络可能不通,但应返回有意义的内容
        if not result.is_error:
            assert "Python" in result.content
            assert "python" in result.content.lower()

    async def test_search_with_results_count(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": "github", "count": 5})
        if not result.is_error:
            # 应该有多个结果条目
            assert "github" in result.content.lower()

    async def test_empty_query(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": ""})
        assert result.is_error

    async def test_search_no_results(self):
        tool = WebSearchTool()
        result = await tool.execute({"query": "zzzznonexistentkey12345xyz"})
        # 应该有搜索结果但可能是空
        if not result.is_error:
            assert result.content is not None

    async def test_parameters(self):
        tool = WebSearchTool()
        params = tool.parameters
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
