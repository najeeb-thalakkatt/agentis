"""Tests for agentis.packs.shell — bash tool."""

from __future__ import annotations

import pytest

from agentis.packs.shell import TOOLS, get_shell_tools
from agentis.types import Permission


class TestShellPackStructure:
    def test_get_tools_returns_list(self) -> None:
        tools = get_shell_tools()
        assert len(tools) == 1

    def test_bash_is_dangerous(self) -> None:
        tools = {t.name: t for t in get_shell_tools()}
        assert tools["bash"].permission == Permission.DANGEROUS

    def test_tools_constant(self) -> None:
        assert TOOLS == get_shell_tools()


class TestBashTool:
    @pytest.mark.asyncio
    async def test_echo(self) -> None:
        tools = {t.name: t for t in get_shell_tools()}
        result = await tools["bash"].execute(command="echo hello")
        assert result.success is True
        assert "hello" in result.data

    @pytest.mark.asyncio
    async def test_returns_exit_code(self) -> None:
        tools = {t.name: t for t in get_shell_tools()}
        result = await tools["bash"].execute(command="true")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_failed_command(self) -> None:
        tools = {t.name: t for t in get_shell_tools()}
        result = await tools["bash"].execute(command="false")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        tools = {t.name: t for t in get_shell_tools()}
        result = await tools["bash"].execute(command="sleep 10", timeout=1)
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_captures_stderr(self) -> None:
        tools = {t.name: t for t in get_shell_tools()}
        result = await tools["bash"].execute(command="echo err >&2")
        # stderr should be captured somewhere in the output
        assert "err" in result.data or "err" in result.full_output
