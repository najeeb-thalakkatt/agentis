"""Tests for agentis.packs.coding — grep, glob tools."""

from __future__ import annotations

import os

import pytest

from agentis.packs.coding import get_coding_tools
from agentis.types import Permission


class TestCodingPackStructure:
    def test_get_tools_returns_list(self) -> None:
        tools = get_coding_tools()
        assert len(tools) >= 2

    def test_tool_names(self) -> None:
        names = {t.name for t in get_coding_tools()}
        assert "grep" in names
        assert "glob" in names

    def test_grep_is_read_only(self) -> None:
        tools = {t.name: t for t in get_coding_tools()}
        assert tools["grep"].permission == Permission.READ_ONLY

    def test_glob_is_read_only(self) -> None:
        tools = {t.name: t for t in get_coding_tools()}
        assert tools["glob"].permission == Permission.READ_ONLY


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_grep_finds_matches(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "code.py")
        with open(path, "w") as f:
            f.write("def hello():\n    return 'world'\n\ndef goodbye():\n    pass\n")

        tools = {t.name: t for t in get_coding_tools()}
        result = await tools["grep"].execute(pattern="def ", path=str(tmp_path))
        assert result.success is True
        assert len(result.data) == 2  # two def lines

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "empty.py")
        with open(path, "w") as f:
            f.write("# nothing here\n")

        tools = {t.name: t for t in get_coding_tools()}
        result = await tools["grep"].execute(pattern="class ", path=str(tmp_path))
        assert result.success is True
        assert len(result.data) == 0

    @pytest.mark.asyncio
    async def test_grep_nonexistent_path(self) -> None:
        tools = {t.name: t for t in get_coding_tools()}
        result = await tools["grep"].execute(pattern="x", path="/tmp/nonexistent_xyz_agentis")
        assert result.success is False


class TestGlobTool:
    @pytest.mark.asyncio
    async def test_glob_finds_files(self, tmp_path: object) -> None:
        for name in ("a.py", "b.py", "c.txt"):
            with open(os.path.join(str(tmp_path), name), "w") as f:
                f.write("")

        tools = {t.name: t for t in get_coding_tools()}
        result = await tools["glob"].execute(pattern="*.py", path=str(tmp_path))
        assert result.success is True
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, tmp_path: object) -> None:
        tools = {t.name: t for t in get_coding_tools()}
        result = await tools["glob"].execute(pattern="*.rs", path=str(tmp_path))
        assert result.success is True
        assert len(result.data) == 0
