"""Tests for agentis.packs.filesystem — file_read, file_write, file_edit, list_directory."""

from __future__ import annotations

import os

import pytest

from agentis.packs.filesystem import TOOLS, get_filesystem_tools
from agentis.types import Permission


class TestFilesystemPackStructure:
    def test_get_tools_returns_list(self) -> None:
        tools = get_filesystem_tools()
        assert isinstance(tools, list)
        assert len(tools) == 4

    def test_tools_constant_matches(self) -> None:
        assert TOOLS == get_filesystem_tools()

    def test_tool_names(self) -> None:
        names = {t.name for t in get_filesystem_tools()}
        assert names == {"file_read", "file_write", "file_edit", "list_directory"}

    def test_permissions(self) -> None:
        tools = {t.name: t for t in get_filesystem_tools()}
        assert tools["file_read"].permission == Permission.READ_ONLY
        assert tools["list_directory"].permission == Permission.READ_ONLY
        assert tools["file_write"].permission == Permission.MUTATING
        assert tools["file_edit"].permission == Permission.MUTATING

    def test_all_have_schemas(self) -> None:
        for t in get_filesystem_tools():
            schema = t.get_schema()
            assert schema.name == t.name
            assert schema.description


class TestFileRead:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "hello.txt")
        with open(path, "w") as f:
            f.write("hello world\n")

        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["file_read"].execute(path=path)
        assert result.success is True
        assert "hello world" in result.data

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "lines.txt")
        with open(path, "w") as f:
            for i in range(10):
                f.write(f"line {i}\n")

        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["file_read"].execute(path=path, start_line=3, end_line=5)
        assert result.success is True
        assert "line 2" in result.data
        assert "line 4" in result.data
        assert "line 0" not in result.data

    @pytest.mark.asyncio
    async def test_read_nonexistent(self) -> None:
        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["file_read"].execute(path="/tmp/nonexistent_agentis_xyz.txt")
        assert result.success is False


class TestFileWrite:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "new.txt")
        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["file_write"].execute(path=path, content="new content\n")
        assert result.success is True
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "new content\n"

    @pytest.mark.asyncio
    async def test_write_overwrites(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "overwrite.txt")
        with open(path, "w") as f:
            f.write("old")
        tools = {t.name: t for t in get_filesystem_tools()}
        await tools["file_write"].execute(path=path, content="new")
        with open(path) as f:
            assert f.read() == "new"


class TestFileEdit:
    @pytest.mark.asyncio
    async def test_edit_replaces_text(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "edit.txt")
        with open(path, "w") as f:
            f.write("hello world\n")

        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["file_edit"].execute(
            path=path, old_text="hello", new_text="goodbye"
        )
        assert result.success is True
        with open(path) as f:
            assert "goodbye world" in f.read()

    @pytest.mark.asyncio
    async def test_edit_old_text_not_found(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "noedit.txt")
        with open(path, "w") as f:
            f.write("hello\n")

        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["file_edit"].execute(
            path=path, old_text="missing", new_text="replacement"
        )
        assert result.success is False


class TestListDirectory:
    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path: object) -> None:
        for name in ("a.py", "b.txt", "c.md"):
            with open(os.path.join(str(tmp_path), name), "w") as f:
                f.write("")

        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["list_directory"].execute(path=str(tmp_path))
        assert result.success is True
        assert "a.py" in str(result.data)
        assert "b.txt" in str(result.data)

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self) -> None:
        tools = {t.name: t for t in get_filesystem_tools()}
        result = await tools["list_directory"].execute(path="/tmp/nonexistent_agentis_dir_xyz")
        assert result.success is False
