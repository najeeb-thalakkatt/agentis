"""Tests for agentis.compaction.dedup — FileDeduplicator."""

from __future__ import annotations

import os

import pytest

from agentis.compaction.dedup import DedupResult, FileDeduplicator


class TestFileDeduplicatorBasics:
    def test_construction(self) -> None:
        d = FileDeduplicator(max_inline_tokens=2000)
        assert d._max_inline == 2000

    @pytest.mark.asyncio
    async def test_read_small_file(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "small.txt")
        with open(path, "w") as f:
            f.write("hello world\n")

        d = FileDeduplicator()
        result = await d.read(path)
        assert isinstance(result, DedupResult)
        assert result.status == "inline"
        assert result.content is not None
        assert "hello world" in result.content
        assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path: object) -> None:
        d = FileDeduplicator()
        result = await d.read("/tmp/nonexistent_agentis_test_file.txt")
        assert result.status == "error"
        assert result.content is None


class TestFileDeduplicatorCaching:
    @pytest.mark.asyncio
    async def test_unchanged_file_returns_cached(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "cached.txt")
        with open(path, "w") as f:
            f.write("original content\n")

        d = FileDeduplicator()
        # First read — should be inline
        r1 = await d.read(path)
        assert r1.status == "inline"

        # Second read — file unchanged, should return "unchanged"
        r2 = await d.read(path)
        assert r2.status == "unchanged"
        assert r2.tokens_used < r1.tokens_used

    @pytest.mark.asyncio
    async def test_changed_file_rereads(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "changing.txt")
        with open(path, "w") as f:
            f.write("v1\n")

        d = FileDeduplicator()
        r1 = await d.read(path)
        assert r1.status == "inline"

        # Modify file
        with open(path, "w") as f:
            f.write("v2 different content\n")

        r2 = await d.read(path)
        assert r2.status == "inline"
        assert "v2" in r2.content


class TestFileDeduplicatorLargeFiles:
    @pytest.mark.asyncio
    async def test_large_file_returns_preview(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "large.py")
        # Create a file that exceeds max_inline_tokens
        lines = []
        lines.append("import os")
        lines.append("import sys")
        lines.append("")
        lines.append("class MyClass:")
        lines.append("    def method_one(self):")
        for i in range(200):
            lines.append(f"        x = {i}  # lots of content")
        lines.append("    def method_two(self):")
        lines.append("        pass")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        d = FileDeduplicator(max_inline_tokens=50)  # very low threshold
        result = await d.read(path)
        assert result.status == "large_file"
        assert result.preview is not None
        assert result.content is None
        # Preview should contain structural info (imports, class, def)
        assert "import" in result.preview or "class" in result.preview or "def" in result.preview
        assert result.tokens_used < 50  # preview is compact


class TestFileDeduplicatorLineRange:
    @pytest.mark.asyncio
    async def test_read_line_range(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "ranged.txt")
        with open(path, "w") as f:
            for i in range(20):
                f.write(f"line {i}\n")

        d = FileDeduplicator()
        result = await d.read(path, line_range=(5, 10))
        assert result.status == "inline"
        assert "line 4" in result.content  # 0-indexed content, 1-indexed range
        assert "line 9" in result.content
        assert "line 0" not in result.content
        assert "line 15" not in result.content


class TestFileDeduplicatorInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_forces_reread(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "inv.txt")
        with open(path, "w") as f:
            f.write("original\n")

        d = FileDeduplicator()
        await d.read(path)

        # Invalidate the cache
        d.invalidate(path)

        # Next read should be fresh (inline, not unchanged)
        result = await d.read(path)
        assert result.status == "inline"

    @pytest.mark.asyncio
    async def test_invalidate_all(self, tmp_path: object) -> None:
        d = FileDeduplicator()
        for name in ("a.txt", "b.txt"):
            path = os.path.join(str(tmp_path), name)
            with open(path, "w") as f:
                f.write(f"content of {name}\n")
            await d.read(path)

        d.invalidate_all()

        # Both should read fresh
        for name in ("a.txt", "b.txt"):
            path = os.path.join(str(tmp_path), name)
            result = await d.read(path)
            assert result.status == "inline"


class TestFileDeduplicatorPreview:
    @pytest.mark.asyncio
    async def test_preview_captures_structure(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "structured.py")
        code = """import os
from pathlib import Path

# A module

class Parser:
    def parse(self, data):
        pass

    def validate(self, data):
        pass

def helper():
    pass
"""
        with open(path, "w") as f:
            f.write(code)

        d = FileDeduplicator(max_inline_tokens=10)
        result = await d.read(path)
        assert result.status == "large_file"
        preview = result.preview
        assert "import os" in preview
        assert "class Parser" in preview
        assert "def parse" in preview
        assert "def helper" in preview
