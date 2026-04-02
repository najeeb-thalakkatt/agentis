"""FileDeduplicator — hash-based dedup and lazy loading for file reads.

Tracks file hashes to avoid re-reading unchanged files. Large files get
a structural preview in context with full content on disk.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import aiofiles

logger = logging.getLogger("agentis")


@dataclass
class DedupResult:
    """Result of a deduplicated file read."""

    status: str  # "inline", "unchanged", "large_file", "error"
    content: str | None
    preview: str | None
    tokens_used: int


@dataclass
class _FileRecord:
    """Internal cache entry for a previously-read file."""

    hash: str
    summary: str
    last_read: float


class FileDeduplicator:
    """Hash-based file deduplication and lazy loading.

    Key patterns:
        1. Hash-based dedup — don't re-read unchanged files
        2. Preview + reference — large files stay compact in context
        3. Targeted reads — line ranges, not whole files
    """

    def __init__(self, max_inline_tokens: int = 2000) -> None:
        self._max_inline = max_inline_tokens
        self._cache: dict[str, _FileRecord] = {}

    async def read(
        self,
        path: str,
        line_range: tuple[int, int] | None = None,
    ) -> DedupResult:
        """Read a file with dedup and lazy loading.

        Args:
            path: File path to read.
            line_range: Optional (start, end) 1-indexed line range.

        Returns:
            DedupResult with status and content/preview.
        """
        # Hash the file
        try:
            current_hash = await self._hash_file(path)
        except FileNotFoundError:
            return DedupResult(
                status="error",
                content=None,
                preview=None,
                tokens_used=5,
            )

        # Check cache — file unchanged?
        if path in self._cache and line_range is None:
            cached = self._cache[path]
            if cached.hash == current_hash:
                return DedupResult(
                    status="unchanged",
                    content=None,
                    preview=None,
                    tokens_used=3,
                )

        # Read the file (or a range)
        try:
            content = await self._read_file(path, line_range)
        except FileNotFoundError:
            return DedupResult(
                status="error",
                content=None,
                preview=None,
                tokens_used=5,
            )

        tokens = len(content) // 3

        # Large file? Return preview only
        if tokens > self._max_inline:
            preview = self._generate_preview(content)
            self._cache[path] = _FileRecord(
                hash=current_hash,
                summary=preview,
                last_read=time.time(),
            )
            return DedupResult(
                status="large_file",
                content=None,
                preview=preview,
                tokens_used=len(preview) // 3,
            )

        # Small file — return inline
        self._cache[path] = _FileRecord(
            hash=current_hash,
            summary=f"{content.count(chr(10))} lines",
            last_read=time.time(),
        )
        return DedupResult(
            status="inline",
            content=content,
            preview=None,
            tokens_used=max(tokens, 1),
        )

    def invalidate(self, path: str) -> None:
        """Remove a file from the cache (call after edits)."""
        self._cache.pop(path, None)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    @staticmethod
    async def _hash_file(path: str) -> str:
        """Compute MD5 hash of a file."""
        h = hashlib.md5()
        async with aiofiles.open(path, "rb") as f:
            content = await f.read()
            h.update(content)
        return h.hexdigest()

    @staticmethod
    async def _read_file(
        path: str,
        line_range: tuple[int, int] | None = None,
    ) -> str:
        """Read a file, optionally returning only a line range."""
        async with aiofiles.open(path, "r") as f:
            content = await f.read()

        if line_range:
            lines = content.split("\n")
            start, end = line_range
            # Convert 1-indexed to 0-indexed
            selected = lines[start - 1:end]
            return "\n".join(selected)

        return content

    @staticmethod
    def _generate_preview(content: str) -> str:
        """Generate a structural preview of file content.

        Shows imports, class/function signatures, and headings — not
        the full body. This gives the LLM enough structure to decide
        if it needs the full content.
        """
        lines = content.split("\n")
        preview_lines: list[str] = []
        structural_keywords = (
            "import ", "from ", "class ", "def ",
            "async def ", "function ", "export ",
            "interface ", "# ", "## ", "### ",
        )

        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(stripped.startswith(kw) for kw in structural_keywords):
                preview_lines.append(f"L{i + 1}: {line}")

        return "\n".join(preview_lines[:30])
