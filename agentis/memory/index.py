"""MemoryIndex — lightweight pointers in context, full data on disk.

The index is always small enough to fit in the LLM context window.
Actual content lives in per-topic files fetched on demand via recall().
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiofiles

from agentis.types import Message

logger = logging.getLogger("agentis")


@dataclass
class MemoryPointer:
    """A lightweight pointer to a topic file on disk."""

    topic: str
    topic_id: str
    file_path: str
    created_at: float
    summary: str
    tags: list[str] = field(default_factory=list)


class MemoryIndex:
    """Index-based memory: store pointers, not data.

    The index (list of MemoryPointer) is always in context.
    Full content lives in topic files under {workspace}/topics/.
    Use recall(topic_id) to fetch full content on demand.
    """

    def __init__(self, workspace: str | Path = ".agentis/memory") -> None:
        self._workspace = Path(workspace)
        self._topics_dir = self._workspace / "topics"
        self._index_path = self._workspace / "MEMORY.md"
        self._pointers: dict[str, MemoryPointer] = {}
        self._dirs_ensured = False

    def _ensure_dirs(self) -> None:
        """Ensure workspace directories exist (lazy, called once)."""
        if not self._dirs_ensured:
            os.makedirs(self._topics_dir, exist_ok=True)
            self._dirs_ensured = True

    @property
    def entries(self) -> list[MemoryPointer]:
        """Return all memory pointers, ordered by creation time."""
        return sorted(self._pointers.values(), key=lambda p: p.created_at)

    async def remember(
        self,
        topic: str,
        content: str,
        tags: list[str] | None = None,
    ) -> str:
        """Store a topic in memory.

        Writes full content to a topic file on disk and adds a lightweight
        pointer to the index. If a topic with the same name exists, it is
        overwritten.

        Args:
            topic: Human-readable topic name.
            content: Full content to store on disk.
            tags: Optional tags for search/filtering.

        Returns:
            The topic_id (12-char hex string).
        """
        topic_id = self._generate_topic_id(topic)
        topic_path = self._topics_dir / f"{topic_id}.md"

        # Ensure directories exist before first write
        await asyncio.to_thread(self._ensure_dirs)

        # Write full content to topic file
        async with aiofiles.open(topic_path, "w") as f:
            await f.write(f"# {topic}\n\n{content}\n")

        # Create or update the pointer
        self._pointers[topic_id] = MemoryPointer(
            topic=topic,
            topic_id=topic_id,
            file_path=f"topics/{topic_id}.md",
            tags=tags or [],
            created_at=time.time(),
            summary=content[:100] if len(content) > 100 else content,
        )

        await self._save_index()
        return topic_id

    async def recall(self, topic_id: str) -> str | None:
        """Fetch full content for a topic by its ID.

        Args:
            topic_id: The 12-char hex ID returned by remember().

        Returns:
            The full topic content, or None if not found.
        """
        topic_path = self._topics_dir / f"{topic_id}.md"
        try:
            async with aiofiles.open(topic_path, "r") as f:
                return await f.read()
        except FileNotFoundError:
            return None

    async def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryPointer]:
        """Search the index by topic name substring and/or tags.

        Args:
            query: Substring to search for in topic names.
            tags: Tags to filter by (any match).

        Returns:
            List of matching MemoryPointer objects.
        """
        results = []
        for ptr in self._pointers.values():
            match = True
            if query and query.lower() not in ptr.topic.lower():
                match = False
            if tags and not any(t in ptr.tags for t in tags):
                match = False
            if match:
                results.append(ptr)
        return results

    async def forget(self, topic_id: str) -> bool:
        """Remove a topic from memory.

        Deletes both the pointer and the topic file on disk.

        Args:
            topic_id: The topic ID to remove.

        Returns:
            True if the topic was found and removed, False otherwise.
        """
        if topic_id not in self._pointers:
            return False

        del self._pointers[topic_id]

        # Remove topic file (non-blocking)
        topic_path = self._topics_dir / f"{topic_id}.md"
        try:
            await asyncio.to_thread(os.remove, topic_path)
        except FileNotFoundError:
            pass

        await self._save_index()
        return True

    def get_context_payload(self) -> Message:
        """Build the memory index message for LLM context.

        Returns a system message containing compact pointers (~150 chars each).
        Includes freshness indicator when timestamps are available.
        """
        lines = ["Your memory index (use recall tool to fetch details):\n"]
        now = time.time()
        for ptr in self.entries:
            tag_str = f" [{', '.join(ptr.tags)}]" if ptr.tags else ""
            age_str = ""
            if ptr.created_at > 0:
                age_str = f" (saved {self._format_age(now - ptr.created_at)})"
            lines.append(f"- {ptr.topic}: ./{ptr.file_path}{age_str}{tag_str}")

        if len(self._pointers) == 0:
            lines.append("(empty)")

        return Message(role="system", content="\n".join(lines))

    @staticmethod
    def _format_age(seconds: float) -> str:
        """Format an age in seconds as a human-readable string."""
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            mins = int(seconds // 60)
            return f"{mins}m ago"
        if seconds < 86400:
            hours = int(seconds // 3600)
            return f"{hours}h ago"
        days = int(seconds // 86400)
        return f"{days}d ago"

    async def load(self) -> None:
        """Load the index from disk.

        Call this when creating a new MemoryIndex instance that should
        pick up state from a previous session.
        """
        if not self._index_path.exists():
            return

        try:
            async with aiofiles.open(self._index_path, "r") as f:
                content = await f.read()
        except FileNotFoundError:
            return

        self._pointers.clear()
        for line in content.split("\n"):
            line = line.strip()
            if not line.startswith("- "):
                continue
            self._parse_index_line(line)

    async def _save_index(self) -> None:
        """Persist the index to MEMORY.md on disk."""
        lines = ["# Agent Memory Index\n"]
        for ptr in self.entries:
            tag_str = f" [{', '.join(ptr.tags)}]" if ptr.tags else ""
            ts_str = f" @{ptr.created_at:.2f}" if ptr.created_at > 0 else ""
            lines.append(f"- {ptr.topic}: ./{ptr.file_path}{ts_str}{tag_str}")

        async with aiofiles.open(self._index_path, "w") as f:
            await f.write("\n".join(lines) + "\n")

    def _parse_index_line(self, line: str) -> None:
        """Parse a single line from MEMORY.md back into a MemoryPointer.

        Format: ``- Topic Name: ./topics/abcd1234.md @1234567890.12 [tag1, tag2]``
        The ``@timestamp`` and ``[tags]`` parts are optional.  Topic names may
        contain ``@`` or ``[`` characters — parsing anchors on ``: ./`` and
        the ``.md`` suffix to avoid ambiguity.
        """
        line = line[2:]  # strip "- "

        # Split topic from path+metadata on the first ": ./"
        if ": ./" not in line:
            return
        topic, rest = line.split(": ./", 1)

        # The file path ends at ".md". Everything after is metadata.
        md_idx = rest.find(".md")
        if md_idx < 0:
            return
        file_path = rest[:md_idx + 3]  # e.g. "topics/abcd1234.md"
        suffix = rest[md_idx + 3:]     # e.g. " @1234567890.12 [tag1, tag2]"

        # Extract tags from suffix (always at the end)
        tags: list[str] = []
        if " [" in suffix and suffix.endswith("]"):
            bracket_start = suffix.rindex(" [")
            tag_str = suffix[bracket_start + 2:-1]
            tags = [t.strip() for t in tag_str.split(",")]
            suffix = suffix[:bracket_start]

        # Extract timestamp from suffix
        created_at = 0.0
        if " @" in suffix:
            at_idx = suffix.rindex(" @")
            ts_str = suffix[at_idx + 2:]
            try:
                created_at = float(ts_str)
            except ValueError:
                pass

        topic_id = Path(file_path).stem

        self._pointers[topic_id] = MemoryPointer(
            topic=topic,
            topic_id=topic_id,
            file_path=file_path,
            tags=tags,
            created_at=created_at,
            summary="",
        )

    @staticmethod
    def _generate_topic_id(topic: str) -> str:
        """Generate a deterministic 12-char hex ID from a topic name."""
        return hashlib.sha256(topic.encode()).hexdigest()[:12]
