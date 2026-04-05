"""SkepticalMemory — trust-but-verify discipline for agent memory.

The agent treats its own recall as unreliable. Memory is a hint,
not ground truth. Before acting on any remembered fact, the agent
should verify against the actual source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentis.memory.index import MemoryIndex
    from agentis.types import ToolResult

# Default tools that fetch fresh data — no need to verify before using.
# Users can extend this via SkepticalMemory(extra_read_tools={"my_tool"}).
_DEFAULT_READ_TOOLS = frozenset({
    "file_read", "grep", "glob", "list_directory",
    "recall", "search_db", "http_get",
})


class SkepticalMemory:
    """Implements the trust-but-verify memory discipline.

    Provides:
    - A verification prompt to inject into the system message
    - A guard that only updates memory after successful tool results
    - A heuristic for when the agent should re-read before acting
    """

    VERIFICATION_PROMPT = """## Memory Protocol

Your memory (conversation history, session notes, memory index) is a HINT, not ground truth.

Before you:
- Edit a file -> READ the current file first
- Reference an API -> GREP for its current signature
- Claim a test passes -> RUN it now
- Use a dependency -> CHECK the requirements

NEVER edit code based solely on what you "remember" from earlier in the conversation.
Files change. Other agents may have modified them. Your memory of line numbers
is especially unreliable after context compaction.

Only trust information you have verified in the current turn. Stale memory
from earlier turns may be outdated or changed since you last read it."""

    def __init__(
        self,
        verify_after_turns: int = 5,
        extra_read_tools: set[str] | None = None,
    ) -> None:
        self._verify_after_turns = verify_after_turns
        self._read_tools = _DEFAULT_READ_TOOLS | (extra_read_tools or set())

    def get_verification_prompt(self) -> str:
        """Return the skeptical memory system prompt addition."""
        return self.VERIFICATION_PROMPT

    async def remember_if_successful(
        self,
        memory: MemoryIndex,
        topic: str,
        content: str,
        result: ToolResult,
        tags: list[str] | None = None,
    ) -> str | None:
        """Only update memory after a verified successful action.

        This prevents context pollution from failed edits.
        If the tool result indicates failure, memory stays clean.

        Args:
            memory: The memory index to update.
            topic: Topic name for the memory entry.
            content: Content to store.
            result: The tool result to check for success.
            tags: Optional tags.

        Returns:
            The topic_id if remembered, None if skipped.
        """
        if not result.success:
            return None

        return await memory.remember(topic, content, tags=tags)

    def should_verify(self, tool_name: str, context_age_turns: int) -> bool:
        """Determine if the agent should re-read before acting.

        Read tools fetch fresh data, so they don't need pre-verification.
        Mutating tools acting on old context should verify first.

        Args:
            tool_name: The tool about to be used.
            context_age_turns: How many turns ago the relevant context was fetched.

        Returns:
            True if the agent should re-read/verify before proceeding.
        """
        if tool_name in self._read_tools:
            return False

        return context_age_turns >= self._verify_after_turns
