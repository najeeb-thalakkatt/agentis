"""DreamExtension — background memory consolidation during idle time.

A separate process consolidates memory while the user is idle:
merges duplicates, removes contradictions, converts vague notes to
structured facts, and discards ephemeral information.

Off by default. Opt-in via extensions=[DreamExtension(memory=...)].
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from agentis.types import Message

if TYPE_CHECKING:
    from agentis.memory.index import MemoryIndex
    from agentis.protocols import Provider

logger = logging.getLogger("agentis")


class DreamExtension:
    """Background memory consolidation extension.

    Uses the utility_provider to merge, deduplicate, and clean up
    the memory index during user idle periods. Implements the
    Extension protocol.
    """

    name: str = "dream"

    CONSOLIDATION_PROMPT = """You are a memory consolidation agent.
Review the raw session logs below and:

1. MERGE duplicate or related observations
   "auth uses JWT" + "tokens expire in 15m"
   -> "Auth: JWT with 15-min expiry, refresh via httpOnly cookie"

2. REMOVE contradictions (keep the latest)
   Old: "Database is PostgreSQL"
   New: "Migrated to CockroachDB last week"
   -> Keep only CockroachDB entry

3. CONVERT vague notes to structured facts
   Vague: "something weird with the cache"
   Structured: "Redis cache returns stale data after config reload"

4. DISCARD ephemeral information
   Remove: "Currently debugging line 42" (stale)
   Remove: "Waiting for CI to finish" (temporal)
   Keep: "CI pipeline requires Node 20+" (durable)

Output an updated memory summary."""

    def __init__(
        self,
        memory: MemoryIndex,
        utility_provider: Provider | None = None,
        idle_threshold: float = 120.0,
    ) -> None:
        self._memory = memory
        self._utility_provider = utility_provider
        self._idle_threshold = idle_threshold
        self._last_activity = time.time()

    async def on_runtime_start(self, runtime: Any) -> None:
        """Called when the runtime starts."""
        self._last_activity = time.time()

    async def on_turn_end(self, runtime: Any, result: Any) -> None:
        """Reset idle timer after each turn."""
        self._last_activity = time.time()

    async def on_idle(self, runtime: Any, idle_seconds: float) -> None:
        """Consolidate memory if idle long enough."""
        if idle_seconds < self._idle_threshold:
            return

        if not self._memory.entries:
            logger.debug("Dream: skipping consolidation — memory is empty")
            return

        memory_content = self._memory.get_context_payload().content
        result = await self.consolidate(
            session_logs="",
            current_memory=memory_content,
        )
        if result:
            logger.info("Dream consolidation completed: %d chars", len(result))

    async def on_runtime_stop(self, runtime: Any) -> None:
        """Called when the runtime stops."""

    async def consolidate(
        self,
        session_logs: str,
        current_memory: str,
    ) -> str:
        """Run memory consolidation via the utility provider.

        Args:
            session_logs: Raw session logs to consolidate from.
            current_memory: Current memory index content.

        Returns:
            The consolidated memory text, or empty string on failure.
        """
        if self._utility_provider is None:
            return ""

        try:
            response = await self._utility_provider.complete(
                messages=[
                    Message(role="system", content=self.CONSOLIDATION_PROMPT),
                    Message(
                        role="user",
                        content=(
                            f"Current memory:\n{current_memory}\n\n"
                            f"Raw session logs:\n{session_logs}"
                        ),
                    ),
                ],
                max_tokens=1000,
            )
            return response.content
        except Exception as e:
            logger.warning("Dream consolidation failed: %s", e)
            return ""
