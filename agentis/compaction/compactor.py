"""ContextCompactor — five-layer tiered eviction for long-running sessions.

Each layer is progressively more aggressive. The compactor runs layers
sequentially until token usage drops below 80% of max capacity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agentis.types import ContextEntry, Message, Priority

logger = logging.getLogger("agentis")


@dataclass
class CompactionResult:
    """Result of a compaction cycle."""

    layers_run: int
    tokens_freed: int
    entries_removed: int


class ContextCompactor:
    """Five-layer context compaction engine.

    Layers (least to most destructive):
        1. Clear stale tool results (replace with one-line summary)
        2. Summarize old conversation turns (requires utility_provider)
        3. Extract durable facts to session memory (requires utility_provider)
        4. Summarize full history (requires utility_provider)
        5. Truncate oldest lowest-priority entries (last resort)

    Layers 2-4 require a utility_provider for LLM-based summarization.
    If no utility_provider is configured, those layers are skipped.
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        utility_provider: Any = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._utility_provider = utility_provider
        self._entries: list[ContextEntry] = []

    def current_tokens(self) -> int:
        """Return total tokens across all entries."""
        return sum(e.token_count for e in self._entries)

    def add_entry(self, entry: ContextEntry) -> None:
        """Add a context entry."""
        self._entries.append(entry)

    def get_entries(self) -> list[ContextEntry]:
        """Return all context entries."""
        return list(self._entries)

    def get_recent_history(self, n: int = 5) -> list[ContextEntry]:
        """Return the N most recent entries."""
        return self._entries[-n:]

    async def compact(self) -> CompactionResult:
        """Run compaction layers until under 85% budget.

        Returns a CompactionResult describing what happened.
        """
        threshold = self._max_tokens * 0.85
        if self.current_tokens() <= threshold:
            return CompactionResult(layers_run=0, tokens_freed=0, entries_removed=0)

        layers = [
            self._layer1_clear_stale_tool_results,
            self._layer2_summarize_old_conversation,
            self._layer3_extract_session_memory,
            self._layer4_summarize_full_history,
            self._layer5_truncate_oldest,
        ]

        total_freed = 0
        total_removed = 0
        layers_run = 0
        initial_count = len(self._entries)

        for layer_fn in layers:
            if self.current_tokens() <= threshold:
                break
            freed = await layer_fn()
            total_freed += freed
            layers_run += 1

        total_removed = initial_count - len(self._entries)

        logger.info(
            "Compaction: %d layers, freed %d tokens, removed %d entries",
            layers_run, total_freed, total_removed,
        )

        return CompactionResult(
            layers_run=layers_run,
            tokens_freed=total_freed,
            entries_removed=total_removed,
        )

    async def _layer1_clear_stale_tool_results(self) -> int:
        """Layer 1: Replace stale tool results with one-line summaries.

        Only affects MEDIUM, LOW, and EPHEMERAL priority tool_result entries.
        """
        freed = 0
        for entry in self._entries:
            if (
                entry.entry_type == "tool_result"
                and entry.priority >= Priority.MEDIUM
            ):
                old_tokens = entry.token_count
                summary = self._one_line_summary(entry.content)
                entry.content = summary
                entry.token_count = len(summary) // 3
                freed += old_tokens - entry.token_count
        return freed

    async def _layer2_summarize_old_conversation(self) -> int:
        """Layer 2: Summarize old conversation turns, keeping last 5 verbatim.

        Requires utility_provider. Skipped if not available.
        """
        if self._utility_provider is None:
            return 0

        convos = [
            (i, e) for i, e in enumerate(self._entries)
            if e.entry_type == "conversation"
        ]

        if len(convos) <= 8:
            return 0

        old_entries = convos[:-8]
        old_content = "\n".join(e.content for _, e in old_entries)
        old_tokens = sum(e.token_count for _, e in old_entries)

        # Ask LLM to summarize
        try:
            response = await self._utility_provider.complete(
                messages=[
                    Message(
                        role="system",
                        content="Summarize the following conversation concisely.",
                    ),
                    Message(role="user", content=old_content),
                ],
                max_tokens=500,
            )
            summary = response.content
        except Exception as e:
            logger.warning("Layer 2 summarization failed: %s", e)
            return 0

        # Remove old entries (in reverse order to preserve indices)
        for idx, _ in reversed(old_entries):
            self._entries.pop(idx)

        # Insert summary at the beginning
        summary_tokens = len(summary) // 3
        self._entries.insert(0, ContextEntry(
            content=f"[Earlier conversation summary]\n{summary}",
            priority=Priority.MEDIUM,
            token_count=summary_tokens,
            timestamp=old_entries[0][1].timestamp,
            entry_type="summary",
        ))

        freed = old_tokens - summary_tokens
        return max(freed, 0)

    async def _layer3_extract_session_memory(self) -> int:
        """Layer 3: Extract durable facts to session memory.

        Requires utility_provider. Skipped if not available.
        """
        if self._utility_provider is None:
            return 0

        # Stub: in a full implementation this would extract facts
        # and save them to memory, then remove the source entries.
        return 0

    async def _layer4_summarize_full_history(self) -> int:
        """Layer 4: Nuclear summarization — summarize everything.

        Requires utility_provider. Skipped if not available.
        """
        if self._utility_provider is None:
            return 0

        # Stub: would summarize entire context into ~500 tokens.
        return 0

    async def _layer5_truncate_oldest(self) -> int:
        """Layer 5: Drop lowest-priority entries. Last resort.

        Never removes CRITICAL entries.
        """
        target = int(self._max_tokens * 0.75)
        freed = 0

        # Sort by priority (highest number = lowest priority = remove first)
        # then by timestamp (oldest first within same priority)
        removable = [
            (i, e) for i, e in enumerate(self._entries)
            if e.priority != Priority.CRITICAL
        ]
        removable.sort(key=lambda x: (-x[1].priority, x[1].timestamp))

        indices_to_remove: list[int] = []
        for idx, entry in removable:
            if self.current_tokens() - freed <= target:
                break
            freed += entry.token_count
            indices_to_remove.append(idx)

        # Remove in reverse order to preserve indices
        for idx in sorted(indices_to_remove, reverse=True):
            self._entries.pop(idx)

        return freed

    @staticmethod
    def _one_line_summary(content: str) -> str:
        """Create a one-line summary of content."""
        lines = content.strip().split("\n")
        return f"[{len(lines)} lines — summarized]"
