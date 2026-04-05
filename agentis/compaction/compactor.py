"""ContextCompactor — five-layer tiered eviction for long-running sessions.

Each layer is progressively more aggressive. The compactor runs layers
sequentially until token usage drops below 80% of max capacity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agentis.token_utils import estimate_tokens
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
        memory: Any = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._utility_provider = utility_provider
        self._memory = memory  # MemoryIndex for Layer 3 extraction
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
                entry.token_count = estimate_tokens(summary)
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
        summary_tokens = estimate_tokens(summary)
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
        """Layer 3: Extract durable facts to session memory before eviction.

        Asks the utility LLM to pull 2-5 key facts from LOW/MEDIUM entries,
        saves them to MemoryIndex with [auto-extracted, compaction] tags,
        then evicts the source entries and leaves a compact marker.

        Requires utility_provider and memory. Skipped if either is missing.
        """
        if self._utility_provider is None or self._memory is None:
            return 0

        # Collect eviction candidates: LOW and MEDIUM priority, non-CRITICAL
        candidates = [
            (i, e) for i, e in enumerate(self._entries)
            if e.priority in (Priority.LOW, Priority.MEDIUM)
            and e.entry_type in ("conversation", "tool_result")
        ]

        if len(candidates) < 3:
            return 0  # Not enough material to extract from

        # Build the content to extract from
        content_block = "\n---\n".join(e.content for _, e in candidates)
        old_tokens = sum(e.token_count for _, e in candidates)

        # Ask LLM to extract key facts
        try:
            response = await self._utility_provider.complete(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Extract 2-5 key durable facts from the following conversation "
                            "and tool results. Output ONLY a numbered list of facts. "
                            "Each fact should be a single sentence capturing something "
                            "the agent would want to remember in future turns. "
                            "Focus on: decisions made, errors found, key data points, "
                            "and current state."
                        ),
                    ),
                    Message(role="user", content=content_block),
                ],
                max_tokens=300,
            )
            extracted = response.content
        except Exception as e:
            logger.warning("Layer 3 fact extraction failed: %s", e)
            return 0

        if not extracted or len(extracted.strip()) < 10:
            return 0

        # Save extracted facts to memory
        try:
            await self._memory.remember(
                topic="Session facts (auto-extracted)",
                content=extracted,
                tags=["auto-extracted", "compaction"],
            )
        except Exception as e:
            logger.warning("Layer 3 memory save failed: %s", e)
            return 0

        # Remove the source entries
        for idx, _ in reversed(candidates):
            self._entries.pop(idx)

        # Insert a compact marker
        marker = f"[{len(candidates)} entries extracted to memory — see 'Session facts (auto-extracted)']"
        marker_tokens = estimate_tokens(marker)
        self._entries.insert(0, ContextEntry(
            content=marker,
            priority=Priority.MEDIUM,
            token_count=marker_tokens,
            timestamp=candidates[0][1].timestamp,
            entry_type="summary",
        ))

        freed = old_tokens - marker_tokens
        logger.info("Layer 3: extracted %d facts from %d entries, freed %d tokens",
                     extracted.count("\n") + 1, len(candidates), freed)
        return max(freed, 0)

    async def _layer4_summarize_full_history(self) -> int:
        """Layer 4: Nuclear summarization — compress entire non-CRITICAL context.

        Fires at 90% pressure. Replaces the entire non-CRITICAL conversation
        with a single 5-8 sentence paragraph preserving the user's goal,
        decisions made, current state, and blockers.

        Requires utility_provider. Skipped if not available.
        """
        if self._utility_provider is None:
            return 0

        # Only fire at high pressure (90%+)
        if self.current_tokens() < self._max_tokens * 0.90:
            return 0

        # Collect all non-CRITICAL entries
        removable = [
            (i, e) for i, e in enumerate(self._entries)
            if e.priority != Priority.CRITICAL
        ]

        if not removable:
            return 0

        all_content = "\n---\n".join(e.content for _, e in removable)
        old_tokens = sum(e.token_count for _, e in removable)

        # Ask LLM for nuclear summary
        try:
            response = await self._utility_provider.complete(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Compress the following conversation into a single paragraph "
                            "of 5-8 sentences. Preserve:\n"
                            "1. The user's original goal/request\n"
                            "2. Key decisions made so far\n"
                            "3. Current state (what has been done, what remains)\n"
                            "4. Any blockers or errors encountered\n"
                            "5. Important data points or findings\n"
                            "Output ONLY the summary paragraph, no headers or bullets."
                        ),
                    ),
                    Message(role="user", content=all_content),
                ],
                max_tokens=400,
            )
            summary = response.content
        except Exception as e:
            logger.warning("Layer 4 nuclear summarization failed: %s", e)
            return 0

        if not summary or len(summary.strip()) < 20:
            return 0

        # Remove all non-CRITICAL entries
        for idx, _ in reversed(removable):
            self._entries.pop(idx)

        # Insert the nuclear summary
        summary_tokens = estimate_tokens(summary)
        self._entries.insert(0, ContextEntry(
            content=f"[Full session summary — nuclear compaction]\n{summary}",
            priority=Priority.HIGH,
            token_count=summary_tokens,
            timestamp=removable[0][1].timestamp,
            entry_type="summary",
        ))

        freed = old_tokens - summary_tokens
        logger.info("Layer 4: nuclear summary, freed %d tokens (%d entries -> 1)",
                     freed, len(removable))
        return max(freed, 0)

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
        """Create a one-line summary preserving the first meaningful line."""
        lines = content.strip().split("\n")
        # Find the first non-empty line as a content hint
        first_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_line = stripped[:80]
                break
        if first_line:
            return f"[{len(lines)} lines] {first_line}"
        return f"[{len(lines)} lines — summarized]"
