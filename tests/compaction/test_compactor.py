"""Tests for agentis.compaction.compactor — ContextCompactor."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentis.compaction.compactor import CompactionResult, ContextCompactor
from agentis.types import ContextEntry, Message, Priority, ProviderResponse, TokenUsage


def _entry(
    content: str = "x" * 300,
    priority: Priority = Priority.MEDIUM,
    entry_type: str = "tool_result",
    age_offset: float = 0.0,
) -> ContextEntry:
    """Helper to create a ContextEntry."""
    token_count = len(content) // 3
    return ContextEntry(
        content=content,
        priority=priority,
        token_count=token_count,
        timestamp=time.time() - age_offset,
        entry_type=entry_type,
    )


class TestCompactorBasics:
    def test_construction_defaults(self) -> None:
        c = ContextCompactor(max_tokens=1000)
        assert c.current_tokens() == 0
        assert c._max_tokens == 1000

    def test_add_entry(self) -> None:
        c = ContextCompactor(max_tokens=1000)
        e = _entry("hello world")
        c.add_entry(e)
        assert c.current_tokens() == e.token_count
        assert len(c.get_entries()) == 1

    def test_current_tokens_sums(self) -> None:
        c = ContextCompactor(max_tokens=10000)
        c.add_entry(_entry("a" * 300))  # 100 tokens
        c.add_entry(_entry("b" * 600))  # 200 tokens
        assert c.current_tokens() == 300

    def test_get_recent_history(self) -> None:
        c = ContextCompactor(max_tokens=10000)
        for i in range(10):
            c.add_entry(_entry(f"entry {i}", entry_type="conversation"))
        recent = c.get_recent_history(n=3)
        assert len(recent) == 3


class TestCompactorLayer1:
    """Layer 1: Clear stale tool results."""

    @pytest.mark.asyncio
    async def test_layer1_summarizes_stale_tool_results(self) -> None:
        c = ContextCompactor(max_tokens=100)
        # Add a large stale tool result (MEDIUM priority)
        big = _entry("x" * 900, priority=Priority.MEDIUM, entry_type="tool_result")
        c.add_entry(big)

        freed = await c._layer1_clear_stale_tool_results()
        assert freed > 0
        # Content should be replaced with a short summary
        assert c.get_entries()[0].token_count < 300

    @pytest.mark.asyncio
    async def test_layer1_preserves_critical(self) -> None:
        c = ContextCompactor(max_tokens=100)
        critical = _entry("important", priority=Priority.CRITICAL, entry_type="tool_result")
        c.add_entry(critical)
        original_tokens = c.current_tokens()

        await c._layer1_clear_stale_tool_results()
        assert c.current_tokens() == original_tokens  # unchanged

    @pytest.mark.asyncio
    async def test_layer1_preserves_high_priority(self) -> None:
        c = ContextCompactor(max_tokens=100)
        high = _entry("x" * 300, priority=Priority.HIGH, entry_type="tool_result")
        c.add_entry(high)
        original_tokens = c.current_tokens()

        await c._layer1_clear_stale_tool_results()
        assert c.current_tokens() == original_tokens


class TestCompactorLayer2:
    """Layer 2: Summarize old conversations (requires utility_provider)."""

    @pytest.mark.asyncio
    async def test_layer2_skipped_without_provider(self) -> None:
        c = ContextCompactor(max_tokens=100)
        for i in range(12):
            c.add_entry(_entry(f"turn {i}", entry_type="conversation"))
        original_count = len(c.get_entries())

        freed = await c._layer2_summarize_old_conversation()
        assert freed == 0  # no provider, no summarization
        assert len(c.get_entries()) == original_count

    @pytest.mark.asyncio
    async def test_layer2_summarizes_with_provider(self) -> None:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ProviderResponse(
                content="Summary of earlier conversation.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=50, output_tokens=20),
            )
        )

        c = ContextCompactor(max_tokens=100, utility_provider=mock_provider)
        for i in range(12):
            c.add_entry(
                _entry(f"conversation turn {i} " * 10, entry_type="conversation")
            )

        freed = await c._layer2_summarize_old_conversation()
        assert freed > 0
        # Old turns replaced by a summary entry
        entries = [e for e in c.get_entries() if e.entry_type in ("conversation", "summary")]
        # Should have fewer entries (8 recent + 1 summary)
        assert len(entries) < 12


class TestCompactorLayer5:
    """Layer 5: Truncate oldest (last resort)."""

    @pytest.mark.asyncio
    async def test_layer5_removes_lowest_priority(self) -> None:
        c = ContextCompactor(max_tokens=100)
        c.add_entry(_entry("ephemeral", priority=Priority.EPHEMERAL))
        c.add_entry(_entry("critical", priority=Priority.CRITICAL))
        c.add_entry(_entry("low", priority=Priority.LOW))

        await c._layer5_truncate_oldest()
        # Should remove EPHEMERAL and LOW first, keep CRITICAL
        remaining_priorities = [e.priority for e in c.get_entries()]
        assert Priority.CRITICAL in remaining_priorities

    @pytest.mark.asyncio
    async def test_layer5_never_removes_critical(self) -> None:
        c = ContextCompactor(max_tokens=10)
        c.add_entry(_entry("vital", priority=Priority.CRITICAL))

        await c._layer5_truncate_oldest()
        assert len(c.get_entries()) == 1
        assert c.get_entries()[0].priority == Priority.CRITICAL


class TestCompactorFullCycle:
    @pytest.mark.asyncio
    async def test_compact_under_threshold_noop(self) -> None:
        c = ContextCompactor(max_tokens=10000)
        c.add_entry(_entry("small"))
        result = await c.compact()
        assert result.layers_run == 0

    @pytest.mark.asyncio
    async def test_compact_runs_layers_until_under_budget(self) -> None:
        c = ContextCompactor(max_tokens=200)
        # Add enough to exceed 85% (170 tokens)
        for _ in range(5):
            c.add_entry(
                _entry("x" * 300, priority=Priority.MEDIUM, entry_type="tool_result")
            )
        assert c.current_tokens() > 170

        result = await c.compact()
        assert isinstance(result, CompactionResult)
        assert result.layers_run > 0
        assert result.tokens_freed > 0

    @pytest.mark.asyncio
    async def test_compact_result_fields(self) -> None:
        c = ContextCompactor(max_tokens=100)
        for _ in range(3):
            c.add_entry(
                _entry("x" * 300, priority=Priority.EPHEMERAL, entry_type="tool_result")
            )

        result = await c.compact()
        assert isinstance(result.layers_run, int)
        assert isinstance(result.tokens_freed, int)
        assert isinstance(result.entries_removed, int)

    @pytest.mark.asyncio
    async def test_compact_with_mixed_priorities(self) -> None:
        c = ContextCompactor(max_tokens=200)
        c.add_entry(_entry("system prompt", priority=Priority.CRITICAL))
        c.add_entry(
            _entry("x" * 600, priority=Priority.EPHEMERAL, entry_type="tool_result")
        )
        c.add_entry(
            _entry("x" * 600, priority=Priority.LOW, entry_type="conversation")
        )

        await c.compact()
        # Critical should survive
        critical = [e for e in c.get_entries() if e.priority == Priority.CRITICAL]
        assert len(critical) == 1
