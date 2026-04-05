"""Tests for Layers 3-4, pack registry, and ON_TURN_START/ON_TURN_END events."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from agentis.compaction.compactor import ContextCompactor
from agentis.token_utils import estimate_tokens
from agentis.types import (
    ContextEntry,
    LifecycleEvent,
    Message,
    Priority,
    ProviderResponse,
    TokenUsage,
)


def _entry(
    content: str = "x" * 300,
    priority: Priority = Priority.MEDIUM,
    entry_type: str = "tool_result",
    age_offset: float = 0.0,
) -> ContextEntry:
    return ContextEntry(
        content=content,
        priority=priority,
        token_count=estimate_tokens(content),
        timestamp=time.time() - age_offset,
        entry_type=entry_type,
    )


# ── Layer 3 Tests ─────────────────────────────────────────


class TestLayer3ExtractSessionMemory:
    """Layer 3: Extract durable facts to memory before eviction."""

    @pytest.mark.asyncio
    async def test_layer3_skipped_without_provider(self) -> None:
        c = ContextCompactor(max_tokens=100, utility_provider=None)
        for _ in range(5):
            c.add_entry(_entry(priority=Priority.LOW))
        freed = await c._layer3_extract_session_memory()
        assert freed == 0

    @pytest.mark.asyncio
    async def test_layer3_skipped_without_memory(self) -> None:
        provider = AsyncMock()
        c = ContextCompactor(max_tokens=100, utility_provider=provider, memory=None)
        for _ in range(5):
            c.add_entry(_entry(priority=Priority.LOW))
        freed = await c._layer3_extract_session_memory()
        assert freed == 0

    @pytest.mark.asyncio
    async def test_layer3_extracts_and_saves(self, tmp_path: object) -> None:
        from agentis.memory.index import MemoryIndex

        memory = MemoryIndex(workspace=str(tmp_path))

        # Mock provider returns extracted facts
        provider = AsyncMock()
        provider.complete.return_value = ProviderResponse(
            content="1. Server has 8GB RAM\n2. OOM caused by memory leak\n3. Fix PR #4521 pending",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

        c = ContextCompactor(max_tokens=100, utility_provider=provider, memory=memory)

        # Add enough LOW/MEDIUM entries to trigger extraction
        for i in range(5):
            c.add_entry(_entry(
                content=f"Tool result {i}: detailed investigation data " * 10,
                priority=Priority.LOW,
                entry_type="tool_result",
            ))
        # Add a CRITICAL entry that should survive
        c.add_entry(_entry(content="System prompt", priority=Priority.CRITICAL, entry_type="conversation"))

        initial_count = len(c.get_entries())
        freed = await c._layer3_extract_session_memory()

        assert freed > 0
        # Provider was called to extract facts
        provider.complete.assert_called_once()
        # Memory was updated
        assert len(memory.entries) == 1
        assert memory.entries[0].topic == "Session facts (auto-extracted)"
        assert "auto-extracted" in memory.entries[0].tags
        assert "compaction" in memory.entries[0].tags
        # Source entries removed, replaced by marker
        assert len(c.get_entries()) < initial_count
        # CRITICAL entry survived
        critical = [e for e in c.get_entries() if e.priority == Priority.CRITICAL]
        assert len(critical) == 1

    @pytest.mark.asyncio
    async def test_layer3_skipped_with_few_candidates(self, tmp_path: object) -> None:
        from agentis.memory.index import MemoryIndex

        memory = MemoryIndex(workspace=str(tmp_path))
        provider = AsyncMock()
        c = ContextCompactor(max_tokens=100, utility_provider=provider, memory=memory)

        # Only 2 candidates (need >= 3)
        c.add_entry(_entry(priority=Priority.LOW))
        c.add_entry(_entry(priority=Priority.MEDIUM))

        freed = await c._layer3_extract_session_memory()
        assert freed == 0
        provider.complete.assert_not_called()


# ── Layer 4 Tests ─────────────────────────────────────────


class TestLayer4NuclearSummarization:
    """Layer 4: Nuclear summarization at 90%+ pressure."""

    @pytest.mark.asyncio
    async def test_layer4_skipped_without_provider(self) -> None:
        c = ContextCompactor(max_tokens=100, utility_provider=None)
        freed = await c._layer4_summarize_full_history()
        assert freed == 0

    @pytest.mark.asyncio
    async def test_layer4_skipped_below_90_percent(self) -> None:
        provider = AsyncMock()
        c = ContextCompactor(max_tokens=1000, utility_provider=provider)
        # Add entries totaling ~80% of 1000
        c.add_entry(_entry(content="x" * 2400, priority=Priority.MEDIUM))  # ~800 tokens
        freed = await c._layer4_summarize_full_history()
        assert freed == 0
        provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_layer4_fires_at_high_pressure(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = ProviderResponse(
            content="The user investigated an OOM incident. Root cause was a memory leak in worker-r8t2v.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=200, output_tokens=50),
        )

        c = ContextCompactor(max_tokens=200, utility_provider=provider)
        # Add entries totaling >90% of 200 tokens
        c.add_entry(_entry(content="x" * 600, priority=Priority.MEDIUM, entry_type="conversation"))
        c.add_entry(_entry(content="y" * 300, priority=Priority.LOW, entry_type="tool_result"))
        # Add CRITICAL that should survive
        c.add_entry(_entry(content="System prompt", priority=Priority.CRITICAL, entry_type="conversation"))

        freed = await c._layer4_summarize_full_history()

        assert freed > 0
        provider.complete.assert_called_once()
        # All non-CRITICAL replaced by single summary
        entries = c.get_entries()
        non_critical = [e for e in entries if e.priority != Priority.CRITICAL]
        assert len(non_critical) == 1
        assert "nuclear compaction" in non_critical[0].content.lower()
        # CRITICAL survived
        critical = [e for e in entries if e.priority == Priority.CRITICAL]
        assert len(critical) == 1


# ── Pack Registry Tests ───────────────────────────────────


class TestPackRegistry:
    def test_list_packs(self) -> None:
        from agentis.packs import list_packs

        packs = list_packs()
        assert "filesystem" in packs
        assert "coding" in packs
        assert "shell" in packs
        assert "data" in packs
        assert "web" in packs

    def test_load_pack_filesystem(self) -> None:
        from agentis.packs import load_pack

        tools = load_pack("filesystem")
        names = [t.name for t in tools]
        assert "file_read" in names
        assert "file_write" in names

    def test_load_pack_coding(self) -> None:
        from agentis.packs import load_pack

        tools = load_pack("coding")
        names = [t.name for t in tools]
        assert "grep" in names
        assert "glob" in names

    def test_load_pack_unknown_raises(self) -> None:
        from agentis.packs import load_pack

        with pytest.raises(KeyError, match="Unknown pack"):
            load_pack("nonexistent")

    def test_load_packs_multiple(self) -> None:
        from agentis.packs import load_packs

        tools = load_packs(["filesystem", "coding"])
        names = [t.name for t in tools]
        assert "file_read" in names
        assert "grep" in names

    def test_register_custom_pack(self) -> None:
        from agentis.packs import list_packs, register_pack

        register_pack("custom_test", "agentis.packs.filesystem")
        assert "custom_test" in list_packs()
        # Clean up
        from agentis.packs import _PACK_REGISTRY
        del _PACK_REGISTRY["custom_test"]


# ── Lifecycle Event Tests ─────────────────────────────────


class TestTurnLifecycleEvents:
    def test_on_turn_start_exists(self) -> None:
        assert hasattr(LifecycleEvent, "ON_TURN_START")
        assert LifecycleEvent.ON_TURN_START.value == "on_turn_start"

    def test_on_turn_end_exists(self) -> None:
        assert hasattr(LifecycleEvent, "ON_TURN_END")
        assert LifecycleEvent.ON_TURN_END.value == "on_turn_end"

    def test_total_events_is_10(self) -> None:
        assert len(LifecycleEvent) == 10

    @pytest.mark.asyncio
    async def test_turn_hooks_fire_during_step(self) -> None:
        from agentis.hooks.registry import HookRegistry
        from agentis.memory.index import MemoryIndex
        from agentis.protocols import ProviderCapabilities
        from agentis.runtime.agent import AgentRuntime
        from agentis.types import HookAction, HookContext, HookResponse

        fired_events: list[str] = []

        def track_turn_start(ctx: HookContext) -> HookResponse:
            fired_events.append("turn_start")
            return HookResponse(action=HookAction.ALLOW)

        def track_turn_end(ctx: HookContext) -> HookResponse:
            fired_events.append("turn_end")
            return HookResponse(action=HookAction.ALLOW)

        hooks = HookRegistry()
        hooks.register(LifecycleEvent.ON_TURN_START, track_turn_start, name="track_start")
        hooks.register(LifecycleEvent.ON_TURN_END, track_turn_end, name="track_end")

        class MockProv:
            async def complete(self, messages, tools=None, **kw):
                return ProviderResponse(
                    content="Done.",
                    tool_calls=[],
                    usage=TokenUsage(input_tokens=10, output_tokens=10),
                )
            async def stream(self, messages, tools=None, **kw):
                yield "x"
            def capabilities(self):
                return ProviderCapabilities(name="test", max_context_tokens=128_000)

        import tempfile
        ws = tempfile.mkdtemp()
        runtime = AgentRuntime(
            provider=MockProv(),
            tools=[],
            system_prompt="test",
            memory=MemoryIndex(workspace=ws),
            hooks=hooks,
            max_turns=1,
        )

        async for _ in runtime.steps("hello"):
            pass

        assert "turn_start" in fired_events
        assert "turn_end" in fired_events
