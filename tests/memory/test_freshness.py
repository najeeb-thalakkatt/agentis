"""Tests for memory freshness tracking and compaction-memory interaction.

Gap 1: MemoryIndex loses created_at on load (set to 0.0).
Gap 2: Compaction Layer 1 summarizes remember() tool results.
"""

from __future__ import annotations

import time

import pytest

from agentis.memory.index import MemoryIndex, MemoryPointer


class TestTimestampPersistence:
    """Verify that created_at survives save/load round-trips."""

    @pytest.fixture
    def memory(self, tmp_path: object) -> MemoryIndex:
        return MemoryIndex(workspace=str(tmp_path))

    @pytest.mark.asyncio
    async def test_created_at_set_on_remember(self, memory: MemoryIndex) -> None:
        before = time.time()
        await memory.remember("Test topic", "content")
        after = time.time()

        entry = memory.entries[0]
        assert entry.created_at >= before
        assert entry.created_at <= after

    @pytest.mark.asyncio
    async def test_created_at_persists_through_save_load(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember("Persistent", "content")
        original_ts = mem1.entries[0].created_at
        assert original_ts > 0

        # Reload in a new instance
        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()

        reloaded_ts = mem2.entries[0].created_at
        # Timestamp should survive round-trip (within 1s tolerance for float precision)
        assert abs(reloaded_ts - original_ts) < 1.0, (
            f"Timestamp lost on reload: original={original_ts}, reloaded={reloaded_ts}"
        )

    @pytest.mark.asyncio
    async def test_created_at_not_zero_after_load(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember("Topic", "content")

        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()

        assert mem2.entries[0].created_at > 0.0, "created_at should not be 0.0 after load"

    @pytest.mark.asyncio
    async def test_multiple_entries_preserve_timestamps(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember("First", "content 1")
        ts1 = mem1.entries[0].created_at

        await mem1.remember("Second", "content 2", tags=["tag1"])
        ts2 = [e for e in mem1.entries if e.topic == "Second"][0].created_at

        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()

        loaded = {e.topic: e.created_at for e in mem2.entries}
        assert abs(loaded["First"] - ts1) < 1.0
        assert abs(loaded["Second"] - ts2) < 1.0

    @pytest.mark.asyncio
    async def test_tags_still_work_with_timestamps(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember("Tagged", "content", tags=["infra", "db"])

        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()

        entry = mem2.entries[0]
        assert entry.tags == ["infra", "db"]
        assert entry.created_at > 0.0


class TestParserEdgeCases:
    """Verify _parse_index_line handles special characters in topic names."""

    @pytest.mark.asyncio
    async def test_topic_with_at_sign(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember("Meeting @ 3pm notes", "discussed auth flow")

        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()

        assert len(mem2.entries) == 1
        assert mem2.entries[0].topic == "Meeting @ 3pm notes"
        assert mem2.entries[0].created_at > 0.0

    @pytest.mark.asyncio
    async def test_topic_with_brackets(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember("Config [production]", "uses Redis cluster", tags=["infra"])

        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()

        assert len(mem2.entries) == 1
        assert mem2.entries[0].topic == "Config [production]"
        assert mem2.entries[0].tags == ["infra"]
        assert mem2.entries[0].created_at > 0.0

    @pytest.mark.asyncio
    async def test_topic_with_at_and_brackets(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember(
            "Deploy @ 2pm [hotfix]", "rolled back v3.2.1", tags=["ops", "incident"]
        )

        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()

        assert len(mem2.entries) == 1
        entry = mem2.entries[0]
        assert entry.topic == "Deploy @ 2pm [hotfix]"
        assert entry.tags == ["ops", "incident"]
        assert entry.created_at > 0.0

        # Verify recall still works
        content = await mem2.recall(entry.topic_id)
        assert content is not None
        assert "rolled back" in content


class TestFreshnessIndicator:
    """Verify freshness info is exposed in context payload and recall."""

    @pytest.fixture
    def memory(self, tmp_path: object) -> MemoryIndex:
        return MemoryIndex(workspace=str(tmp_path))

    @pytest.mark.asyncio
    async def test_context_payload_includes_age(self, memory: MemoryIndex) -> None:
        await memory.remember("Old topic", "content")

        msg = memory.get_context_payload()
        # Should contain some age/freshness indicator
        assert "ago" in msg.content.lower() or "saved" in msg.content.lower(), (
            f"Context payload should include freshness indicator:\n{msg.content}"
        )

    @pytest.mark.asyncio
    async def test_recall_tool_includes_freshness(self, tmp_path: object) -> None:
        """RecallTool result should include freshness metadata."""
        from agentis.memory.recall_tool import RecallTool

        memory = MemoryIndex(workspace=str(tmp_path))
        topic_id = await memory.remember("Test", "content here")

        tool = RecallTool(memory=memory)
        result = await tool.execute(topic_id=topic_id)

        assert result.success
        # The result should mention when the memory was saved
        output = result.full_output or result.summary
        assert "saved" in output.lower() or "ago" in output.lower(), (
            f"Recall result should include freshness: {output}"
        )


class TestCompactionMemoryPriority:
    """Verify that remember() tool results get CRITICAL priority in compaction sync."""

    @pytest.mark.asyncio
    async def test_remember_results_get_critical_priority(self, tmp_path: object) -> None:
        from agentis.types import Message, Priority

        memory = MemoryIndex(workspace=str(tmp_path))
        from agentis.protocols import ProviderCapabilities

        # Create a mock provider
        class MockProv:
            async def complete(self, messages, tools=None, **kw):
                pass
            async def stream(self, messages, tools=None, **kw):
                yield "x"
            def capabilities(self):
                return ProviderCapabilities(name="test", max_context_tokens=128_000)

        from agentis.runtime.agent import AgentRuntime

        runtime = AgentRuntime(
            provider=MockProv(),
            tools=[],
            system_prompt="test",
            memory=memory,
            max_turns=1,
        )

        # Add messages including a remember tool result
        runtime._session.add_message(Message(role="user", content="investigate"))
        runtime._session.add_message(Message(role="assistant", content="saving findings"))
        runtime._session.add_message(Message(
            role="tool",
            content="Memory saved: Root Cause -> abc12345",
            metadata={"tool_use_id": "mem1", "tool_call_id": "mem1", "tool_name": "remember"},
        ))
        runtime._session.add_message(Message(role="assistant", content="checking more"))
        runtime._session.add_message(Message(
            role="tool",
            content="Pods in production: api-server running",
            metadata={"tool_use_id": "t2", "tool_call_id": "t2", "tool_name": "get_pods"},
        ))

        # Sync to compactor
        runtime._sync_session_to_compactor()

        # Find the remember-related tool result entry
        entries = runtime._compactor.get_entries()
        memory_entry = None
        regular_entry = None
        for e in entries:
            if "Memory saved:" in e.content:
                memory_entry = e
            elif "Pods in production" in e.content:
                regular_entry = e

        assert memory_entry is not None, "Memory-anchor entry not found in compactor"
        assert memory_entry.priority == Priority.CRITICAL, (
            f"Memory-anchor should be CRITICAL, got {memory_entry.priority}"
        )

        # Regular tool result should NOT be CRITICAL
        assert regular_entry is not None
        assert regular_entry.priority != Priority.CRITICAL, (
            "Regular tool result should not be CRITICAL priority"
        )

    @pytest.mark.asyncio
    async def test_content_with_memory_saved_but_wrong_tool_not_critical(
        self, tmp_path: object,
    ) -> None:
        """A tool whose OUTPUT contains 'Memory saved:' but isn't the remember
        tool should NOT get CRITICAL priority. Detection is structural (metadata),
        not content-based."""
        from agentis.types import Message, Priority

        memory = MemoryIndex(workspace=str(tmp_path))
        from agentis.protocols import ProviderCapabilities

        class MockProv:
            async def complete(self, messages, tools=None, **kw):
                pass
            async def stream(self, messages, tools=None, **kw):
                yield "x"
            def capabilities(self):
                return ProviderCapabilities(name="test", max_context_tokens=128_000)

        from agentis.runtime.agent import AgentRuntime

        runtime = AgentRuntime(
            provider=MockProv(),
            tools=[],
            system_prompt="test",
            memory=memory,
            max_turns=1,
        )

        # A grep tool that happens to return content containing "Memory saved:"
        runtime._session.add_message(Message(role="user", content="search logs"))
        runtime._session.add_message(Message(
            role="tool",
            content="Log line 42: Memory saved: Root Cause -> abc12345 (from agent session)",
            metadata={"tool_use_id": "g1", "tool_call_id": "g1", "tool_name": "grep"},
        ))

        runtime._sync_session_to_compactor()
        entries = runtime._compactor.get_entries()

        grep_entry = [e for e in entries if "Log line 42" in e.content]
        assert len(grep_entry) == 1
        assert grep_entry[0].priority != Priority.CRITICAL, (
            "grep result should not be CRITICAL just because content mentions 'Memory saved:'"
        )
