"""Tests for agentis.memory.index — MemoryIndex."""

from __future__ import annotations

import pytest

from agentis.memory.index import MemoryIndex, MemoryPointer


class TestMemoryPointer:
    def test_construction(self) -> None:
        ptr = MemoryPointer(
            topic="Auth architecture",
            topic_id="a1b2c3d4",
            file_path="topics/a1b2c3d4.md",
            tags=["security"],
            created_at=1000.0,
            summary="Auth: JWT with 15-min expiry",
        )
        assert ptr.topic == "Auth architecture"
        assert ptr.topic_id == "a1b2c3d4"
        assert ptr.tags == ["security"]

    def test_default_tags(self) -> None:
        ptr = MemoryPointer(
            topic="test",
            topic_id="abc",
            file_path="topics/abc.md",
            created_at=0.0,
            summary="test",
        )
        assert ptr.tags == []


class TestMemoryIndex:
    @pytest.fixture
    def memory(self, tmp_path: object) -> MemoryIndex:
        return MemoryIndex(workspace=str(tmp_path))

    # ── remember ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_remember_returns_topic_id(self, memory: MemoryIndex) -> None:
        topic_id = await memory.remember("Auth system", "Uses JWT tokens")
        assert isinstance(topic_id, str)
        assert len(topic_id) == 8

    @pytest.mark.asyncio
    async def test_remember_deterministic_id(self, memory: MemoryIndex) -> None:
        id1 = await memory.remember("Same topic", "content 1")
        # Same topic name should produce the same base id
        id2 = await memory.remember("Same topic", "content 2")
        # But the second call should still work (overwrites)
        assert id1 == id2

    @pytest.mark.asyncio
    async def test_remember_with_tags(self, memory: MemoryIndex) -> None:
        await memory.remember(
            "DB config", "PostgreSQL on port 5432", tags=["infra", "database"]
        )
        entries = memory.entries
        assert len(entries) == 1
        assert entries[0].tags == ["infra", "database"]

    @pytest.mark.asyncio
    async def test_remember_creates_topic_file(self, memory: MemoryIndex, tmp_path: object) -> None:
        topic_id = await memory.remember("Test topic", "Full content here")
        import os

        topic_path = os.path.join(str(tmp_path), "topics", f"{topic_id}.md")
        assert os.path.exists(topic_path)

    # ── recall ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_recall_existing(self, memory: MemoryIndex) -> None:
        topic_id = await memory.remember("Auth", "JWT tokens with 15-min expiry")
        content = await memory.recall(topic_id)
        assert content is not None
        assert "JWT tokens" in content

    @pytest.mark.asyncio
    async def test_recall_nonexistent(self, memory: MemoryIndex) -> None:
        result = await memory.recall("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_remember_then_recall_round_trip(self, memory: MemoryIndex) -> None:
        topic_id = await memory.remember(
            "API endpoints",
            "GET /users, POST /users, DELETE /users/:id",
            tags=["api"],
        )
        content = await memory.recall(topic_id)
        assert "GET /users" in content
        assert "DELETE /users/:id" in content

    # ── search ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_search_by_query(self, memory: MemoryIndex) -> None:
        await memory.remember("Auth system", "JWT tokens")
        await memory.remember("Database config", "PostgreSQL")
        await memory.remember("Auth middleware", "Express middleware for auth")

        results = await memory.search("Auth")
        assert len(results) == 2
        topics = [r.topic for r in results]
        assert "Auth system" in topics
        assert "Auth middleware" in topics

    @pytest.mark.asyncio
    async def test_search_by_tags(self, memory: MemoryIndex) -> None:
        await memory.remember("Auth", "JWT", tags=["security"])
        await memory.remember("DB", "Postgres", tags=["infra"])
        await memory.remember("API key", "rotate monthly", tags=["security"])

        results = await memory.search(tags=["security"])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_no_results(self, memory: MemoryIndex) -> None:
        await memory.remember("Auth", "JWT")
        results = await memory.search("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_index(self, memory: MemoryIndex) -> None:
        results = await memory.search("anything")
        assert results == []

    # ── forget ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_forget_existing(self, memory: MemoryIndex) -> None:
        topic_id = await memory.remember("Temp note", "delete me")
        assert await memory.forget(topic_id) is True
        assert await memory.recall(topic_id) is None
        assert len(memory.entries) == 0

    @pytest.mark.asyncio
    async def test_forget_nonexistent(self, memory: MemoryIndex) -> None:
        assert await memory.forget("nonexistent") is False

    @pytest.mark.asyncio
    async def test_forget_removes_topic_file(self, memory: MemoryIndex, tmp_path: object) -> None:
        import os

        topic_id = await memory.remember("Temp", "content")
        topic_path = os.path.join(str(tmp_path), "topics", f"{topic_id}.md")
        assert os.path.exists(topic_path)

        await memory.forget(topic_id)
        assert not os.path.exists(topic_path)

    # ── persistence ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_index_persists_to_disk(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        await mem1.remember("Persistent topic", "This should survive")

        # Create a new instance pointing to the same workspace
        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()
        assert len(mem2.entries) == 1
        assert mem2.entries[0].topic == "Persistent topic"

    @pytest.mark.asyncio
    async def test_recall_after_reload(self, tmp_path: object) -> None:
        mem1 = MemoryIndex(workspace=str(tmp_path))
        topic_id = await mem1.remember("Reload test", "Content survives reload")

        mem2 = MemoryIndex(workspace=str(tmp_path))
        await mem2.load()
        content = await mem2.recall(topic_id)
        assert content is not None
        assert "Content survives reload" in content

    # ── get_context_payload ─────────────────────────────

    @pytest.mark.asyncio
    async def test_context_payload_empty(self, memory: MemoryIndex) -> None:
        msg = memory.get_context_payload()
        assert msg.role == "system"
        assert "memory index" in msg.content.lower()

    @pytest.mark.asyncio
    async def test_context_payload_with_entries(self, memory: MemoryIndex) -> None:
        await memory.remember("Auth", "JWT tokens", tags=["security"])
        await memory.remember("DB", "PostgreSQL config", tags=["infra"])

        msg = memory.get_context_payload()
        assert msg.role == "system"
        assert "Auth" in msg.content
        assert "DB" in msg.content
        # Each pointer should be compact
        lines = [line for line in msg.content.split("\n") if line.startswith("- ")]
        assert len(lines) == 2

    # ── entries property ────────────────────────────────

    @pytest.mark.asyncio
    async def test_entries_property(self, memory: MemoryIndex) -> None:
        assert memory.entries == []
        await memory.remember("Topic 1", "Content 1")
        await memory.remember("Topic 2", "Content 2")
        assert len(memory.entries) == 2
        assert all(isinstance(e, MemoryPointer) for e in memory.entries)

    # ── edge cases ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_remember_overwrites_same_topic(self, memory: MemoryIndex) -> None:
        await memory.remember("Config", "v1 content")
        await memory.remember("Config", "v2 content")
        assert len(memory.entries) == 1
        topic_id = memory.entries[0].topic_id
        content = await memory.recall(topic_id)
        assert "v2 content" in content

    @pytest.mark.asyncio
    async def test_multiple_topics(self, memory: MemoryIndex) -> None:
        ids = []
        for i in range(5):
            tid = await memory.remember(f"Topic {i}", f"Content {i}")
            ids.append(tid)
        assert len(memory.entries) == 5
        # All IDs should be unique
        assert len(set(ids)) == 5
