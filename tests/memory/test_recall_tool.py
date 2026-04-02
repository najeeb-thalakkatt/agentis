"""Tests for agentis.memory.recall_tool — RecallTool."""

from __future__ import annotations

import pytest

from agentis.memory.index import MemoryIndex
from agentis.memory.recall_tool import RecallTool
from agentis.types import Permission


class TestRecallTool:
    @pytest.fixture
    def memory(self, tmp_path: object) -> MemoryIndex:
        return MemoryIndex(workspace=str(tmp_path))

    @pytest.fixture
    def tool(self, memory: MemoryIndex) -> RecallTool:
        return RecallTool(memory=memory)

    def test_name(self, tool: RecallTool) -> None:
        assert tool.name == "recall"

    def test_permission_is_read_only(self, tool: RecallTool) -> None:
        assert tool.permission == Permission.READ_ONLY

    def test_description(self, tool: RecallTool) -> None:
        assert tool.description
        assert len(tool.description) > 10

    def test_get_schema(self, tool: RecallTool) -> None:
        schema = tool.get_schema()
        assert schema.name == "recall"
        assert "topic_id" in str(schema.parameters)

    @pytest.mark.asyncio
    async def test_execute_existing_topic(
        self, tool: RecallTool, memory: MemoryIndex
    ) -> None:
        topic_id = await memory.remember("Auth", "JWT with 15-min expiry")
        result = await tool.execute(topic_id=topic_id)
        assert result.success is True
        assert "JWT" in result.data

    @pytest.mark.asyncio
    async def test_execute_nonexistent_topic(self, tool: RecallTool) -> None:
        result = await tool.execute(topic_id="nonexistent")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_returns_summary(
        self, tool: RecallTool, memory: MemoryIndex
    ) -> None:
        topic_id = await memory.remember("Config", "Database connection string")
        result = await tool.execute(topic_id=topic_id)
        assert result.summary  # Should have a non-empty summary
