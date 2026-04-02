"""Tests for agentis.extensions.dream — DreamExtension."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentis.extensions.dream import DreamExtension
from agentis.memory.index import MemoryIndex
from agentis.types import ProviderResponse, TokenUsage


class TestDreamExtensionBasics:
    def test_name(self) -> None:
        ext = DreamExtension(memory=MagicMock())
        assert ext.name == "dream"

    def test_construction_defaults(self) -> None:
        ext = DreamExtension(memory=MagicMock())
        assert ext._idle_threshold == 120.0

    def test_custom_idle_threshold(self) -> None:
        ext = DreamExtension(memory=MagicMock(), idle_threshold=60.0)
        assert ext._idle_threshold == 60.0


class TestDreamConsolidation:
    @pytest.mark.asyncio
    async def test_consolidate_with_provider(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        await memory.remember("Auth", "JWT tokens")
        await memory.remember("DB", "PostgreSQL on 5432")

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ProviderResponse(
                content="Auth: JWT tokens. DB: PostgreSQL:5432.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=50, output_tokens=20),
            )
        )

        ext = DreamExtension(
            memory=memory,
            utility_provider=mock_provider,
        )
        result = await ext.consolidate(
            session_logs="User discussed auth and database config.",
            current_memory=memory.get_context_payload().content,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        mock_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidate_without_provider(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        ext = DreamExtension(memory=memory)

        result = await ext.consolidate(
            session_logs="Some logs",
            current_memory="Some memory",
        )
        assert result == ""  # No provider, nothing to consolidate

    @pytest.mark.asyncio
    async def test_consolidate_handles_provider_error(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("API down"))

        ext = DreamExtension(memory=memory, utility_provider=mock_provider)
        result = await ext.consolidate(
            session_logs="logs",
            current_memory="memory",
        )
        assert result == ""  # Graceful failure


class TestDreamExtensionLifecycle:
    @pytest.mark.asyncio
    async def test_on_runtime_start(self) -> None:
        ext = DreamExtension(memory=MagicMock())
        # Should not raise
        await ext.on_runtime_start(MagicMock())

    @pytest.mark.asyncio
    async def test_on_runtime_stop(self) -> None:
        ext = DreamExtension(memory=MagicMock())
        await ext.on_runtime_start(MagicMock())
        # Should not raise, should cancel any background task
        await ext.on_runtime_stop(MagicMock())

    @pytest.mark.asyncio
    async def test_on_turn_end_resets_activity(self) -> None:
        ext = DreamExtension(memory=MagicMock())
        import time

        before = time.time()
        await ext.on_turn_end(MagicMock(), MagicMock())
        assert ext._last_activity >= before

    @pytest.mark.asyncio
    async def test_on_idle_triggers_consolidation(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        await memory.remember("Test", "Content")

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(
            return_value=ProviderResponse(
                content="Consolidated.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        )

        ext = DreamExtension(
            memory=memory,
            utility_provider=mock_provider,
            idle_threshold=1.0,
        )
        # Calling on_idle with enough idle time should trigger consolidation
        await ext.on_idle(MagicMock(), idle_seconds=5.0)
        mock_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_idle_skips_when_not_enough_time(self, tmp_path: object) -> None:
        memory = MemoryIndex(workspace=str(tmp_path))
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock()

        ext = DreamExtension(
            memory=memory,
            utility_provider=mock_provider,
            idle_threshold=120.0,
        )
        await ext.on_idle(MagicMock(), idle_seconds=10.0)
        mock_provider.complete.assert_not_called()


class TestDreamConsolidationPrompt:
    def test_prompt_content(self) -> None:
        ext = DreamExtension(memory=MagicMock())
        assert "MERGE" in ext.CONSOLIDATION_PROMPT or "merge" in ext.CONSOLIDATION_PROMPT.lower()
        assert "contradiction" in ext.CONSOLIDATION_PROMPT.lower()
