"""Tests for agentis.agents.worktree — WorktreeAgent."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentis.agents.isolation import TempDirIsolation
from agentis.agents.worktree import WorktreeAgent
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import AgentRuntime
from agentis.types import ProviderResponse, TokenUsage


class MockProvider:
    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return ProviderResponse(
            content="Worktree result.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, messages, **kw) -> AsyncIterator[str]:
        yield "done"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="mock")


class TestWorktreeAgent:
    @pytest.mark.asyncio
    async def test_run_with_temp_dir_isolation(self) -> None:
        rt = AgentRuntime(provider=MockProvider())
        iso = TempDirIsolation()
        agent = WorktreeAgent(name="refactor", runtime=rt, isolation=iso)

        result = await agent.run("Refactor auth module")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_isolation_setup_and_cleanup(self) -> None:
        rt = AgentRuntime(provider=MockProvider())
        iso = TempDirIsolation()
        agent = WorktreeAgent(name="test-agent", runtime=rt, isolation=iso)

        await agent.run("Do work")
        # After run, isolation should be cleaned up
        assert not iso.working_dir().exists()

    @pytest.mark.asyncio
    async def test_cleanup_called_on_error(self) -> None:
        """Cleanup should run even if the task fails."""

        class FailProvider:
            async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
                raise RuntimeError("provider crashed")

            async def stream(self, messages, **kw) -> AsyncIterator[str]:
                yield ""

            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(name="fail")

        rt = AgentRuntime(provider=FailProvider())
        iso = TempDirIsolation()
        agent = WorktreeAgent(name="crasher", runtime=rt, isolation=iso)

        # Should not raise — error handled internally
        result = await agent.run("Do something risky")
        assert "error" in result.lower() or len(result) > 0
        # Isolation should still be cleaned up
        assert not iso.working_dir().exists()

    @pytest.mark.asyncio
    async def test_name_stored(self) -> None:
        rt = AgentRuntime(provider=MockProvider())
        iso = TempDirIsolation()
        agent = WorktreeAgent(name="my-agent", runtime=rt, isolation=iso)
        assert agent.name == "my-agent"
