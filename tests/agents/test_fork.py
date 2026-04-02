"""Tests for agentis.agents.fork — ForkAgent."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentis.agents.fork import ForkAgent
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import AgentRuntime
from agentis.tools.decorator import tool
from agentis.types import (
    Permission,
    ProviderResponse,
    TokenUsage,
)


class MockProvider:
    def __init__(self, response_text: str = "Fork result.") -> None:
        self._text = response_text

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return ProviderResponse(
            content=self._text,
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, messages, **kw) -> AsyncIterator[str]:
        yield self._text

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="mock")


@tool()
async def read_tool(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


@tool(permission=Permission.MUTATING)
async def write_tool(path: str, content: str) -> bool:
    """Write a file."""
    return True


class TestForkAgent:
    @pytest.mark.asyncio
    async def test_fork_runs_task(self) -> None:
        parent = AgentRuntime(provider=MockProvider(), tools=[read_tool])
        fork = ForkAgent(parent=parent, task="Find the bug in auth.py")
        result = await fork.run()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fork_strips_mutating_tools(self) -> None:
        parent = AgentRuntime(
            provider=MockProvider(),
            tools=[read_tool, write_tool],
        )
        fork = ForkAgent(parent=parent, task="investigate")
        # Fork should only have read-only tools (read_tool + recall)
        fork_tools = fork._runtime._orchestrator.list_tools()
        for t in fork_tools:
            assert t.permission == Permission.READ_ONLY

    @pytest.mark.asyncio
    async def test_fork_inherits_conversation(self) -> None:
        parent = AgentRuntime(provider=MockProvider())
        await parent.run("Context message")

        fork = ForkAgent(parent=parent, task="Use the context")
        # Fork's session should have the parent's messages
        assert len(fork._runtime._session.get_messages()) > 0

    @pytest.mark.asyncio
    async def test_fork_isolated_from_parent(self) -> None:
        parent = AgentRuntime(provider=MockProvider())
        await parent.run("Setup context")
        original_count = len(parent._session.get_messages())

        fork = ForkAgent(parent=parent, task="Do something")
        await fork.run()

        # Parent should be unchanged
        assert len(parent._session.get_messages()) == original_count


class TestParallelInvestigation:
    @pytest.mark.asyncio
    async def test_parallel_investigate(self) -> None:
        parent = AgentRuntime(provider=MockProvider("Found it."))
        tasks = [
            "Check file A",
            "Check file B",
            "Check file C",
        ]
        results = await ForkAgent.parallel_investigate(parent, tasks)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all("Found it." in r for r in results)

    @pytest.mark.asyncio
    async def test_parallel_investigate_empty(self) -> None:
        parent = AgentRuntime(provider=MockProvider())
        results = await ForkAgent.parallel_investigate(parent, [])
        assert results == []
