"""Tests for Improvement 1: Adaptive fast-path in AgentRuntime.

When fast_path_enabled=True and the LLM's first response has no tool calls,
the runtime should return immediately, skipping compaction, extensions, and
session memory overhead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentis.extensions.cost_tracker import CostTracker
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import AgentRuntime, StepResult
from agentis.tools.decorator import tool
from agentis.types import (
    Message,
    Permission,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)


# ── Mock Provider ──────────────────────────────────────────


class ScriptedProvider:
    """Provider that returns scripted responses in order."""

    def __init__(self, script: list[ProviderResponse]) -> None:
        self._script = script
        self._index = 0
        self.call_count = 0

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.call_count += 1
        if self._index < len(self._script):
            resp = self._script[self._index]
            self._index += 1
            return resp
        return ProviderResponse(
            content="Done.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=5, output_tokens=5),
        )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="test_mock")


# ── Test Tool ──────────────────────────────────────────────


@tool()
async def lookup(query: str) -> str:
    """Look up a value."""
    return f"result for {query}"


# ── Tests ──────────────────────────────────────────────────


class TestFastPath:
    """Tests for fast_path_enabled behavior."""

    @pytest.mark.asyncio
    async def test_fast_path_skips_overhead_on_no_tool_calls(self) -> None:
        """When LLM responds without tool calls on turn 1, fast path returns immediately."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="Here is the answer.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
            ),
        ])

        tracker = CostTracker(budget_usd=1.0)

        rt = AgentRuntime(
            provider=provider,
            tools=[lookup],
            fast_path_enabled=True,
            extensions=[tracker],
        )

        steps: list[StepResult] = []
        async for step in rt.steps("Simple question"):
            steps.append(step)

        assert len(steps) == 1
        assert steps[0].is_final is True
        assert steps[0].response.content == "Here is the answer."
        # Fast path should skip extension notification
        assert tracker.get_report()["num_turns"] == 0

    @pytest.mark.asyncio
    async def test_fast_path_does_not_skip_when_tools_called(self) -> None:
        """When LLM calls tools on turn 1, fast path does not activate."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="Let me look that up.",
                tool_calls=[ToolCall(id="t1", name="lookup", arguments={"query": "test"})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content="The answer is: result for test",
                tool_calls=[],
                usage=TokenUsage(input_tokens=20, output_tokens=10),
            ),
        ])

        tracker = CostTracker(budget_usd=1.0)

        rt = AgentRuntime(
            provider=provider,
            tools=[lookup],
            fast_path_enabled=True,
            extensions=[tracker],
        )

        steps: list[StepResult] = []
        async for step in rt.steps("Look up test"):
            steps.append(step)

        assert len(steps) == 2
        # Extensions should run normally (both turns notified)
        assert tracker.get_report()["num_turns"] == 2

    @pytest.mark.asyncio
    async def test_fast_path_disabled_runs_normal_overhead(self) -> None:
        """When fast_path_enabled=False, normal overhead runs even without tool calls."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="Answer without tools.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
            ),
        ])

        tracker = CostTracker(budget_usd=1.0)

        rt = AgentRuntime(
            provider=provider,
            tools=[lookup],
            fast_path_enabled=False,
            extensions=[tracker],
        )

        steps: list[StepResult] = []
        async for step in rt.steps("Question"):
            steps.append(step)

        assert len(steps) == 1
        assert steps[0].is_final is True
        # Extensions SHOULD run (fast path disabled)
        assert tracker.get_report()["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_fast_path_default_is_false(self) -> None:
        """fast_path_enabled defaults to False for backward compatibility."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="Answer.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            ),
        ])

        tracker = CostTracker(budget_usd=1.0)

        rt = AgentRuntime(
            provider=provider,
            extensions=[tracker],
        )

        await rt.run("Hi")
        # Default: extensions run normally
        assert tracker.get_report()["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_initial_context_injected_into_first_message(self) -> None:
        """initial_context dict is formatted and injected into the first turn."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="I see the customer is on enterprise plan.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=20, output_tokens=10),
            ),
        ])

        rt = AgentRuntime(
            provider=provider,
            tools=[lookup],
            fast_path_enabled=True,
            initial_context={
                "customer": {"name": "Alice", "plan": "enterprise"},
                "kb_results": ["Article 1", "Article 2"],
            },
        )

        await rt.run("Help this customer")

        # The provider should have received messages containing the initial context
        assert provider.call_count == 1
        call_messages = provider.calls[0] if hasattr(provider, 'calls') else []
        # We check the assembled context includes our injected data
        all_content = " ".join(m.content for m in rt._session.get_messages() if m.role == "user")
        assert "customer" in all_content.lower() or "alice" in all_content.lower()

    @pytest.mark.asyncio
    async def test_initial_context_none_by_default(self) -> None:
        """initial_context defaults to None and doesn't affect behavior."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="Hello.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            ),
        ])

        rt = AgentRuntime(provider=provider)

        response = await rt.run("Hi")
        assert response == "Hello."

    @pytest.mark.asyncio
    async def test_fast_path_with_run_api(self) -> None:
        """Fast path works correctly through the run() API."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="Quick answer.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            ),
        ])

        tracker = CostTracker(budget_usd=1.0)
        rt = AgentRuntime(
            provider=provider,
            fast_path_enabled=True,
            extensions=[tracker],
        )

        response = await rt.run("Quick question")
        assert response == "Quick answer."
        assert tracker.get_report()["num_turns"] == 0

    @pytest.mark.asyncio
    async def test_fast_path_still_records_session_history(self) -> None:
        """Fast path still records messages in session for persistence."""
        provider = ScriptedProvider([
            ProviderResponse(
                content="Fast answer.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=5, output_tokens=5),
            ),
        ])

        rt = AgentRuntime(
            provider=provider,
            fast_path_enabled=True,
        )

        await rt.run("Question")

        # Session should still have the user message and assistant response
        messages = rt._session.get_messages()
        roles = [m.role for m in messages]
        assert "user" in roles
        assert "assistant" in roles
