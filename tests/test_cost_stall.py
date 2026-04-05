"""Tests for Improvement 3: Cost-aware early stopping.

CostTracker should detect when the agent is stalling (making turns
without new tool calls) while consuming budget, and signal the runtime
to inject a "wrap it up" message.
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


class StallProvider:
    """Provider that simulates a stalling agent.

    After the initial tool calls, it makes turns with tool calls
    that don't add new information (repeating the same tool).
    """

    def __init__(self, script: list[ProviderResponse]) -> None:
        self._script = script
        self._index = 0
        self.call_count = 0
        self.last_messages: list[Message] = []

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.call_count += 1
        self.last_messages = messages
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
        return ProviderCapabilities(name="stall_mock")


# ── Test Tools ─────────────────────────────────────────────


@tool()
async def search(query: str) -> str:
    """Search for something."""
    return f"results for {query}"


@tool()
async def read_doc(doc_id: str) -> str:
    """Read a document."""
    return f"contents of {doc_id}"


# ── Tests ──────────────────────────────────────────────────


class TestCostStallDetection:
    """Tests for stall detection in CostTracker."""

    def test_stall_detection_defaults(self) -> None:
        """Stall detection fields have sensible defaults."""
        tracker = CostTracker(budget_usd=1.0)
        assert tracker.should_force_conclude is False

    def test_stall_detection_with_config(self) -> None:
        """Stall detection can be configured."""
        tracker = CostTracker(
            budget_usd=1.0,
            stall_turn_threshold=3,
            stall_budget_fraction=0.5,
        )
        assert tracker._stall_turn_threshold == 3
        assert tracker._stall_budget_fraction == 0.5

    def test_stall_not_triggered_under_budget(self) -> None:
        """No stall signal when under the budget fraction."""
        tracker = CostTracker(
            budget_usd=1.0,
            stall_turn_threshold=2,
            stall_budget_fraction=0.7,
        )
        # Simulate low-cost turns with no new tools
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.1)
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.2)
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.3)
        assert tracker.should_force_conclude is False

    def test_stall_not_triggered_when_new_tools_used(self) -> None:
        """No stall signal when new tools are being called."""
        tracker = CostTracker(
            budget_usd=1.0,
            stall_turn_threshold=2,
            stall_budget_fraction=0.5,
        )
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.6)
        tracker._record_turn_for_stall(tool_names=["read_doc"], cost_so_far=0.7)
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.8)
        assert tracker.should_force_conclude is False

    def test_stall_triggered_when_no_new_tools_and_over_budget_fraction(self) -> None:
        """Stall triggers when turns have no new tools and budget > threshold."""
        tracker = CostTracker(
            budget_usd=1.0,
            stall_turn_threshold=3,
            stall_budget_fraction=0.7,
        )
        # First turns with tools (not stalling yet)
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.2)
        assert tracker.should_force_conclude is False

        # Now stalling turns with no tools, cost rising
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.5)
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.6)
        assert tracker.should_force_conclude is False  # only 2 stall turns

        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.75)
        assert tracker.should_force_conclude is True  # 3 stall turns + over 70%

    def test_stall_triggered_with_repeated_tools(self) -> None:
        """Stall triggers when same tools keep being called without new ones."""
        tracker = CostTracker(
            budget_usd=1.0,
            stall_turn_threshold=3,
            stall_budget_fraction=0.7,
        )
        # All tools have been seen
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.3)
        # Repeating same tool without new ones
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.5)
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.6)
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.8)
        assert tracker.should_force_conclude is True

    def test_stall_resets_when_new_tool_appears(self) -> None:
        """Stall counter resets when a new tool is called."""
        tracker = CostTracker(
            budget_usd=1.0,
            stall_turn_threshold=3,
            stall_budget_fraction=0.7,
        )
        tracker._record_turn_for_stall(tool_names=["search"], cost_so_far=0.3)
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.5)
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=0.6)
        # 2 stall turns, then a new tool resets
        tracker._record_turn_for_stall(tool_names=["read_doc"], cost_so_far=0.7)
        assert tracker.should_force_conclude is False

    def test_stall_disabled_without_budget(self) -> None:
        """Stall detection doesn't trigger when no budget is set."""
        tracker = CostTracker(
            budget_usd=None,
            stall_turn_threshold=2,
            stall_budget_fraction=0.7,
        )
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=100.0)
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=200.0)
        tracker._record_turn_for_stall(tool_names=[], cost_so_far=300.0)
        assert tracker.should_force_conclude is False

    def test_force_conclude_message(self) -> None:
        """force_conclude_message returns appropriate text."""
        tracker = CostTracker(budget_usd=1.0)
        msg = tracker.force_conclude_message
        assert "budget" in msg.lower()

    @pytest.mark.asyncio
    async def test_on_turn_end_updates_stall_tracking(self) -> None:
        """on_turn_end integrates with stall detection via tool_calls in response."""
        tracker = CostTracker(
            budget_usd=0.001,  # tiny budget
            stall_turn_threshold=2,
            stall_budget_fraction=0.5,
        )

        # Simulate a response with tool calls
        response_with_tools = ProviderResponse(
            content="Searching.",
            tool_calls=[ToolCall(id="t1", name="search", arguments={"query": "x"})],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        await tracker.on_turn_end(None, response_with_tools)
        assert tracker.should_force_conclude is False

        # Simulate responses without new tools
        response_no_tools = ProviderResponse(
            content="Thinking.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        await tracker.on_turn_end(None, response_no_tools)
        await tracker.on_turn_end(None, response_no_tools)
        # Over budget (tiny budget) + stall turns >= threshold
        assert tracker.should_force_conclude is True


class TestCostStallIntegration:
    """Integration test: CostTracker stall detection with AgentRuntime."""

    @pytest.mark.asyncio
    async def test_runtime_injects_conclude_message_on_stall(self) -> None:
        """When CostTracker signals stall, runtime injects conclude prompt."""
        # Script layout (stall_turn_threshold=3):
        # idx 0: initial search (turn 1) → stall_turns=0 (new tool)
        # idx 1: repeat search (turn 2, _continue_step) → stall_turns=1
        # idx 2: repeat search (turn 3, _continue_step) → stall_turns=2
        # idx 3: repeat search (turn 4, _continue_step) → stall_turns=3 → triggers!
        # idx 4: conclude response (turn 5, step(conclude_msg)) → final
        script = [
            ProviderResponse(
                content="Searching.",
                tool_calls=[ToolCall(id="t1", name="search", arguments={"query": "test"})],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content="Still thinking 0.",
                tool_calls=[ToolCall(id="t2", name="search", arguments={"query": "test"})],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content="Still thinking 1.",
                tool_calls=[ToolCall(id="t3", name="search", arguments={"query": "test"})],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content="Still thinking 2.",
                tool_calls=[ToolCall(id="t4", name="search", arguments={"query": "test"})],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
                stop_reason="tool_use",
            ),
            # This is what the LLM returns after the conclude message is injected
            ProviderResponse(
                content="Here is my best answer based on what I found.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=100, output_tokens=50),
            ),
        ]

        provider = StallProvider(script)
        tracker = CostTracker(
            budget_usd=0.001,  # tiny budget to trigger quickly
            stall_turn_threshold=3,
            stall_budget_fraction=0.5,
        )

        rt = AgentRuntime(
            provider=provider,
            tools=[search, read_doc],
            extensions=[tracker],
            max_turns=10,
        )

        steps: list[StepResult] = []
        async for step in rt.steps("Research this topic"):
            steps.append(step)

        # The runtime should have concluded (not hit max_turns of 10)
        assert steps[-1].is_final is True
        assert len(steps) == 5  # 4 normal turns + 1 conclude
        assert "best answer" in steps[-1].response.content
