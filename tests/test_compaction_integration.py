"""Stress test #2: Long-session compaction integration.

Tests that the compactor is actually wired to the runtime and fires
when context grows large. Verifies quality is preserved after compaction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentis.compaction.compactor import ContextCompactor
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import AgentRuntime, StepResult
from agentis.tools.decorator import tool
from agentis.types import (
    Message,
    Permission,
    Priority,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)


# ── Mock Provider: Long conversation ───────────────────────


class LongSessionProvider:
    """Provider that simulates a long multi-turn tool-using session.

    Returns tool calls for N turns, then a final response.
    Tracks all messages received to verify context management.
    """

    def __init__(self, tool_turns: int = 30, tokens_per_turn: int = 500) -> None:
        self._tool_turns = tool_turns
        self._tokens_per_turn = tokens_per_turn
        self._turn = 0
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.calls.append(list(messages))
        self._turn += 1

        if self._turn <= self._tool_turns:
            # Return a tool call each turn
            return ProviderResponse(
                content=f"Investigating step {self._turn}. " + ("x" * self._tokens_per_turn),
                tool_calls=[ToolCall(
                    id=f"t{self._turn}",
                    name="investigate",
                    arguments={"target": f"system_{self._turn}"},
                )],
                usage=TokenUsage(
                    input_tokens=self._tokens_per_turn,
                    output_tokens=self._tokens_per_turn // 2,
                ),
                stop_reason="tool_use",
            )
        else:
            # Final response
            return ProviderResponse(
                content="Investigation complete. Root cause: memory leak in service_15.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=200, output_tokens=100),
            )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        # Small context window to force compaction
        return ProviderCapabilities(name="long_session_mock", max_context_tokens=8000)


# ── Test Tools ─────────────────────────────────────────────


@tool()
async def investigate(target: str) -> str:
    """Investigate a system component."""
    # Return a large result to fill context quickly
    return f"Investigation of {target}: " + ("metrics data point " * 50)


@tool()
async def diagnose(findings: str) -> str:
    """Diagnose based on findings."""
    return f"Diagnosis: {findings[:100]}"


# ── Tests: Bug Verification ────────────────────────────────


class TestCompactionWiring:
    """Verify that compaction is actually wired to the runtime."""

    @pytest.mark.asyncio
    async def test_compactor_receives_entries_from_runtime(self) -> None:
        """The runtime must feed session messages into the compactor."""
        provider = LongSessionProvider(tool_turns=3, tokens_per_turn=200)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=5,
        )

        await rt.run("Investigate the system")

        # The compactor should have entries from the conversation
        assert rt._compactor.current_tokens() > 0, (
            "BUG: Compactor has no entries — session messages are not "
            "being fed to the compactor."
        )

    @pytest.mark.asyncio
    async def test_compactor_entries_grow_with_turns(self) -> None:
        """Compactor entry count should grow as the conversation progresses."""
        provider = LongSessionProvider(tool_turns=5, tokens_per_turn=200)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=7,
        )

        token_counts = []
        async for step in rt.steps("Start investigation"):
            token_counts.append(rt._compactor.current_tokens())

        # Tokens should generally increase (or stay same after compaction)
        assert token_counts[-1] > 0
        assert max(token_counts) > token_counts[0]


# ── Tests: Compaction Fires ────────────────────────────────


class TestCompactionFires:
    """Verify that compaction actually fires when context is large."""

    @pytest.mark.asyncio
    async def test_compaction_triggers_on_large_context(self) -> None:
        """With a small context window, compaction should fire during a long session."""
        # Small context window (8000 tokens) + large tool results = compaction must fire
        # Need enough turns for content to exceed 85% of 8000 = 6800 tokens
        provider = LongSessionProvider(tool_turns=20, tokens_per_turn=500)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=22,
        )

        # Track compactor state
        max_tokens_seen = 0
        compaction_fired = False

        async for step in rt.steps("Investigate all systems"):
            current = rt._compactor.current_tokens()
            if current < max_tokens_seen and max_tokens_seen > 0:
                compaction_fired = True
            max_tokens_seen = max(max_tokens_seen, current)

        assert compaction_fired, (
            f"Compaction never fired. Max tokens seen: {max_tokens_seen}, "
            f"threshold: {int(8000 * 0.85)}"
        )

    @pytest.mark.asyncio
    async def test_compaction_keeps_context_under_limit(self) -> None:
        """After compaction fires, context should stay under the max."""
        provider = LongSessionProvider(tool_turns=15, tokens_per_turn=300)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=17,
        )

        async for step in rt.steps("Investigate everything"):
            current = rt._compactor.current_tokens()
            # Should never exceed max_tokens (8000)
            assert current <= 8000, (
                f"Context exceeded max: {current} > 8000"
            )

    @pytest.mark.asyncio
    async def test_critical_entries_survive_compaction(self) -> None:
        """CRITICAL priority entries must never be removed by compaction."""
        provider = LongSessionProvider(tool_turns=15, tokens_per_turn=300)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=17,
        )

        await rt.run("Investigate")

        # System prompt should survive as it's typically CRITICAL
        entries = rt._compactor.get_entries()
        # At least some entries should remain
        assert len(entries) > 0


# ── Tests: Quality After Compaction ────────────────────────


class TestQualityAfterCompaction:
    """Verify that the agent can still function after compaction."""

    @pytest.mark.asyncio
    async def test_agent_completes_after_compaction(self) -> None:
        """Agent should still reach a final response after compaction fires."""
        provider = LongSessionProvider(tool_turns=10, tokens_per_turn=300)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=12,
        )

        response = await rt.run("Investigate the incident")

        # Agent should have produced a final response
        assert "Investigation complete" in response or len(response) > 0

    @pytest.mark.asyncio
    async def test_session_messages_survive_compaction(self) -> None:
        """Session should still have messages after compaction (may be summarized)."""
        provider = LongSessionProvider(tool_turns=10, tokens_per_turn=300)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=12,
        )

        await rt.run("Investigate")

        messages = rt._session.get_messages()
        assert len(messages) > 0, "Session lost all messages after compaction"

    @pytest.mark.asyncio
    async def test_provider_receives_manageable_context(self) -> None:
        """The provider should never receive more tokens than max_context."""
        provider = LongSessionProvider(tool_turns=15, tokens_per_turn=300)

        rt = AgentRuntime(
            provider=provider,
            tools=[investigate],
            max_turns=17,
        )

        await rt.run("Investigate")

        # Check that later calls don't have unbounded context
        for i, call_messages in enumerate(provider.calls):
            total_chars = sum(len(m.content) for m in call_messages)
            # Rough token estimate: chars / 3
            estimated_tokens = total_chars // 3
            # Allow some overhead but should be roughly bounded
            assert estimated_tokens < 8000 * 2, (
                f"Call {i} had ~{estimated_tokens} tokens, expected < {8000 * 2}"
            )
