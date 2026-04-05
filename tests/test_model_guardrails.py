"""Tests for model-adaptive guardrails.

Weaker models get stricter instructions injected into context.
Stronger models get lighter guardrails.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentis.guardrails import ModelGuardrails
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import AgentRuntime
from agentis.tools.decorator import tool
from agentis.types import (
    Message,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)


# ── Mock Provider ──────────────────────────────────────────


class TierMockProvider:
    def __init__(self, tier: str = "strong") -> None:
        self._tier = tier
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.calls.append(list(messages))
        return ProviderResponse(
            content="Done.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="test", model_tier=self._tier)


@tool()
async def investigate(target: str) -> str:
    """Investigate something."""
    return f"results for {target}"


# ── Tests: ModelGuardrails ─────────────────────────────────


class TestModelGuardrails:
    """Test the ModelGuardrails class."""

    def test_strong_model_gets_minimal_guardrails(self) -> None:
        prompt = ModelGuardrails.get_prompt("strong")
        # Strong models get light or no guardrails
        assert prompt == "" or len(prompt) < 200

    def test_moderate_model_gets_some_guardrails(self) -> None:
        prompt = ModelGuardrails.get_prompt("moderate")
        assert len(prompt) > 0
        assert "tool" in prompt.lower() or "step" in prompt.lower()

    def test_weak_model_gets_strict_guardrails(self) -> None:
        prompt = ModelGuardrails.get_prompt("weak")
        assert len(prompt) > 200
        assert "MUST" in prompt
        assert "remediat" in prompt.lower() or "action" in prompt.lower()

    def test_unknown_tier_returns_empty(self) -> None:
        prompt = ModelGuardrails.get_prompt("unknown")
        assert prompt == ""

    def test_default_tier_is_strong(self) -> None:
        caps = ProviderCapabilities(name="test")
        assert caps.model_tier == "strong"


# ── Tests: Runtime Integration ─────────────────────────────


class TestGuardrailsIntegration:
    """Test that guardrails are injected into the runtime context."""

    @pytest.mark.asyncio
    async def test_weak_model_gets_guardrails_in_context(self) -> None:
        """AgentRuntime injects guardrails for weak models."""
        provider = TierMockProvider(tier="weak")
        rt = AgentRuntime(provider=provider, tools=[investigate])

        await rt.run("Do something")

        # Check that the LLM received guardrail instructions in messages
        messages = provider.calls[0]
        system_messages = [m for m in messages if m.role == "system"]
        all_system_text = " ".join(m.content for m in system_messages)
        assert "MUST" in all_system_text, (
            "Weak model did not receive guardrail instructions"
        )

    @pytest.mark.asyncio
    async def test_strong_model_gets_no_extra_guardrails(self) -> None:
        """AgentRuntime does not inject heavy guardrails for strong models."""
        provider = TierMockProvider(tier="strong")
        rt = AgentRuntime(provider=provider, tools=[investigate])

        await rt.run("Do something")

        messages = provider.calls[0]
        system_messages = [m for m in messages if m.role == "system"]
        all_system_text = " ".join(m.content for m in system_messages)
        # Should NOT contain the weak-model-specific guardrail text
        assert "you MUST call at least one" not in all_system_text

    @pytest.mark.asyncio
    async def test_moderate_model_gets_moderate_guardrails(self) -> None:
        """AgentRuntime injects moderate guardrails."""
        provider = TierMockProvider(tier="moderate")
        rt = AgentRuntime(provider=provider, tools=[investigate])

        await rt.run("Do something")

        messages = provider.calls[0]
        system_messages = [m for m in messages if m.role == "system"]
        all_system_text = " ".join(m.content for m in system_messages)
        assert len(all_system_text) > 50  # Has some content
