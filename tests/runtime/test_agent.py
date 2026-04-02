"""Tests for agentis.runtime.agent — AgentRuntime."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agentis.hooks.registry import HookRegistry
from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.runtime.agent import AgentRuntime, StepResult
from agentis.tools.decorator import tool
from agentis.types import (
    HookAction,
    HookContext,
    HookResponse,
    LifecycleEvent,
    Message,
    Permission,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)

# ── Mock Provider ───────────────────────────────────────────


class MockProvider:
    """A mock LLM provider for testing the runtime loop."""

    def __init__(self, responses: list[ProviderResponse] | None = None) -> None:
        self._responses = responses or []
        self._call_index = 0
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        self.calls.append(messages)
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return ProviderResponse(
            content="Done.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "streamed "
        yield "response"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="mock", max_context_tokens=128_000)


# ── Test Tools ──────────────────────────────────────────────


@tool()
async def mock_grep(pattern: str) -> list[str]:
    """Search for a pattern."""
    return [f"match: {pattern}"]


@tool(permission=Permission.MUTATING)
async def mock_write(path: str, content: str) -> bool:
    """Write to a file."""
    return True


# ── Tests ───────────────────────────────────────────────────


class TestAgentRuntimeConstruction:
    def test_minimal_construction(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)
        assert rt._provider is provider
        assert rt._session is not None

    def test_construction_with_tools(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider, tools=[mock_grep, mock_write])
        assert len(rt._orchestrator.list_tools()) == 3  # 2 user tools + recall

    def test_construction_with_system_prompt(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider, system_prompt="You are helpful.")
        assert rt._system_prompt == "You are helpful."

    def test_construction_with_hooks(self) -> None:
        provider = MockProvider()
        hooks = HookRegistry()
        rt = AgentRuntime(provider=provider, hooks=hooks)
        assert rt._hooks is hooks


class TestAgentRuntimeStep:
    @pytest.mark.asyncio
    async def test_step_returns_step_result(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)

        result = await rt.step("Hello")
        assert isinstance(result, StepResult)
        assert result.response is not None
        assert result.is_final is True  # No tool calls
        assert result.turn_number == 1

    @pytest.mark.asyncio
    async def test_step_with_tool_call(self) -> None:
        provider = MockProvider(responses=[
            ProviderResponse(
                content="Let me search.",
                tool_calls=[ToolCall(id="tc1", name="mock_grep", arguments={"pattern": "foo"})],
                usage=TokenUsage(input_tokens=20, output_tokens=15),
                stop_reason="tool_use",
            ),
        ])
        rt = AgentRuntime(provider=provider, tools=[mock_grep])

        result = await rt.step("Search for foo")
        assert result.is_final is False  # Has tool calls, not final
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is True

    @pytest.mark.asyncio
    async def test_step_adds_to_session(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)

        await rt.step("Hello")
        messages = rt._session.get_messages()
        # Should have user message + assistant response
        assert len(messages) >= 2
        assert messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_step_unknown_tool_returns_error_to_llm(self) -> None:
        provider = MockProvider(responses=[
            ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="nonexistent", arguments={})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            ),
        ])
        rt = AgentRuntime(provider=provider, tools=[mock_grep])

        result = await rt.step("Do something")
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False


class TestAgentRuntimeRun:
    @pytest.mark.asyncio
    async def test_run_simple_response(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)

        response = await rt.run("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_run_with_tool_loop(self) -> None:
        """Run should loop: LLM calls tool, gets result, then responds."""
        provider = MockProvider(responses=[
            # Turn 1: LLM requests tool call
            ProviderResponse(
                content="Searching...",
                tool_calls=[ToolCall(id="tc1", name="mock_grep", arguments={"pattern": "bug"})],
                usage=TokenUsage(input_tokens=20, output_tokens=15),
                stop_reason="tool_use",
            ),
            # Turn 2: LLM responds with final answer
            ProviderResponse(
                content="Found the bug in line 42.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=30, output_tokens=20),
                stop_reason="end_turn",
            ),
        ])
        rt = AgentRuntime(provider=provider, tools=[mock_grep])

        response = await rt.run("Find the bug")
        assert "Found the bug" in response
        assert len(provider.calls) == 2  # Two LLM calls

    @pytest.mark.asyncio
    async def test_run_respects_max_turns(self) -> None:
        """Run should stop after max_turns even if LLM keeps calling tools."""
        # Provider always wants to call tools
        infinite_tool_responses = [
            ProviderResponse(
                content="",
                tool_calls=[ToolCall(id=f"tc{i}", name="mock_grep", arguments={"pattern": "x"})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            )
            for i in range(100)
        ]
        provider = MockProvider(responses=infinite_tool_responses)
        rt = AgentRuntime(provider=provider, tools=[mock_grep], max_turns=3)

        await rt.run("infinite loop")
        assert len(provider.calls) <= 3

    @pytest.mark.asyncio
    async def test_run_updates_session(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)

        await rt.run("Hello")
        assert rt._session.turn_count >= 1
        assert len(rt._session.get_messages()) >= 2


class TestAgentRuntimeSteps:
    @pytest.mark.asyncio
    async def test_steps_yields_step_results(self) -> None:
        provider = MockProvider(responses=[
            ProviderResponse(
                content="Searching.",
                tool_calls=[ToolCall(id="tc1", name="mock_grep", arguments={"pattern": "x"})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content="Done.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        ])
        rt = AgentRuntime(provider=provider, tools=[mock_grep])

        results = []
        async for step_result in rt.steps("search"):
            results.append(step_result)

        assert len(results) == 2
        assert results[0].is_final is False
        assert results[1].is_final is True


class TestAgentRuntimeHooks:
    @pytest.mark.asyncio
    async def test_hook_deny_blocks_tool(self) -> None:
        def block_all(ctx: HookContext) -> HookResponse:
            return HookResponse(action=HookAction.DENY, reason="blocked")

        hooks = HookRegistry()
        hooks.register(LifecycleEvent.PRE_TOOL_USE, block_all, name="blocker")

        provider = MockProvider(responses=[
            ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="mock_grep", arguments={"pattern": "x"})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            ),
        ])
        rt = AgentRuntime(provider=provider, tools=[mock_grep], hooks=hooks)

        result = await rt.step("search")
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "blocked" in (result.tool_results[0].error or "")


class TestAgentRuntimeSessionManagement:
    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)
        await rt.run("Hello")
        assert len(rt._session.get_messages()) > 0

        rt.reset()
        assert len(rt._session.get_messages()) == 0

    @pytest.mark.asyncio
    async def test_fork(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)
        await rt.run("Hello")

        forked = rt.fork()
        assert forked._session.session_id != rt._session.session_id
        # Forked should have the conversation history
        assert len(forked._session.get_messages()) == len(rt._session.get_messages())

    @pytest.mark.asyncio
    async def test_fork_is_isolated(self) -> None:
        provider = MockProvider()
        rt = AgentRuntime(provider=provider)
        await rt.run("Hello")
        original_count = len(rt._session.get_messages())

        forked = rt.fork()
        await forked.run("Another question")

        # Parent unchanged
        assert len(rt._session.get_messages()) == original_count
