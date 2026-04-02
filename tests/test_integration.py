"""Integration tests — end-to-end runtime with tools, hooks, memory, compaction."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentis.compaction.compactor import ContextCompactor
from agentis.hooks.registry import HookRegistry
from agentis.memory.index import MemoryIndex
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
    ToolResult,
)


# ── Mock Provider ───────────────────────────────────────────


class IntegrationMockProvider:
    """Provider that simulates a multi-turn tool-using conversation."""

    def __init__(self, script: list[ProviderResponse]) -> None:
        self._script = script
        self._index = 0
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.calls.append(messages)
        if self._index < len(self._script):
            resp = self._script[self._index]
            self._index += 1
            return resp
        return ProviderResponse(
            content="Script exhausted.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=5, output_tokens=5),
        )

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="integration_mock")


# ── Test Tools ──────────────────────────────────────────────


@tool()
async def search_db(query: str) -> list[dict[str, Any]]:
    """Search the database for records."""
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


@tool(permission=Permission.MUTATING)
async def update_record(record_id: int, name: str) -> bool:
    """Update a database record."""
    return True


@tool(permission=Permission.DANGEROUS)
async def drop_table(table: str) -> str:
    """Drop a database table."""
    return f"Dropped {table}"


# ── Integration Tests ───────────────────────────────────────


class TestFullAgentLoop:
    """Test the complete agentic loop: context → LLM → tools → memory → respond."""

    @pytest.mark.asyncio
    async def test_multi_turn_tool_conversation(self) -> None:
        """Agent searches DB, then updates a record, then responds."""
        provider = IntegrationMockProvider([
            # Turn 1: search
            ProviderResponse(
                content="Let me search.",
                tool_calls=[ToolCall(id="t1", name="search_db", arguments={"query": "Alice"})],
                usage=TokenUsage(input_tokens=20, output_tokens=15),
                stop_reason="tool_use",
            ),
            # Turn 2: update based on search results
            ProviderResponse(
                content="Found Alice, updating.",
                tool_calls=[ToolCall(id="t2", name="update_record", arguments={"record_id": 1, "name": "Alice B."})],
                usage=TokenUsage(input_tokens=40, output_tokens=20),
                stop_reason="tool_use",
            ),
            # Turn 3: final response
            ProviderResponse(
                content="Updated Alice's record successfully.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=50, output_tokens=10),
                stop_reason="end_turn",
            ),
        ])

        rt = AgentRuntime(
            provider=provider,
            tools=[search_db, update_record],
            system_prompt="You are a database assistant.",
        )

        response = await rt.run("Update Alice's last name to B.")
        assert "Updated" in response
        assert rt._session.turn_count == 3

    @pytest.mark.asyncio
    async def test_hooks_block_dangerous_tool(self) -> None:
        """Safety hook blocks a dangerous tool, error goes to LLM."""
        def block_dangerous(ctx: HookContext) -> HookResponse:
            if ctx.metadata.get("permission") == "dangerous":
                return HookResponse(action=HookAction.DENY, reason="Dangerous ops blocked")
            return HookResponse(action=HookAction.ALLOW)

        hooks = HookRegistry()
        hooks.register(LifecycleEvent.PRE_TOOL_USE, block_dangerous, name="safety")

        provider = IntegrationMockProvider([
            # LLM tries to drop table
            ProviderResponse(
                content="Dropping table.",
                tool_calls=[ToolCall(id="t1", name="drop_table", arguments={"table": "users"})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            ),
            # LLM responds after seeing denial
            ProviderResponse(
                content="I cannot drop the table — it was blocked by safety policy.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=20, output_tokens=15),
            ),
        ])

        rt = AgentRuntime(
            provider=provider,
            tools=[drop_table],
            hooks=hooks,
        )

        response = await rt.run("Drop the users table")
        assert "cannot" in response.lower() or "blocked" in response.lower()

    @pytest.mark.asyncio
    async def test_memory_integration(self, tmp_path: object) -> None:
        """Agent uses memory index in context and recall tool works."""
        memory = MemoryIndex(workspace=str(tmp_path))
        await memory.remember("API docs", "REST API at /api/v1/users", tags=["api"])

        provider = IntegrationMockProvider([
            # LLM calls recall
            ProviderResponse(
                content="Let me check the docs.",
                tool_calls=[ToolCall(
                    id="t1", name="recall",
                    arguments={"topic_id": memory.entries[0].topic_id},
                )],
                usage=TokenUsage(input_tokens=30, output_tokens=10),
                stop_reason="tool_use",
            ),
            # LLM final response
            ProviderResponse(
                content="The API is at /api/v1/users.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=40, output_tokens=10),
            ),
        ])

        rt = AgentRuntime(provider=provider, memory=memory)

        response = await rt.run("Where is the API?")
        assert "/api/v1/users" in response

    @pytest.mark.asyncio
    async def test_steps_api_yields_progress(self) -> None:
        """steps() async iterator yields each step for UI rendering."""
        provider = IntegrationMockProvider([
            ProviderResponse(
                content="Searching.",
                tool_calls=[ToolCall(id="t1", name="search_db", arguments={"query": "all"})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content="Found 2 records.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=20, output_tokens=10),
            ),
        ])

        rt = AgentRuntime(provider=provider, tools=[search_db])

        steps: list[StepResult] = []
        async for step in rt.steps("Show all records"):
            steps.append(step)

        assert len(steps) == 2
        assert steps[0].is_final is False
        assert steps[0].tool_results[0].success is True
        assert steps[1].is_final is True

    @pytest.mark.asyncio
    async def test_session_persistence_across_runs(self) -> None:
        """Multiple run() calls share session state."""
        provider = IntegrationMockProvider([
            ProviderResponse(content="Hello!", tool_calls=[], usage=TokenUsage(input_tokens=5, output_tokens=5)),
            ProviderResponse(content="Your name is Alice.", tool_calls=[], usage=TokenUsage(input_tokens=10, output_tokens=5)),
        ])

        rt = AgentRuntime(provider=provider)

        await rt.run("My name is Alice.")
        await rt.run("What is my name?")

        # Second call should have full history in provider messages
        second_call_messages = provider.calls[1]
        user_messages = [m for m in second_call_messages if m.role == "user"]
        assert len(user_messages) == 2  # Both user messages present

    @pytest.mark.asyncio
    async def test_fork_creates_independent_branch(self) -> None:
        """fork() creates an independent runtime with shared history."""
        provider = IntegrationMockProvider([
            ProviderResponse(content="Noted.", tool_calls=[], usage=TokenUsage(input_tokens=5, output_tokens=5)),
        ])

        rt = AgentRuntime(provider=provider)
        await rt.run("Remember: project uses PostgreSQL")

        forked = rt.fork()
        assert len(forked._session.get_messages()) == len(rt._session.get_messages())
        assert forked._session.session_id != rt._session.session_id

    @pytest.mark.asyncio
    async def test_approval_callback_integration(self) -> None:
        """DANGEROUS tools go through approval callback."""
        approval_log: list[str] = []

        async def track_approval(req) -> bool:
            approval_log.append(req.tool_name)
            return False  # Deny everything

        provider = IntegrationMockProvider([
            ProviderResponse(
                content="Dropping.",
                tool_calls=[ToolCall(id="t1", name="drop_table", arguments={"table": "x"})],
                usage=TokenUsage(input_tokens=10, output_tokens=10),
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content="Operation denied.",
                tool_calls=[],
                usage=TokenUsage(input_tokens=15, output_tokens=10),
            ),
        ])

        rt = AgentRuntime(
            provider=provider,
            tools=[drop_table],
            approval_callback=track_approval,
        )

        await rt.run("Drop table x")
        assert "drop_table" in approval_log
