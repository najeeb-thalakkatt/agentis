"""Tests for agentis.types — core dataclasses and enums."""

from __future__ import annotations

import time

from agentis.types import (
    ApprovalRequest,
    ContextEntry,
    CostRecord,
    HookAction,
    HookContext,
    HookResponse,
    LifecycleEvent,
    Message,
    Permission,
    Priority,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolResult,
)

# ── Enum tests ──────────────────────────────────────────────


class TestPriority:
    def test_ordering(self) -> None:
        assert (
            Priority.CRITICAL < Priority.HIGH < Priority.MEDIUM
            < Priority.LOW < Priority.EPHEMERAL
        )

    def test_values(self) -> None:
        assert Priority.CRITICAL == 0
        assert Priority.EPHEMERAL == 4

    def test_sortable(self) -> None:
        items = [Priority.LOW, Priority.CRITICAL, Priority.MEDIUM]
        assert sorted(items) == [Priority.CRITICAL, Priority.MEDIUM, Priority.LOW]


class TestPermission:
    def test_values(self) -> None:
        assert Permission.READ_ONLY.value == "read_only"
        assert Permission.MUTATING.value == "mutating"
        assert Permission.DANGEROUS.value == "dangerous"

    def test_all_members(self) -> None:
        assert len(Permission) == 3


class TestHookAction:
    def test_values(self) -> None:
        assert HookAction.ALLOW.value == "allow"
        assert HookAction.DENY.value == "deny"
        assert HookAction.ASK_HUMAN.value == "ask_human"

    def test_all_members(self) -> None:
        assert len(HookAction) == 3


class TestLifecycleEvent:
    def test_has_minimal_set(self) -> None:
        expected = {
            "SESSION_START",
            "SESSION_END",
            "ON_TURN_START",
            "ON_TURN_END",
            "PRE_TOOL_USE",
            "POST_TOOL_USE",
            "PRE_LLM_CALL",
            "POST_LLM_CALL",
            "PRE_COMPACT",
            "ON_ERROR",
        }
        actual = {e.name for e in LifecycleEvent}
        assert expected == actual

    def test_count(self) -> None:
        assert len(LifecycleEvent) == 10


# ── Dataclass tests ─────────────────────────────────────────


class TestMessage:
    def test_minimal_construction(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.tool_results is None
        assert msg.metadata == {}

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", name="grep", arguments={"pattern": "foo"})
        msg = Message(role="assistant", content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "grep"

    def test_with_metadata(self) -> None:
        msg = Message(role="system", content="prompt", metadata={"source": "test"})
        assert msg.metadata["source"] == "test"


class TestToolCall:
    def test_construction(self) -> None:
        tc = ToolCall(id="abc", name="file_read", arguments={"path": "/tmp/x"})
        assert tc.id == "abc"
        assert tc.name == "file_read"
        assert tc.arguments == {"path": "/tmp/x"}


class TestToolResult:
    def test_success(self) -> None:
        tr = ToolResult(
            success=True,
            data={"matches": 5},
            summary="Found 5 matches",
            full_output="",
            tokens_used=20,
        )
        assert tr.success is True
        assert tr.error is None

    def test_failure(self) -> None:
        tr = ToolResult(
            success=False,
            data=None,
            summary="File not found",
            full_output="",
            tokens_used=5,
            error="FileNotFoundError: /tmp/missing.py",
        )
        assert tr.success is False
        assert "FileNotFoundError" in tr.error

    def test_defaults(self) -> None:
        tr = ToolResult(success=True, data=None, summary="ok", full_output="", tokens_used=0)
        assert tr.error is None


class TestContextEntry:
    def test_construction(self) -> None:
        now = time.time()
        entry = ContextEntry(
            content="some tool output",
            priority=Priority.MEDIUM,
            token_count=50,
            timestamp=now,
            entry_type="tool_result",
        )
        assert entry.priority == Priority.MEDIUM
        assert entry.entry_type == "tool_result"
        assert entry.timestamp == now


class TestApprovalRequest:
    def test_construction(self) -> None:
        req = ApprovalRequest(
            tool_name="bash",
            arguments={"command": "rm -rf /tmp/test"},
            reason="Dangerous command detected",
        )
        assert req.tool_name == "bash"
        assert req.reason == "Dangerous command detected"


class TestHookContext:
    def test_construction(self) -> None:
        ctx = HookContext(
            event=LifecycleEvent.PRE_TOOL_USE,
            tool_name="file_edit",
            arguments={"path": "/tmp/x"},
            agent_id="agent-1",
            session_id="sess-1",
        )
        assert ctx.event == LifecycleEvent.PRE_TOOL_USE
        assert ctx.tool_name == "file_edit"
        assert ctx.metadata == {}

    def test_with_metadata(self) -> None:
        ctx = HookContext(
            event=LifecycleEvent.ON_ERROR,
            tool_name=None,
            arguments=None,
            agent_id="a",
            session_id="s",
            metadata={"error_type": "timeout"},
        )
        assert ctx.metadata["error_type"] == "timeout"


class TestHookResponse:
    def test_allow(self) -> None:
        resp = HookResponse(action=HookAction.ALLOW)
        assert resp.reason is None
        assert resp.metadata is None

    def test_deny_with_reason(self) -> None:
        resp = HookResponse(
            action=HookAction.DENY,
            reason="Destructive command blocked",
            metadata={"pattern": "rm -rf"},
        )
        assert resp.action == HookAction.DENY
        assert resp.reason == "Destructive command blocked"
        assert resp.metadata["pattern"] == "rm -rf"


class TestTokenUsage:
    def test_construction(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0

    def test_with_cache(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=80,
            cache_write_tokens=20,
        )
        assert usage.cache_read_tokens == 80


class TestProviderResponse:
    def test_construction(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        tc = ToolCall(id="t1", name="grep", arguments={})
        resp = ProviderResponse(
            content="I'll search for that.",
            tool_calls=[tc],
            usage=usage,
            stop_reason="tool_use",
        )
        assert resp.content == "I'll search for that."
        assert len(resp.tool_calls) == 1
        assert resp.stop_reason == "tool_use"

    def test_no_tool_calls(self) -> None:
        usage = TokenUsage(input_tokens=50, output_tokens=30)
        resp = ProviderResponse(
            content="Done.",
            tool_calls=[],
            usage=usage,
        )
        assert resp.tool_calls == []
        assert resp.stop_reason is None


class TestCostRecord:
    def test_construction(self) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        now = time.time()
        record = CostRecord(
            provider="anthropic",
            model="claude-sonnet-4-6",
            usage=usage,
            estimated_cost_usd=0.015,
            timestamp=now,
        )
        assert record.provider == "anthropic"
        assert record.estimated_cost_usd == 0.015
