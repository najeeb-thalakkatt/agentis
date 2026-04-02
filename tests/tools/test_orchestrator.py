"""Tests for agentis.tools.orchestrator — ToolOrchestrator."""

from __future__ import annotations

import asyncio
import time

import pytest

from agentis.tools.decorator import tool
from agentis.tools.orchestrator import ToolOrchestrator
from agentis.types import ApprovalRequest, Permission, ToolCall, ToolResult


# ── Test tools ──────────────────────────────────────────────


@tool()
async def grep(pattern: str, path: str = ".") -> list[str]:
    """Search for patterns in files."""
    return [f"match: {pattern} in {path}"]


@tool()
async def glob(pattern: str) -> list[str]:
    """Find files by pattern."""
    return [f"found: {pattern}"]


@tool(permission=Permission.MUTATING)
async def file_write(path: str, content: str) -> bool:
    """Write content to a file."""
    return True


@tool(permission=Permission.MUTATING)
async def slow_write(path: str) -> bool:
    """A slow mutating tool for testing serial execution."""
    await asyncio.sleep(0.1)
    return True


@tool(permission=Permission.DANGEROUS)
async def bash(command: str) -> str:
    """Execute a shell command."""
    return f"output of: {command}"


@tool()
async def failing_tool(x: str) -> str:
    """Always fails."""
    raise ValueError("tool error")


# ── Tests ───────────────────────────────────────────────────


class TestOrchestratorBasics:
    def test_construction(self) -> None:
        orch = ToolOrchestrator(tools=[grep, glob, file_write])
        assert len(orch.list_tools()) == 3

    def test_get_tool(self) -> None:
        orch = ToolOrchestrator(tools=[grep, glob])
        assert orch.get_tool("grep") is grep
        assert orch.get_tool("nonexistent") is None

    def test_get_schemas(self) -> None:
        orch = ToolOrchestrator(tools=[grep, file_write])
        schemas = orch.get_schemas()
        assert len(schemas) == 2
        names = {s.name for s in schemas}
        assert names == {"grep", "file_write"}


class TestOrchestratorExecute:
    @pytest.mark.asyncio
    async def test_execute_read_only(self) -> None:
        orch = ToolOrchestrator(tools=[grep])
        result = await orch.execute("grep", {"pattern": "foo"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_mutating(self) -> None:
        orch = ToolOrchestrator(tools=[file_write])
        result = await orch.execute("file_write", {"path": "/tmp/x", "content": "hi"})
        assert result.success is True
        assert result.data is True

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self) -> None:
        orch = ToolOrchestrator(tools=[grep])
        result = await orch.execute("nonexistent", {})
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_tool_error_returns_result(self) -> None:
        orch = ToolOrchestrator(tools=[failing_tool])
        result = await orch.execute("failing_tool", {"x": "test"})
        assert result.success is False
        assert "tool error" in result.error


class TestOrchestratorParallel:
    @pytest.mark.asyncio
    async def test_parallel_read_only(self) -> None:
        orch = ToolOrchestrator(tools=[grep, glob])
        calls = [
            ToolCall(id="1", name="grep", arguments={"pattern": "foo"}),
            ToolCall(id="2", name="glob", arguments={"pattern": "*.py"}),
        ]
        results = await orch.execute_parallel(calls)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_parallel_rejects_mutating(self) -> None:
        orch = ToolOrchestrator(tools=[grep, file_write])
        calls = [
            ToolCall(id="1", name="grep", arguments={"pattern": "foo"}),
            ToolCall(id="2", name="file_write", arguments={"path": "/tmp/x", "content": "hi"}),
        ]
        with pytest.raises(ValueError, match="READ_ONLY"):
            await orch.execute_parallel(calls)

    @pytest.mark.asyncio
    async def test_parallel_actually_concurrent(self) -> None:
        """Verify parallel calls don't run sequentially."""

        @tool()
        async def slow_read(n: int) -> int:
            """Slow read tool."""
            await asyncio.sleep(0.05)
            return n

        orch = ToolOrchestrator(tools=[slow_read])
        calls = [
            ToolCall(id=str(i), name="slow_read", arguments={"n": i})
            for i in range(5)
        ]
        start = time.monotonic()
        results = await orch.execute_parallel(calls)
        elapsed = time.monotonic() - start

        assert len(results) == 5
        # 5 x 0.05s sequentially = 0.25s; parallel should be ~0.05s
        assert elapsed < 0.2


class TestOrchestratorSerial:
    @pytest.mark.asyncio
    async def test_mutating_tools_serialized(self) -> None:
        """Verify mutating tools don't run concurrently."""
        execution_log: list[str] = []

        @tool(permission=Permission.MUTATING)
        async def tracked_write(label: str) -> str:
            """A mutating tool that logs execution order."""
            execution_log.append(f"start_{label}")
            await asyncio.sleep(0.05)
            execution_log.append(f"end_{label}")
            return label

        orch = ToolOrchestrator(tools=[tracked_write])

        # Fire both concurrently — they should be serialized by the lock
        results = await asyncio.gather(
            orch.execute("tracked_write", {"label": "A"}),
            orch.execute("tracked_write", {"label": "B"}),
        )
        assert all(r.success for r in results)

        # Verify serial execution: one must fully complete before the other starts
        # Either [start_A, end_A, start_B, end_B] or [start_B, end_B, start_A, end_A]
        assert (
            execution_log == ["start_A", "end_A", "start_B", "end_B"]
            or execution_log == ["start_B", "end_B", "start_A", "end_A"]
        )


class TestOrchestratorDangerous:
    @pytest.mark.asyncio
    async def test_dangerous_approved(self) -> None:
        async def always_approve(req: ApprovalRequest) -> bool:
            return True

        orch = ToolOrchestrator(tools=[bash], approval_callback=always_approve)
        result = await orch.execute("bash", {"command": "echo hi"})
        assert result.success is True
        assert "echo hi" in str(result.data)

    @pytest.mark.asyncio
    async def test_dangerous_denied(self) -> None:
        async def always_deny(req: ApprovalRequest) -> bool:
            return False

        orch = ToolOrchestrator(tools=[bash], approval_callback=always_deny)
        result = await orch.execute("bash", {"command": "rm -rf /"})
        assert result.success is False
        assert "denied" in result.error.lower() or "denied" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_dangerous_no_callback(self) -> None:
        """Without an approval callback, dangerous tools are denied."""
        orch = ToolOrchestrator(tools=[bash])
        result = await orch.execute("bash", {"command": "echo hi"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_approval_receives_context(self) -> None:
        received: list[ApprovalRequest] = []

        async def capture_approval(req: ApprovalRequest) -> bool:
            received.append(req)
            return True

        orch = ToolOrchestrator(tools=[bash], approval_callback=capture_approval)
        await orch.execute("bash", {"command": "ls -la"})

        assert len(received) == 1
        assert received[0].tool_name == "bash"
        assert received[0].arguments == {"command": "ls -la"}
