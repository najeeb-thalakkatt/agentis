"""ToolOrchestrator — concurrency control for tool execution.

READ_ONLY tools run in parallel. MUTATING tools run serially behind a lock.
DANGEROUS tools require human approval via a callback.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agentis.protocols import Tool, ToolSchema
from agentis.types import ApprovalRequest, Permission, ToolCall, ToolResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger("agentis")


class ToolOrchestrator:
    """Routes tool calls with concurrency control and approval gating.

    - READ_ONLY tools can run in parallel via ``execute_parallel()``.
    - MUTATING tools run one at a time behind an ``asyncio.Lock``.
    - DANGEROUS tools require human approval via ``approval_callback``.
    """

    def __init__(
        self,
        tools: list[Tool],
        approval_callback: Callable[[ApprovalRequest], Awaitable[bool]] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self._mutation_lock = asyncio.Lock()
        self._approval_callback = approval_callback

    def get_tool(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[ToolSchema]:
        """Return JSON schemas for all registered tools."""
        return [t.get_schema() for t in self._tools.values()]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a single tool with appropriate concurrency control.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Keyword arguments to pass to the tool.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                data=None,
                summary=f"Unknown tool: {tool_name}",
                full_output="",
                tokens_used=5,
                error=f"No tool named '{tool_name}' is registered.",
            )

        # DANGEROUS: require approval
        if tool.permission == Permission.DANGEROUS:
            return await self._execute_with_approval(tool, arguments)

        # MUTATING: serialize behind lock
        if tool.permission == Permission.MUTATING:
            async with self._mutation_lock:
                return await tool.execute(**arguments)

        # READ_ONLY: just run it
        return await tool.execute(**arguments)

    async def execute_parallel(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple READ_ONLY tools concurrently.

        All tools in the batch must be READ_ONLY. Raises ValueError
        if any tool is MUTATING or DANGEROUS.

        Args:
            calls: List of ToolCall objects to execute in parallel.

        Returns:
            List of ToolResult objects in the same order as the calls.

        Raises:
            ValueError: If any tool in the batch is not READ_ONLY.
        """
        # Validate all tools are READ_ONLY
        for call in calls:
            tool = self._tools.get(call.name)
            if tool is None:
                raise ValueError(f"Unknown tool: {call.name}")
            if tool.permission != Permission.READ_ONLY:
                raise ValueError(
                    f"Tool '{call.name}' has permission {tool.permission.value}, "
                    f"but execute_parallel only accepts READ_ONLY tools."
                )

        # Run all in parallel using TaskGroup
        results: list[ToolResult] = [None] * len(calls)  # type: ignore[list-item]

        async def _run(index: int, call: ToolCall) -> None:
            tool = self._tools[call.name]
            results[index] = await tool.execute(**call.arguments)

        async with asyncio.TaskGroup() as tg:
            for i, call in enumerate(calls):
                tg.create_task(_run(i, call))

        return results

    async def _execute_with_approval(
        self,
        tool: Tool,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a DANGEROUS tool after getting human approval."""
        if self._approval_callback is None:
            logger.warning(
                "Dangerous tool '%s' called without approval callback — denying.",
                tool.name,
            )
            return ToolResult(
                success=False,
                data=None,
                summary=f"Tool '{tool.name}' denied: no approval callback configured",
                full_output="",
                tokens_used=5,
                error=f"Dangerous tool '{tool.name}' denied: no approval callback configured.",
            )

        request = ApprovalRequest(
            tool_name=tool.name,
            arguments=arguments,
            reason=f"Tool '{tool.name}' requires human approval (DANGEROUS permission).",
        )

        approved = await self._approval_callback(request)
        if not approved:
            return ToolResult(
                success=False,
                data=None,
                summary=f"Tool '{tool.name}' denied by user",
                full_output="",
                tokens_used=5,
                error=f"User denied execution of '{tool.name}'.",
            )

        # Approved — execute with mutation lock (dangerous ops are also serialized)
        async with self._mutation_lock:
            return await tool.execute(**arguments)
