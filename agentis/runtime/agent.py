"""AgentRuntime — the complete agentic loop.

Composes all 9 patterns into one coherent runtime:
    context assembly -> LLM call -> hooks -> tool execution ->
    memory update -> compaction check -> respond

Three API surfaces: run() (autonomous), step() (single turn), steps() (async iterator).
step() is the primitive; run() and steps() are built on it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentis.compaction.compactor import ContextCompactor
from agentis.compaction.dedup import FileDeduplicator
from agentis.hooks.registry import HookRegistry
from agentis.memory.index import MemoryIndex
from agentis.memory.recall_tool import RecallTool
from agentis.memory.skeptical import SkepticalMemory
from agentis.runtime.session import Session
from agentis.tools.orchestrator import ToolOrchestrator
from agentis.types import (
    ApprovalRequest,
    HookAction,
    HookContext,
    LifecycleEvent,
    Message,
    ProviderResponse,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from agentis.protocols import Extension, Provider, Tool

logger = logging.getLogger("agentis")


@dataclass
class StepResult:
    """Result of a single step (one LLM call + tool execution)."""

    response: ProviderResponse
    tool_results: list[ToolResult]
    is_final: bool
    turn_number: int


class AgentRuntime:
    """The full production agentic loop.

    Composes memory, tools, providers, hooks, compaction, and session
    management into a single runtime. Three API surfaces:

    - ``run(message)`` — autonomous loop until done or max_turns
    - ``step(message)`` — single LLM call + tool execution
    - ``steps(message)`` — async iterator yielding each step

    ``step()`` is the primitive. ``run()`` and ``steps()`` are built on it.
    """

    def __init__(
        self,
        provider: Provider,
        tools: list[Tool] | None = None,
        system_prompt: str = "",
        memory: MemoryIndex | None = None,
        hooks: HookRegistry | None = None,
        utility_provider: Provider | None = None,
        max_turns: int = 50,
        max_tokens: int = 0,
        approval_callback: Callable[[ApprovalRequest], Awaitable[bool]] | None = None,
        extensions: list[Extension] | None = None,
        skeptical: bool = True,
    ) -> None:
        self._provider = provider
        self._utility_provider = utility_provider
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._approval_callback = approval_callback
        self._extensions = extensions or []
        self._skeptical = skeptical

        # Memory
        self._memory = memory or MemoryIndex()

        # Tools — always include the recall tool
        all_tools: list[Tool] = list(tools or [])
        recall_tool = RecallTool(memory=self._memory)
        all_tools.append(recall_tool)  # type: ignore[arg-type]
        self._orchestrator = ToolOrchestrator(
            tools=all_tools,
            approval_callback=self._wrap_approval_callback(),
        )

        # Hooks
        self._hooks = hooks or HookRegistry()

        # Compaction
        max_ctx = max_tokens or provider.capabilities().max_context_tokens
        self._compactor = ContextCompactor(
            max_tokens=max_ctx,
            utility_provider=utility_provider or provider,
        )

        # Dedup
        self._dedup = FileDeduplicator()

        # Session
        self._session = Session()

    # ── Public API: Three Levels of Control ──────────────

    async def run(self, user_message: str) -> str:
        """Run the autonomous agent loop until done or max_turns.

        Args:
            user_message: The user's input message.

        Returns:
            The final text response from the agent.
        """
        last_content = ""
        async for step_result in self.steps(user_message):
            last_content = step_result.response.content
        return last_content

    async def step(self, user_message: str) -> StepResult:
        """Execute a single step: one LLM call + tool execution.

        Args:
            user_message: The user's input message.

        Returns:
            StepResult with the LLM response and any tool results.
        """
        # Add user message to session
        self._session.add_message(Message(role="user", content=user_message))

        # Assemble context
        messages = self._assemble_context()

        # Fire pre_llm_call hook
        self._fire_hook(LifecycleEvent.PRE_LLM_CALL)

        # Call LLM
        try:
            response = await self._provider.complete(
                messages=messages,
                tools=self._orchestrator.get_schemas(),
            )
        except Exception as e:
            logger.error("Provider error: %s", e)
            self._fire_hook(LifecycleEvent.ON_ERROR, metadata={"error": str(e)})
            raise

        # Fire post_llm_call hook
        self._fire_hook(LifecycleEvent.POST_LLM_CALL)

        # Add assistant response to session
        self._session.add_message(Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        ))

        # Execute tool calls
        tool_results: list[ToolResult] = []
        if response.tool_calls:
            tool_results = await self._execute_tool_calls(response.tool_calls)

            # Add tool results to session
            for tc, tr in zip(response.tool_calls, tool_results):
                self._session.add_message(Message(
                    role="tool",
                    content=tr.summary if tr.summary else str(tr.data),
                    metadata={"tool_use_id": tc.id, "tool_call_id": tc.id},
                ))

        # Increment turn
        self._session.increment_turn()

        # Compaction check
        await self._maybe_compact()

        # Notify extensions
        await self._notify_extensions_turn_end(response)

        is_final = not response.tool_calls
        return StepResult(
            response=response,
            tool_results=tool_results,
            is_final=is_final,
            turn_number=self._session.turn_count,
        )

    async def steps(self, user_message: str) -> AsyncIterator[StepResult]:
        """Async iterator yielding each step of the agent loop.

        Yields StepResult for each LLM call + tool execution round.
        Stops when the LLM returns no tool calls or max_turns is reached.
        """
        # First step uses the user message
        result = await self.step(user_message)
        yield result

        # Continue while LLM wants to call tools
        turns = 1
        while not result.is_final and turns < self._max_turns:
            # Feed tool results back (step adds them to session already)
            # Next step with empty user message to continue the loop
            result = await self._continue_step()
            yield result
            turns += 1

    # ── Session Management ──────────────────────────────

    def reset(self) -> None:
        """Clear conversation history. Keeps memory."""
        self._session.reset()

    def fork(self) -> AgentRuntime:
        """Create a new runtime with forked session state.

        The forked runtime shares the same provider and tools
        but has an independent session (conversation history).
        """
        forked = AgentRuntime(
            provider=self._provider,
            tools=[t for t in self._orchestrator.list_tools() if t.name != "recall"],
            system_prompt=self._system_prompt,
            memory=self._memory,
            hooks=self._hooks,
            utility_provider=self._utility_provider,
            max_turns=self._max_turns,
            approval_callback=self._approval_callback,
            extensions=self._extensions,
            skeptical=self._skeptical,
        )
        forked._session = self._session.fork()
        return forked

    # ── Internal Methods ────────────────────────────────

    def _assemble_context(self) -> list[Message]:
        """Build the full message list for the LLM call."""
        messages: list[Message] = []

        # System prompt
        if self._system_prompt:
            messages.append(Message(role="system", content=self._system_prompt))

        # Skeptical memory prompt (Pattern 5)
        if self._skeptical:
            messages.append(Message(
                role="system",
                content=SkepticalMemory.VERIFICATION_PROMPT,
            ))

        # Memory index
        memory_msg = self._memory.get_context_payload()
        messages.append(memory_msg)

        # Conversation history
        messages.extend(self._session.get_messages())

        return messages

    async def _continue_step(self) -> StepResult:
        """Continue the agent loop after tool execution (no new user message)."""
        messages = self._assemble_context()

        self._fire_hook(LifecycleEvent.PRE_LLM_CALL)

        try:
            response = await self._provider.complete(
                messages=messages,
                tools=self._orchestrator.get_schemas(),
            )
        except Exception as e:
            logger.error("Provider error: %s", e)
            self._fire_hook(LifecycleEvent.ON_ERROR, metadata={"error": str(e)})
            raise

        self._fire_hook(LifecycleEvent.POST_LLM_CALL)

        self._session.add_message(Message(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        ))

        tool_results: list[ToolResult] = []
        if response.tool_calls:
            tool_results = await self._execute_tool_calls(response.tool_calls)
            for tc, tr in zip(response.tool_calls, tool_results):
                self._session.add_message(Message(
                    role="tool",
                    content=tr.summary if tr.summary else str(tr.data),
                    metadata={"tool_use_id": tc.id, "tool_call_id": tc.id},
                ))

        self._session.increment_turn()
        await self._maybe_compact()
        await self._notify_extensions_turn_end(response)

        return StepResult(
            response=response,
            tool_results=tool_results,
            is_final=not response.tool_calls,
            turn_number=self._session.turn_count,
        )

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls with hook checks."""
        results: list[ToolResult] = []

        for tc in tool_calls:
            # Build hook context with permission metadata
            tool_obj = self._orchestrator.get_tool(tc.name)
            permission = tool_obj.permission.value if tool_obj else "unknown"

            hook_ctx = HookContext(
                event=LifecycleEvent.PRE_TOOL_USE,
                tool_name=tc.name,
                arguments=tc.arguments,
                agent_id="main",
                session_id=self._session.session_id,
                metadata={"permission": permission},
            )

            # Run pre_tool_use hooks
            hook_response = self._hooks.run_hooks(LifecycleEvent.PRE_TOOL_USE, hook_ctx)

            if hook_response.action == HookAction.DENY:
                results.append(ToolResult(
                    success=False,
                    data=None,
                    summary=f"Tool '{tc.name}' blocked by safety hook",
                    full_output="",
                    tokens_used=10,
                    error=f"Blocked: {hook_response.reason or 'denied by hook'}",
                ))
                continue

            if hook_response.action == HookAction.ASK_HUMAN:
                if self._approval_callback:
                    request = ApprovalRequest(
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        reason=hook_response.reason or "Hook requires approval",
                        agent_id="main",
                        session_id=self._session.session_id,
                        turn_number=self._session.turn_count,
                    )
                    approved = await self._approval_callback(request)
                    if not approved:
                        results.append(ToolResult(
                            success=False,
                            data=None,
                            summary=f"Tool '{tc.name}' denied by user",
                            full_output="",
                            tokens_used=5,
                            error=f"User denied execution of '{tc.name}'.",
                        ))
                        continue

            # Execute through orchestrator
            result = await self._orchestrator.execute(tc.name, tc.arguments)
            results.append(result)

            # Post tool use hook
            self._fire_hook(
                LifecycleEvent.POST_TOOL_USE,
                tool_name=tc.name,
                metadata={"success": result.success},
            )

        return results

    def _fire_hook(
        self,
        event: LifecycleEvent,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HookContext:
        """Fire a lifecycle hook."""
        ctx = HookContext(
            event=event,
            tool_name=tool_name,
            arguments=None,
            agent_id="main",
            session_id=self._session.session_id,
            metadata=metadata or {},
        )
        self._hooks.run_hooks(event, ctx)
        return ctx

    def _wrap_approval_callback(
        self,
    ) -> Callable[[ApprovalRequest], Awaitable[bool]] | None:
        """Wrap the user's approval callback to add session context."""
        if self._approval_callback is None:
            return None

        original = self._approval_callback

        async def wrapped(request: ApprovalRequest) -> bool:
            request.session_id = self._session.session_id
            request.agent_id = "main"
            request.turn_number = self._session.turn_count
            return await original(request)

        return wrapped

    async def _maybe_compact(self) -> None:
        """Run compaction if context is over threshold."""
        self._fire_hook(LifecycleEvent.PRE_COMPACT)
        await self._compactor.compact()

    async def _notify_extensions_turn_end(self, response: ProviderResponse) -> None:
        """Notify extensions after each turn. Failures are logged, not raised."""
        for ext in self._extensions:
            try:
                await ext.on_turn_end(self, response)
            except Exception as e:
                ext_name = getattr(ext, "name", "?")
                logger.warning("Extension '%s' on_turn_end failed: %s", ext_name, e)
