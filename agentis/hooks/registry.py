"""HookRegistry — lifecycle hook registration and execution.

Hooks are synchronous functions that intercept operations at the code level.
They execute deterministically regardless of what the LLM wants.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from agentis.types import HookAction, HookContext, HookResponse, LifecycleEvent

logger = logging.getLogger("agentis")

# The safety-critical event where hook failures result in DENY (fail-closed)
_FAIL_CLOSED_EVENTS = frozenset({LifecycleEvent.PRE_TOOL_USE})


@dataclass
class RegisteredHook:
    """A hook registered with the registry."""

    fn: Callable[[HookContext], HookResponse]
    name: str | None
    tools: set[str] | None
    event: LifecycleEvent


class HookRegistry:
    """Registry for lifecycle hooks with 8 events.

    Hooks are sync functions — they must be fast and deterministic.
    The registry supports tool-type filtering and fail-open/closed behavior.

    - PRE_TOOL_USE is fail-closed: hook exceptions result in DENY.
    - All other events are fail-open: exceptions are logged, operation proceeds.
    """

    def __init__(self) -> None:
        self._hooks: dict[LifecycleEvent, list[RegisteredHook]] = {
            event: [] for event in LifecycleEvent
        }

    def register(
        self,
        event: LifecycleEvent,
        fn: Callable[[HookContext], HookResponse],
        name: str | None = None,
        tools: set[str] | None = None,
    ) -> None:
        """Register a hook for a lifecycle event.

        Args:
            event: The lifecycle event to hook into.
            fn: Sync function that receives HookContext and returns HookResponse.
            name: Optional name for later unregistration.
            tools: Optional set of tool names — hook only fires for these tools.
                   If None, fires for all tools.
        """
        self._hooks[event].append(
            RegisteredHook(fn=fn, name=name, tools=tools, event=event)
        )

    def unregister(self, event: LifecycleEvent, name: str) -> bool:
        """Remove a named hook from an event.

        Returns True if the hook was found and removed.
        """
        hooks = self._hooks[event]
        for i, hook in enumerate(hooks):
            if hook.name == name:
                hooks.pop(i)
                return True
        return False

    def run_hooks(self, event: LifecycleEvent, context: HookContext) -> HookResponse:
        """Run all hooks for an event, returning the aggregate result.

        Execution rules:
            - First DENY wins (short-circuits, later hooks don't run).
            - First ASK_HUMAN wins if no DENY is encountered.
            - If all hooks ALLOW, the result is ALLOW.

        Failure rules:
            - PRE_TOOL_USE is fail-closed: exceptions become DENY.
            - All other events are fail-open: exceptions are logged, treated as ALLOW.
        """
        fail_closed = event in _FAIL_CLOSED_EVENTS
        ask_human_response: HookResponse | None = None

        for hook in self._hooks[event]:
            # Tool filtering: skip if hook has a tools filter and tool_name doesn't match
            if hook.tools is not None and context.tool_name not in hook.tools:
                continue

            try:
                response = hook.fn(context)
            except Exception as e:
                if fail_closed:
                    logger.error(
                        "Hook '%s' raised in fail-closed event %s: %s",
                        hook.name, event.value, e,
                    )
                    return HookResponse(
                        action=HookAction.DENY,
                        reason=f"Hook '{hook.name}' failed: {e}",
                    )
                else:
                    logger.warning(
                        "Hook '%s' raised in fail-open event %s: %s (continuing)",
                        hook.name, event.value, e,
                    )
                    continue

            if response.action == HookAction.DENY:
                return response

            if response.action == HookAction.ASK_HUMAN and ask_human_response is None:
                ask_human_response = response

        if ask_human_response is not None:
            return ask_human_response

        return HookResponse(action=HookAction.ALLOW)

    def list_hooks(self, event: LifecycleEvent | None = None) -> list[RegisteredHook]:
        """List registered hooks, optionally filtered by event."""
        if event is not None:
            return list(self._hooks[event])
        result: list[RegisteredHook] = []
        for hooks in self._hooks.values():
            result.extend(hooks)
        return result
