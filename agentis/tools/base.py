"""FunctionTool — wraps an async function into a Tool with structured I/O."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine

from agentis.protocols import ToolSchema
from agentis.types import Permission, ToolResult

logger = logging.getLogger("agentis")

# Max chars for full_output before truncation
_MAX_FULL_OUTPUT = 8000


class FunctionTool:
    """A tool that wraps an async function.

    Provides structured schema, permission gating, and exception-safe
    execution. The original function is stored in ``_fn``.

    This class IS the tool — it has ``.name``, ``.permission``,
    ``.get_schema()``, and ``.execute()``. The ``@tool`` decorator
    returns an instance of this class, not the original function.
    """

    def __init__(
        self,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        name: str,
        description: str,
        permission: Permission,
        parameter_schema: dict[str, Any],
        use_utility: bool = False,
    ) -> None:
        self._fn = fn
        self.name = name
        self.description = description
        self.permission = permission
        self._parameter_schema = parameter_schema
        self.use_utility = use_utility

    def get_schema(self) -> ToolSchema:
        """Return the JSON schema for this tool's parameters."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self._parameter_schema,
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function, catching all exceptions.

        Tools never raise — they return ``ToolResult(success=False)``
        on failure.
        """
        try:
            data = await self._fn(**kwargs)

            # Serialize to JSON for clean output; fall back to str()
            try:
                data_str = json.dumps(data, indent=2, default=str)
            except (TypeError, ValueError):
                data_str = str(data)

            # Truncate large outputs to prevent context bloat
            if len(data_str) > _MAX_FULL_OUTPUT:
                truncated = data_str[:_MAX_FULL_OUTPUT]
                data_str = truncated + f"\n... [{len(data_str) - _MAX_FULL_OUTPUT} chars truncated]"

            tokens = len(data_str) // 3

            return ToolResult(
                success=True,
                data=data,
                summary=f"{self.name} completed successfully",
                full_output=data_str,
                tokens_used=max(tokens, 1),
            )
        except Exception as e:
            logger.warning("Tool %s failed: %s", self.name, e)
            return ToolResult(
                success=False,
                data=None,
                summary=f"{self.name} failed: {e}",
                full_output="",
                tokens_used=10,
                error=str(e),
            )

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name!r}, permission={self.permission.value!r})"
