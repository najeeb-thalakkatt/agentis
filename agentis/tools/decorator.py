"""@tool decorator — turns async functions into FunctionTool instances."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Coroutine

from agentis.tools.base import FunctionTool
from agentis.types import Permission

# Mapping from Python types to JSON Schema types
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(
    name: str | None = None,
    description: str | None = None,
    permission: Permission = Permission.READ_ONLY,
) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], FunctionTool]:
    """Decorator that wraps an async function into a FunctionTool.

    Args:
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the function's docstring.
        permission: Permission level. Defaults to READ_ONLY.

    Returns:
        A decorator that returns a FunctionTool instance.

    Example::

        @tool(permission=Permission.MUTATING)
        async def file_write(path: str, content: str) -> bool:
            \"\"\"Write content to a file.\"\"\"
            ...
    """

    def decorator(fn: Callable[..., Coroutine[Any, Any, Any]]) -> FunctionTool:
        tool_name = name or fn.__name__
        tool_description = description or _extract_docstring(fn)
        parameter_schema = _extract_schema(fn)

        return FunctionTool(
            fn=fn,
            name=tool_name,
            description=tool_description,
            permission=permission,
            parameter_schema=parameter_schema,
        )

    return decorator


def _extract_docstring(fn: Callable[..., Any]) -> str:
    """Extract the first line of a function's docstring."""
    doc = fn.__doc__
    if not doc:
        return ""
    # Return the first non-empty line, stripped
    for line in doc.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def _extract_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema dict from function signature and type hints."""
    sig = inspect.signature(fn)
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        hints = fn.__annotations__

    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "kwargs", "args"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        # Get the type hint
        hint = hints.get(param_name)
        json_type = _python_type_to_json(hint)

        prop: dict[str, str] = {}
        if json_type:
            prop["type"] = json_type

        properties[param_name] = prop

        # Parameters without defaults are required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json(hint: Any) -> str | None:
    """Convert a Python type hint to a JSON Schema type string."""
    if hint is None:
        return None

    # Handle direct type matches
    if hint in _TYPE_MAP:
        return _TYPE_MAP[hint]

    # Handle Optional/Union types — extract the base type
    origin = getattr(hint, "__origin__", None)
    if origin is not None:
        # For list[X], dict[X, Y], etc.
        if origin in _TYPE_MAP:
            return _TYPE_MAP[origin]

    return "string"  # safe fallback
