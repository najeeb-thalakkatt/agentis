"""Data tool pack — JSON querying.

Install: pip install agentis[data]  (no extra deps needed)

Usage:
    from agentis.packs.data import TOOLS
    agent = AgentRuntime(provider=provider, tools=TOOLS)
"""

from __future__ import annotations

import json
from typing import Any

import aiofiles

from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool


@tool()
async def query_json(path: str, key: str = "") -> Any:
    """Query a JSON file, optionally extracting a nested key.

    Args:
        path: Path to the JSON file.
        key: Dot-separated key path (e.g., "user.name"). Empty returns entire file.

    Returns:
        The value at the key path, or the full parsed JSON.
    """
    try:
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        data = json.loads(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    if not key:
        return data

    # Navigate dot-separated key path
    current = data
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Key '{key}' not found in {path}")

    return current


def get_data_tools() -> list[FunctionTool]:
    """Return all data tools."""
    return [query_json]


TOOLS = get_data_tools()
