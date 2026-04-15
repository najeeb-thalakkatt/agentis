"""Coding tool pack — grep, glob for code search and navigation.

Install: pip install agentis-ai[coding]  (no extra deps needed for these)

Usage:
    from agentis.packs.coding import TOOLS
    agent = AgentRuntime(provider=provider, tools=TOOLS)
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re

from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool


@tool()
async def grep(pattern: str, path: str = ".", max_results: int = 50) -> list[dict[str, str]]:
    """Search for a regex pattern across files in a directory.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in.
        max_results: Maximum number of matches to return.

    Returns:
        List of match dicts with file, line_number, and line content.
    """
    return await asyncio.to_thread(_grep_sync, pattern, path, max_results)


def _grep_sync(pattern: str, path: str, max_results: int) -> list[dict[str, str]]:
    """Synchronous grep implementation, run via to_thread."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}")

    matches: list[dict[str, str]] = []

    if os.path.isfile(path):
        _search_file(path, compiled, matches, max_results)
    else:
        for root, _dirs, files in os.walk(path):
            for fname in sorted(files):
                if len(matches) >= max_results:
                    break
                filepath = os.path.join(root, fname)
                _search_file(filepath, compiled, matches, max_results)

    return matches


def _search_file(
    filepath: str,
    pattern: re.Pattern[str],
    matches: list[dict[str, str]],
    max_results: int,
) -> None:
    """Search a single file for pattern matches."""
    try:
        with open(filepath, "r", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if len(matches) >= max_results:
                    return
                if pattern.search(line):
                    matches.append({
                        "file": filepath,
                        "line_number": str(i),
                        "line": line.rstrip(),
                    })
    except (PermissionError, IsADirectoryError, UnicodeDecodeError):
        pass


@tool()
async def glob(pattern: str, path: str = ".") -> list[str]:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.py", "**/*.ts").
        path: Directory to search in.

    Returns:
        List of matching file paths.
    """
    return await asyncio.to_thread(_glob_sync, pattern, path)


def _glob_sync(pattern: str, path: str) -> list[str]:
    """Synchronous glob implementation, run via to_thread."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    matches: list[str] = []
    for root, _dirs, files in os.walk(path):
        for fname in sorted(files):
            if fnmatch.fnmatch(fname, pattern):
                matches.append(os.path.join(root, fname))

    return matches


def get_coding_tools() -> list[FunctionTool]:
    """Return all coding tools."""
    return [grep, glob]


TOOLS = get_coding_tools()
