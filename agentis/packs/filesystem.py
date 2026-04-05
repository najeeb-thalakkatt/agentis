"""Filesystem tool pack — file_read, file_write, file_edit, list_directory.

Install: pip install agentis[filesystem]  (no extra deps needed)

Usage:
    from agentis.packs.filesystem import TOOLS
    agent = AgentRuntime(provider=provider, tools=TOOLS)
"""

from __future__ import annotations

import asyncio
import os

import aiofiles

from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool
from agentis.types import Permission


@tool()
async def file_read(
    path: str,
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    """Read a file from disk, optionally returning a line range.

    Args:
        path: File path to read.
        start_line: Start line (1-indexed). 0 means from beginning.
        end_line: End line (1-indexed, inclusive). 0 means to end.
    """
    try:
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except PermissionError:
        raise PermissionError(f"Permission denied: {path}")

    if start_line > 0 or end_line > 0:
        lines = content.split("\n")
        start = max(start_line - 1, 0)
        end = end_line if end_line > 0 else len(lines)
        return "\n".join(lines[start:end])

    return content


@tool(permission=Permission.MUTATING)
async def file_write(path: str, content: str) -> bool:
    """Write content to a file, creating directories if needed.

    Args:
        path: File path to write to.
        content: Content to write.
    """
    try:
        await asyncio.to_thread(os.makedirs, os.path.dirname(path) or ".", exist_ok=True)
        async with aiofiles.open(path, "w") as f:
            await f.write(content)
        return True
    except PermissionError:
        raise PermissionError(f"Permission denied: {path}")


@tool(permission=Permission.MUTATING)
async def file_edit(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing old_text with new_text.

    Args:
        path: File path to edit.
        old_text: Text to find and replace.
        new_text: Replacement text.
    """
    try:
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

    if old_text not in content:
        raise ValueError(f"old_text not found in {path}")

    new_content = content.replace(old_text, new_text, 1)
    async with aiofiles.open(path, "w") as f:
        await f.write(new_content)

    return f"Edited {path}"


@tool()
async def list_directory(path: str) -> list[str]:
    """List files and directories at the given path.

    Args:
        path: Directory path to list.
    """
    try:
        entries = sorted(await asyncio.to_thread(os.listdir, path))
        return entries
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {path}")
    except NotADirectoryError:
        raise ValueError(f"Not a directory: {path}")


def get_filesystem_tools() -> list[FunctionTool]:
    """Return all filesystem tools."""
    return [file_read, file_write, file_edit, list_directory]


TOOLS = get_filesystem_tools()
