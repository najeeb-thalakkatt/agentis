"""Mailbox backends for inter-agent communication.

Two implementations:
- InMemoryMailbox — fast, for testing and single-process use
- FileMailbox — persistent, survives crashes, for production use
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger("agentis")


class InMemoryMailbox:
    """In-memory mailbox backend for testing and single-process use.

    Messages are stored in a dict of lists. Fast but non-persistent.
    """

    def __init__(self) -> None:
        self._boxes: dict[str, list[dict[str, Any]]] = {}

    async def send(self, to: str, message: dict[str, Any]) -> None:
        """Send a message to an agent's mailbox."""
        if to not in self._boxes:
            self._boxes[to] = []
        self._boxes[to].append(message)

    async def receive(self, agent_name: str) -> list[dict[str, Any]]:
        """Receive and consume all pending messages."""
        msgs = self._boxes.pop(agent_name, [])
        return msgs

    async def peek(self, agent_name: str) -> list[dict[str, Any]]:
        """Peek at pending messages without consuming them."""
        return list(self._boxes.get(agent_name, []))

    async def clear(self, agent_name: str) -> None:
        """Clear all messages for an agent."""
        self._boxes.pop(agent_name, None)


class FileMailbox:
    """File-based mailbox backend for persistent inter-agent communication.

    Each message is stored as a JSON file in the agent's mailbox directory.
    Messages survive agent crashes and restarts.
    """

    def __init__(self, base_dir: str | Path = ".agentis/mailbox") -> None:
        self._base_dir = Path(base_dir).resolve()
        os.makedirs(self._base_dir, exist_ok=True)

    def _safe_agent_dir(self, agent_name: str) -> Path:
        """Resolve agent directory, preventing path traversal."""
        agent_dir = (self._base_dir / agent_name).resolve()
        if not str(agent_dir).startswith(str(self._base_dir)):
            raise ValueError(f"Invalid agent name (path traversal): {agent_name}")
        return agent_dir

    async def send(self, to: str, message: dict[str, Any]) -> None:
        """Send a message to an agent's mailbox directory."""
        agent_dir = self._safe_agent_dir(to)
        os.makedirs(agent_dir, exist_ok=True)

        # Use timestamp-based filename for ordering
        filename = f"{time.time_ns()}.json"
        filepath = agent_dir / filename

        async with aiofiles.open(filepath, "w") as f:
            await f.write(json.dumps(message))

    async def receive(self, agent_name: str) -> list[dict[str, Any]]:
        """Receive and consume all pending messages."""
        agent_dir = self._safe_agent_dir(agent_name)
        if not agent_dir.exists():
            return []

        messages: list[dict[str, Any]] = []
        for filepath in sorted(agent_dir.iterdir()):
            if filepath.suffix == ".json":
                try:
                    async with aiofiles.open(filepath, "r") as f:
                        content = await f.read()
                    messages.append(json.loads(content))
                    os.remove(filepath)
                except Exception as e:
                    logger.warning("Failed to read mailbox message %s: %s", filepath, e)

        return messages

    async def peek(self, agent_name: str) -> list[dict[str, Any]]:
        """Peek at pending messages without consuming them."""
        agent_dir = self._safe_agent_dir(agent_name)
        if not agent_dir.exists():
            return []

        messages: list[dict[str, Any]] = []
        for filepath in sorted(agent_dir.iterdir()):
            if filepath.suffix == ".json":
                try:
                    async with aiofiles.open(filepath, "r") as f:
                        content = await f.read()
                    messages.append(json.loads(content))
                except Exception as e:
                    logger.warning("Failed to peek mailbox message %s: %s", filepath, e)

        return messages

    async def clear(self, agent_name: str) -> None:
        """Clear all messages for an agent."""
        agent_dir = self._safe_agent_dir(agent_name)
        if not agent_dir.exists():
            return

        for filepath in agent_dir.iterdir():
            if filepath.suffix == ".json":
                try:
                    os.remove(filepath)
                except OSError:
                    pass
