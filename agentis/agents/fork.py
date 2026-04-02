"""ForkAgent — cheap clones for parallel read-only investigation.

Creates a byte-identical copy of the parent context. On providers with
prompt caching, running N forks costs approximately 1x (shared prefix).
Forks can only use READ_ONLY tools.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agentis.types import Permission

if TYPE_CHECKING:
    from agentis.runtime.agent import AgentRuntime

logger = logging.getLogger("agentis")


class ForkAgent:
    """A read-only clone of a parent runtime for parallel investigation.

    The fork inherits the parent's conversation context but can only
    use READ_ONLY tools. It cannot mutate files or state.

    Use when: "Search these 5 files for the bug" — spawn 5 forks.
    """

    def __init__(self, parent: AgentRuntime, task: str) -> None:
        self._task = task
        # Fork the parent, stripping non-read-only tools
        read_only_tools = [
            t for t in parent._orchestrator.list_tools()
            if t.permission == Permission.READ_ONLY
        ]
        self._runtime = parent.fork()
        # Replace orchestrator with read-only-only tools
        from agentis.tools.orchestrator import ToolOrchestrator

        self._runtime._orchestrator = ToolOrchestrator(tools=read_only_tools)

    async def run(self) -> str:
        """Run the fork with its assigned task. Returns the text result."""
        return await self._runtime.run(
            f"SUBTASK (read-only): {self._task}\n"
            f"Return your findings. Do NOT edit files."
        )

    @staticmethod
    async def parallel_investigate(
        parent: AgentRuntime,
        tasks: list[str],
    ) -> list[str]:
        """Spawn N forks and run them in parallel.

        On providers with prompt caching, this costs approximately 1x
        because all forks share the same context prefix.

        Args:
            parent: The parent runtime to fork from.
            tasks: List of investigation tasks.

        Returns:
            List of results in the same order as tasks.
        """
        if not tasks:
            return []

        forks = [ForkAgent(parent=parent, task=t) for t in tasks]
        results: list[str] = [""] * len(forks)

        async def _run(index: int, fork: ForkAgent) -> None:
            results[index] = await fork.run()

        async with asyncio.TaskGroup() as tg:
            for i, fork in enumerate(forks):
                tg.create_task(_run(i, fork))

        return results
