"""WorktreeAgent — isolated execution with merge-back.

Each agent works in an isolated workspace (temp dir, git worktree, etc.)
provided by a pluggable IsolationStrategy. After completion, changes
can be merged back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentis.protocols import IsolationStrategy
    from agentis.runtime.agent import AgentRuntime

logger = logging.getLogger("agentis")


class WorktreeAgent:
    """An agent that works in an isolated workspace.

    Uses an IsolationStrategy to set up and tear down the workspace.
    The agent's runtime runs inside the isolated directory.

    Use when: "Refactor auth AND add tests simultaneously"
    """

    def __init__(
        self,
        name: str,
        runtime: AgentRuntime,
        isolation: IsolationStrategy,
    ) -> None:
        self.name = name
        self._runtime = runtime
        self._isolation = isolation

    async def run(self, task: str) -> str:
        """Run the agent in an isolated workspace.

        Sets up isolation, runs the task, then cleans up.
        Cleanup runs even if the task fails.

        Args:
            task: The task description for the agent.

        Returns:
            The agent's text response, or an error message.
        """
        await self._isolation.setup()

        try:
            result = await self._runtime.run(task)
            return result
        except Exception as e:
            logger.error("WorktreeAgent '%s' failed: %s", self.name, e)
            return f"Agent '{self.name}' encountered an error: {e}"
        finally:
            await self._isolation.cleanup()
