"""TeammateAgent — independent agents with mailbox-based coordination.

Each teammate has its own runtime and communicates with other teammates
through a shared mailbox backend. Use for parallel work streams that
need coordination.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentis.protocols import MailboxBackend
    from agentis.runtime.agent import AgentRuntime

logger = logging.getLogger("agentis")


class TeammateAgent:
    """An independent agent that coordinates via mailbox.

    Two teammates must share the same mailbox instance for routing.
    Each teammate has its own runtime (provider, tools, memory).

    Use when: "You handle the frontend, I'll do the API"
    """

    def __init__(
        self,
        name: str,
        runtime: AgentRuntime,
        mailbox: MailboxBackend,
    ) -> None:
        self.name = name
        self._runtime = runtime
        self._mailbox = mailbox

    async def send(self, to: str, message: dict[str, Any]) -> None:
        """Send a message to another teammate's mailbox."""
        await self._mailbox.send(to, {
            "from": self.name,
            "content": message,
        })

    async def check_mail(self) -> list[dict[str, Any]]:
        """Check and consume all pending messages."""
        return await self._mailbox.receive(self.name)

    async def run(self, task: str) -> str:
        """Run the teammate on a task.

        Executes the task using the teammate's runtime and returns
        the final text response.
        """
        return await self._runtime.run(task)
