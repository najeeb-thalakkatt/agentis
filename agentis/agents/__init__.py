"""Multi-agent subsystem — Fork, Teammate, and Worktree agent models."""

from __future__ import annotations

from agentis.agents.fork import ForkAgent
from agentis.agents.isolation import NoIsolation, TempDirIsolation
from agentis.agents.mailbox import FileMailbox, InMemoryMailbox
from agentis.agents.teammate import TeammateAgent
from agentis.agents.worktree import WorktreeAgent

__all__ = [
    "ForkAgent",
    "TeammateAgent",
    "WorktreeAgent",
    "TempDirIsolation",
    "NoIsolation",
    "InMemoryMailbox",
    "FileMailbox",
]
