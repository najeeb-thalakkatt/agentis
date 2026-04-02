"""Isolation strategies for WorktreeAgent.

Three strategies:
- NoIsolation — current directory, no-op (for agents that don't mutate)
- TempDirIsolation — temporary directory (default)
- GitWorktreeIsolation — git worktree (in coding pack, protocol here)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger("agentis")


class NoIsolation:
    """No isolation — agent works in the current directory.

    Use when the agent doesn't mutate files or when isolation
    is handled externally.
    """

    def __init__(self) -> None:
        self._path = Path.cwd()

    async def setup(self) -> Path:
        """Return the current working directory."""
        return self._path

    async def cleanup(self) -> None:
        """No-op — nothing to clean up."""

    def working_dir(self) -> Path:
        """Return the working directory."""
        return self._path


class TempDirIsolation:
    """Temporary directory isolation.

    Creates a fresh temp directory for each agent. Cleaned up
    after the agent finishes. This is the default isolation strategy.
    """

    def __init__(self, prefix: str = "agentis-") -> None:
        self._prefix = prefix
        self._path: Path = Path(tempfile.gettempdir())  # placeholder

    async def setup(self) -> Path:
        """Create and return a temporary directory."""
        self._path = Path(tempfile.mkdtemp(prefix=self._prefix))
        return self._path

    async def cleanup(self) -> None:
        """Remove the temporary directory and all contents."""
        if self._path.exists():
            try:
                shutil.rmtree(self._path)
            except OSError as e:
                logger.warning("Failed to clean up temp dir %s: %s", self._path, e)

    def working_dir(self) -> Path:
        """Return the temporary directory path."""
        return self._path
