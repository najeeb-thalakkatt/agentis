"""Isolation strategies for WorktreeAgent.

Three strategies:
- NoIsolation — current directory, no-op (for agents that don't mutate)
- TempDirIsolation — temporary directory (default)
- GitWorktreeIsolation — git worktree (full repo copy via ``git worktree add``)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from agentis.errors import ConfigError

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


class GitWorktreeIsolation:
    """Git worktree isolation — full repo copy via ``git worktree add``.

    Creates a detached worktree from the current HEAD. The agent gets
    an isolated copy of the entire repo that can be modified without
    affecting the main checkout. On cleanup, the worktree is removed.

    Requires git to be installed and the current directory to be inside
    a git repository. Raises ``ConfigError`` if either is missing.
    """

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self._repo_root = Path(repo_root) if repo_root else None
        self._worktree_path: Path | None = None

    async def setup(self) -> Path:
        """Create a detached git worktree and return its path."""
        # Resolve repo root
        if self._repo_root is None:
            self._repo_root = await self._find_repo_root()

        # git worktree add requires the target dir to NOT exist.
        # Use mkdtemp to get a unique name, then remove it atomically.
        tmp = tempfile.mkdtemp(prefix="agentis-wt-")
        worktree_dir = tmp + "-wt"
        shutil.rmtree(tmp)

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "worktree", "add", worktree_dir, "--detach",
                cwd=str(self._repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise ConfigError(
                    f"git worktree add failed (exit {proc.returncode}): "
                    f"{stderr.decode().strip()}"
                )
        except FileNotFoundError:
            raise ConfigError("git is not installed or not in PATH")

        self._worktree_path = Path(worktree_dir)
        logger.info("Created git worktree at %s", self._worktree_path)
        return self._worktree_path

    async def cleanup(self) -> None:
        """Remove the git worktree."""
        if self._worktree_path is None or not self._worktree_path.exists():
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "worktree", "remove", str(self._worktree_path), "--force",
                cwd=str(self._repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode != 0:
                # Fallback: manual cleanup
                logger.warning(
                    "git worktree remove failed, falling back to rmtree: %s",
                    self._worktree_path,
                )
                shutil.rmtree(self._worktree_path, ignore_errors=True)
        except Exception as e:
            logger.warning("Worktree cleanup error: %s", e)
            shutil.rmtree(self._worktree_path, ignore_errors=True)

        logger.info("Removed git worktree at %s", self._worktree_path)
        self._worktree_path = None

    def working_dir(self) -> Path:
        """Return the worktree path."""
        if self._worktree_path is None:
            raise ConfigError("Worktree not set up yet — call setup() first")
        return self._worktree_path

    @staticmethod
    async def _find_repo_root() -> Path:
        """Find the git repository root from the current directory."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--show-toplevel",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise ConfigError(
                    "Not inside a git repository. "
                    "GitWorktreeIsolation requires a git repo."
                )
            return Path(stdout.decode().strip())
        except FileNotFoundError:
            raise ConfigError("git is not installed or not in PATH")
