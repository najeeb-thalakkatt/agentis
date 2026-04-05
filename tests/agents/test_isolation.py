"""Tests for agentis.agents.isolation — isolation strategies."""

from __future__ import annotations

import pytest

import shutil
import subprocess

from agentis.agents.isolation import GitWorktreeIsolation, NoIsolation, TempDirIsolation
from agentis.errors import ConfigError

# Check if we're in a git repo for worktree tests
_IN_GIT_REPO = False
try:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    _IN_GIT_REPO = result.returncode == 0
except FileNotFoundError:
    pass


class TestNoIsolation:
    @pytest.mark.asyncio
    async def test_returns_current_dir(self) -> None:
        iso = NoIsolation()
        path = await iso.setup()
        assert path.exists()
        assert iso.working_dir() == path

    @pytest.mark.asyncio
    async def test_cleanup_is_noop(self) -> None:
        iso = NoIsolation()
        await iso.setup()
        await iso.cleanup()  # Should not raise


class TestTempDirIsolation:
    @pytest.mark.asyncio
    async def test_creates_temp_directory(self) -> None:
        iso = TempDirIsolation()
        path = await iso.setup()
        assert path.exists()
        assert path.is_dir()

    @pytest.mark.asyncio
    async def test_working_dir_matches_setup(self) -> None:
        iso = TempDirIsolation()
        setup_path = await iso.setup()
        assert iso.working_dir() == setup_path

    @pytest.mark.asyncio
    async def test_cleanup_removes_directory(self) -> None:
        iso = TempDirIsolation()
        path = await iso.setup()
        assert path.exists()

        await iso.cleanup()
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_prefix(self) -> None:
        iso = TempDirIsolation(prefix="agentis-test-")
        path = await iso.setup()
        assert "agentis-test-" in str(path)
        await iso.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_instances_isolated(self) -> None:
        iso1 = TempDirIsolation()
        iso2 = TempDirIsolation()
        path1 = await iso1.setup()
        path2 = await iso2.setup()
        assert path1 != path2
        await iso1.cleanup()
        await iso2.cleanup()


@pytest.mark.skipif(not _IN_GIT_REPO, reason="Not in a git repository")
class TestGitWorktreeIsolation:
    @pytest.mark.asyncio
    async def test_creates_worktree(self) -> None:
        iso = GitWorktreeIsolation()
        path = await iso.setup()
        try:
            assert path.exists()
            assert path.is_dir()
            # Should contain a .git file (worktrees use .git files, not dirs)
            assert (path / ".git").exists()
        finally:
            await iso.cleanup()

    @pytest.mark.asyncio
    async def test_working_dir_matches_setup(self) -> None:
        iso = GitWorktreeIsolation()
        setup_path = await iso.setup()
        try:
            assert iso.working_dir() == setup_path
        finally:
            await iso.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_removes_worktree(self) -> None:
        iso = GitWorktreeIsolation()
        path = await iso.setup()
        assert path.exists()
        await iso.cleanup()
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_worktree_has_repo_files(self) -> None:
        iso = GitWorktreeIsolation()
        path = await iso.setup()
        try:
            # Should have the same files as the repo
            assert (path / "agentis").is_dir()
            assert (path / "pyproject.toml").exists()
        finally:
            await iso.cleanup()

    @pytest.mark.asyncio
    async def test_worktree_is_isolated(self) -> None:
        iso = GitWorktreeIsolation()
        path = await iso.setup()
        try:
            # Create a file in the worktree — should not appear in main repo
            test_file = path / "_worktree_test_sentinel.txt"
            test_file.write_text("isolation test")
            assert test_file.exists()
            # Main repo should not have this file
            from pathlib import Path as P
            main_file = P.cwd() / "_worktree_test_sentinel.txt"
            assert not main_file.exists()
        finally:
            await iso.cleanup()

    @pytest.mark.asyncio
    async def test_working_dir_before_setup_raises(self) -> None:
        iso = GitWorktreeIsolation()
        with pytest.raises(ConfigError, match="not set up"):
            iso.working_dir()

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self) -> None:
        iso = GitWorktreeIsolation()
        await iso.setup()
        await iso.cleanup()
        await iso.cleanup()  # Should not raise
