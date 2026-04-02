"""Tests for agentis.agents.isolation — isolation strategies."""

from __future__ import annotations

import pytest

from agentis.agents.isolation import NoIsolation, TempDirIsolation


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
