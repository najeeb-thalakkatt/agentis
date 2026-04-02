"""Tests for agentis.packs.data — query_json tool."""

from __future__ import annotations

import json
import os

import pytest

from agentis.packs.data import get_data_tools
from agentis.types import Permission


class TestDataPackStructure:
    def test_get_tools_returns_list(self) -> None:
        tools = get_data_tools()
        assert len(tools) == 1

    def test_query_json_is_read_only(self) -> None:
        tools = {t.name: t for t in get_data_tools()}
        assert tools["query_json"].permission == Permission.READ_ONLY


class TestQueryJson:
    @pytest.mark.asyncio
    async def test_query_top_level_key(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "data.json")
        with open(path, "w") as f:
            json.dump({"name": "Alice", "age": 30}, f)

        tools = {t.name: t for t in get_data_tools()}
        result = await tools["query_json"].execute(path=path, key="name")
        assert result.success is True
        assert result.data == "Alice"

    @pytest.mark.asyncio
    async def test_query_nested_key(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "nested.json")
        with open(path, "w") as f:
            json.dump({"user": {"name": "Bob", "role": "admin"}}, f)

        tools = {t.name: t for t in get_data_tools()}
        result = await tools["query_json"].execute(path=path, key="user.name")
        assert result.success is True
        assert result.data == "Bob"

    @pytest.mark.asyncio
    async def test_query_missing_key(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "data.json")
        with open(path, "w") as f:
            json.dump({"name": "Alice"}, f)

        tools = {t.name: t for t in get_data_tools()}
        result = await tools["query_json"].execute(path=path, key="missing")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_query_entire_file(self, tmp_path: object) -> None:
        path = os.path.join(str(tmp_path), "all.json")
        data = {"a": 1, "b": 2}
        with open(path, "w") as f:
            json.dump(data, f)

        tools = {t.name: t for t in get_data_tools()}
        result = await tools["query_json"].execute(path=path)
        assert result.success is True
        assert result.data == data

    @pytest.mark.asyncio
    async def test_query_nonexistent_file(self) -> None:
        tools = {t.name: t for t in get_data_tools()}
        result = await tools["query_json"].execute(path="/tmp/nonexistent_agentis_xyz.json")
        assert result.success is False
