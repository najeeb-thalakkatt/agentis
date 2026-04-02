"""Tests for agentis.packs.web — http_get, http_post tools."""

from __future__ import annotations

from agentis.packs.web import get_web_tools
from agentis.types import Permission


class TestWebPackStructure:
    def test_get_tools_returns_list(self) -> None:
        tools = get_web_tools()
        assert len(tools) == 2

    def test_tool_names(self) -> None:
        names = {t.name for t in get_web_tools()}
        assert names == {"http_get", "http_post"}

    def test_http_get_is_read_only(self) -> None:
        tools = {t.name: t for t in get_web_tools()}
        assert tools["http_get"].permission == Permission.READ_ONLY

    def test_http_post_is_mutating(self) -> None:
        tools = {t.name: t for t in get_web_tools()}
        assert tools["http_post"].permission == Permission.MUTATING

    def test_all_have_schemas(self) -> None:
        for t in get_web_tools():
            schema = t.get_schema()
            assert schema.name
            assert "url" in str(schema.parameters)
