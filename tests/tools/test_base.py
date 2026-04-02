"""Tests for agentis.tools.base — FunctionTool."""

from __future__ import annotations

import pytest

from agentis.tools.base import FunctionTool
from agentis.types import Permission, ToolResult


async def dummy_read(path: str, line_range: tuple[int, int] | None = None) -> str:
    """Read a file from disk."""
    return f"contents of {path}"


async def dummy_add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


async def dummy_failing(**kwargs: object) -> str:
    """A tool that always raises."""
    raise ValueError("something went wrong")


async def no_docstring(x: str) -> str:
    return x


class TestFunctionToolConstruction:
    def test_basic_construction(self) -> None:
        ft = FunctionTool(
            fn=dummy_read,
            name="file_read",
            description="Read a file",
            permission=Permission.READ_ONLY,
            parameter_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        )
        assert ft.name == "file_read"
        assert ft.description == "Read a file"
        assert ft.permission == Permission.READ_ONLY

    def test_stores_original_function(self) -> None:
        ft = FunctionTool(
            fn=dummy_read,
            name="file_read",
            description="Read a file",
            permission=Permission.READ_ONLY,
            parameter_schema={},
        )
        assert ft._fn is dummy_read


class TestFunctionToolSchema:
    def test_get_schema(self) -> None:
        ft = FunctionTool(
            fn=dummy_read,
            name="file_read",
            description="Read a file",
            permission=Permission.READ_ONLY,
            parameter_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        )
        schema = ft.get_schema()
        assert schema.name == "file_read"
        assert schema.description == "Read a file"
        assert schema.parameters["properties"]["path"]["type"] == "string"
        assert "path" in schema.parameters["required"]


class TestFunctionToolExecute:
    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        ft = FunctionTool(
            fn=dummy_add,
            name="add",
            description="Add numbers",
            permission=Permission.READ_ONLY,
            parameter_schema={},
        )
        result = await ft.execute(a=2, b=3)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == 5

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self) -> None:
        ft = FunctionTool(
            fn=dummy_read,
            name="file_read",
            description="Read a file",
            permission=Permission.READ_ONLY,
            parameter_schema={},
        )
        result = await ft.execute(path="/tmp/test.txt")
        assert result.success is True
        assert "contents of /tmp/test.txt" in str(result.data)

    @pytest.mark.asyncio
    async def test_execute_catches_exceptions(self) -> None:
        ft = FunctionTool(
            fn=dummy_failing,
            name="bad_tool",
            description="Always fails",
            permission=Permission.READ_ONLY,
            parameter_schema={},
        )
        result = await ft.execute()
        assert result.success is False
        assert result.error is not None
        assert "something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_execute_never_raises(self) -> None:
        ft = FunctionTool(
            fn=dummy_failing,
            name="bad_tool",
            description="Always fails",
            permission=Permission.READ_ONLY,
            parameter_schema={},
        )
        # Should not raise — returns ToolResult with success=False
        result = await ft.execute()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_summary_on_success(self) -> None:
        ft = FunctionTool(
            fn=dummy_add,
            name="add",
            description="Add numbers",
            permission=Permission.READ_ONLY,
            parameter_schema={},
        )
        result = await ft.execute(a=1, b=2)
        assert result.summary  # non-empty summary

    @pytest.mark.asyncio
    async def test_execute_tokens_estimated(self) -> None:
        ft = FunctionTool(
            fn=dummy_read,
            name="file_read",
            description="Read a file",
            permission=Permission.READ_ONLY,
            parameter_schema={},
        )
        result = await ft.execute(path="/tmp/x")
        assert result.tokens_used > 0
