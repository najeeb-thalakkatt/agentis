"""Tests for agentis.tools.decorator — @tool decorator."""

from __future__ import annotations

import pytest

from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool
from agentis.types import Permission, ToolResult


class TestToolDecorator:
    def test_returns_function_tool(self) -> None:
        @tool()
        async def my_tool(x: str) -> str:
            """Do something."""
            return x

        assert isinstance(my_tool, FunctionTool)

    def test_auto_name_from_function(self) -> None:
        @tool()
        async def search_files(pattern: str) -> list[str]:
            """Search for files matching a pattern."""
            return []

        assert search_files.name == "search_files"

    def test_explicit_name(self) -> None:
        @tool(name="custom_name")
        async def my_func(x: str) -> str:
            """A tool."""
            return x

        assert my_func.name == "custom_name"

    def test_auto_description_from_docstring(self) -> None:
        @tool()
        async def grep(pattern: str, path: str = ".") -> list[str]:
            """Search for patterns across files."""
            return []

        assert grep.description == "Search for patterns across files."

    def test_explicit_description(self) -> None:
        @tool(description="Custom description")
        async def my_func(x: str) -> str:
            """Original docstring."""
            return x

        assert my_func.description == "Custom description"

    def test_default_permission_is_read_only(self) -> None:
        @tool()
        async def my_func(x: str) -> str:
            """A tool."""
            return x

        assert my_func.permission == Permission.READ_ONLY

    def test_explicit_permission(self) -> None:
        @tool(permission=Permission.MUTATING)
        async def file_write(path: str, content: str) -> bool:
            """Write a file."""
            return True

        assert file_write.permission == Permission.MUTATING

    def test_dangerous_permission(self) -> None:
        @tool(permission=Permission.DANGEROUS)
        async def bash(command: str) -> str:
            """Execute a shell command."""
            return ""

        assert bash.permission == Permission.DANGEROUS


class TestToolDecoratorSchema:
    def test_schema_extracts_parameters(self) -> None:
        @tool()
        async def grep(pattern: str, path: str = ".", max_results: int = 50) -> list[str]:
            """Search for patterns."""
            return []

        schema = grep.get_schema()
        props = schema.parameters["properties"]
        assert "pattern" in props
        assert "path" in props
        assert "max_results" in props

    def test_schema_required_params(self) -> None:
        @tool()
        async def grep(pattern: str, path: str = ".") -> list[str]:
            """Search for patterns."""
            return []

        schema = grep.get_schema()
        assert "pattern" in schema.parameters["required"]
        # path has a default, so should NOT be required
        assert "path" not in schema.parameters["required"]

    def test_schema_type_mapping(self) -> None:
        @tool()
        async def multi_types(
            name: str,
            count: int,
            ratio: float,
            active: bool,
        ) -> str:
            """Test type mapping."""
            return ""

        schema = multi_types.get_schema()
        props = schema.parameters["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["active"]["type"] == "boolean"

    def test_no_docstring_uses_empty_description(self) -> None:
        @tool()
        async def no_doc(x: str) -> str:
            return x

        assert no_doc.description == ""


class TestToolDecoratorExecution:
    @pytest.mark.asyncio
    async def test_decorated_tool_executes(self) -> None:
        @tool()
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add.execute(a=3, b=4)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == 7

    @pytest.mark.asyncio
    async def test_decorated_tool_catches_errors(self) -> None:
        @tool()
        async def fail_tool(x: str) -> str:
            """Always fails."""
            raise RuntimeError("boom")

        result = await fail_tool.execute(x="test")
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_can_pass_to_list(self) -> None:
        """Tools created by decorator can be collected into a list."""

        @tool()
        async def t1(x: str) -> str:
            """Tool 1."""
            return x

        @tool()
        async def t2(y: int) -> int:
            """Tool 2."""
            return y

        tools = [t1, t2]
        assert len(tools) == 2
        assert tools[0].name == "t1"
        assert tools[1].name == "t2"
