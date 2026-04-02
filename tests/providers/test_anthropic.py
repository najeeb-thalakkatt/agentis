"""Tests for agentis.providers.anthropic — AnthropicProvider.

All SDK calls are mocked — no real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentis.errors import ConfigError, ProviderError
from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.types import Message, ToolCall


class TestAnthropicProviderImport:
    def test_raises_config_error_without_sdk(self) -> None:
        """If anthropic SDK isn't installed, construction should raise ConfigError."""
        with patch.dict("sys.modules", {"anthropic": None}):
            # Force reimport to trigger the import check
            import importlib

            from agentis.providers import anthropic as anth_mod

            importlib.reload(anth_mod)
            if not anth_mod._HAS_ANTHROPIC:
                with pytest.raises(ConfigError, match="anthropic"):
                    anth_mod.AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")


class TestAnthropicProviderCapabilities:
    def test_capabilities(self) -> None:
        from agentis.providers.anthropic import AnthropicProvider

        try:
            p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
        except ConfigError:
            pytest.skip("anthropic SDK not installed")

        caps = p.capabilities()
        assert isinstance(caps, ProviderCapabilities)
        assert caps.name == "anthropic"
        assert caps.supports_prompt_caching is True
        assert caps.supports_tool_use is True
        assert caps.supports_streaming is True


class TestAnthropicMessageConversion:
    def _make_provider(self) -> Any:
        from agentis.providers.anthropic import AnthropicProvider

        try:
            return AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
        except ConfigError:
            pytest.skip("anthropic SDK not installed")

    def test_system_message_separated(self) -> None:
        """Anthropic requires system message separate from messages array."""
        p = self._make_provider()
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]
        system, converted = p._prepare_messages(messages)
        assert system == "Be helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self) -> None:
        p = self._make_provider()
        messages = [
            Message(role="system", content="Rule 1."),
            Message(role="system", content="Rule 2."),
            Message(role="user", content="Hi"),
        ]
        system, converted = p._prepare_messages(messages)
        assert "Rule 1." in system
        assert "Rule 2." in system
        assert len(converted) == 1

    def test_no_system_message(self) -> None:
        p = self._make_provider()
        messages = [Message(role="user", content="Hello")]
        system, converted = p._prepare_messages(messages)
        assert system == ""
        assert len(converted) == 1

    def test_tool_result_message_format(self) -> None:
        """Tool results should be converted to Anthropic's tool_result format."""
        p = self._make_provider()
        from agentis.types import ToolResult

        msg = Message(
            role="tool",
            content="found 5 matches",
            metadata={"tool_use_id": "tc1"},
        )
        _, converted = p._prepare_messages([
            Message(role="user", content="search"),
            msg,
        ])
        tool_msg = converted[-1]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"


class TestAnthropicToolConversion:
    def _make_provider(self) -> Any:
        from agentis.providers.anthropic import AnthropicProvider

        try:
            return AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
        except ConfigError:
            pytest.skip("anthropic SDK not installed")

    def test_converts_tool_schema(self) -> None:
        p = self._make_provider()
        tools = [
            ToolSchema(
                name="grep",
                description="Search files",
                parameters={
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            ),
        ]
        converted = p._convert_tools_anthropic(tools)
        assert len(converted) == 1
        assert converted[0]["name"] == "grep"
        assert "input_schema" in converted[0]


class TestAnthropicComplete:
    @pytest.mark.asyncio
    async def test_complete_parses_text_response(self) -> None:
        from agentis.providers.anthropic import AnthropicProvider

        try:
            p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
        except ConfigError:
            pytest.skip("anthropic SDK not installed")

        # Mock the client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=5,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        mock_response.model = "claude-sonnet-4-6"

        p._client = MagicMock()
        p._client.messages = MagicMock()
        p._client.messages.create = AsyncMock(return_value=mock_response)

        result = await p.complete([Message(role="user", content="Hi")])
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.usage.input_tokens == 10
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_complete_parses_tool_use(self) -> None:
        from agentis.providers.anthropic import AnthropicProvider

        try:
            p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
        except ConfigError:
            pytest.skip("anthropic SDK not installed")

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tc_123"
        tool_block.name = "grep"
        tool_block.input = {"pattern": "foo"}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me search."

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock(
            input_tokens=20, output_tokens=15,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )

        p._client = MagicMock()
        p._client.messages = MagicMock()
        p._client.messages.create = AsyncMock(return_value=mock_response)

        result = await p.complete([Message(role="user", content="search")])
        assert result.content == "Let me search."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "grep"
        assert result.tool_calls[0].id == "tc_123"
        assert result.tool_calls[0].arguments == {"pattern": "foo"}
        assert result.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_complete_raises_provider_error(self) -> None:
        from agentis.providers.anthropic import AnthropicProvider

        try:
            p = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-test")
        except ConfigError:
            pytest.skip("anthropic SDK not installed")

        p._client = MagicMock()
        p._client.messages = MagicMock()
        p._client.messages.create = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(ProviderError, match="API error"):
            await p.complete([Message(role="user", content="Hi")])
