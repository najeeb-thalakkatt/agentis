"""Tests for agentis.providers.openai — OpenAIProvider and OpenAICompatibleProvider.

All SDK calls are mocked — no real API calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentis.errors import ConfigError, ProviderError
from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.types import Message


def _make_openai_provider() -> Any:
    from agentis.providers.openai import OpenAIProvider

    try:
        return OpenAIProvider(model="gpt-4o", api_key="sk-test")
    except ConfigError:
        pytest.skip("openai SDK not installed")


def _make_compatible_provider() -> Any:
    from agentis.providers.openai_compatible import OpenAICompatibleProvider

    try:
        return OpenAICompatibleProvider(
            model="llama-3",
            api_key="sk-test",
            base_url="https://api.together.xyz/v1",
        )
    except ConfigError:
        pytest.skip("openai SDK not installed")


class TestOpenAIProviderCapabilities:
    def test_capabilities(self) -> None:
        p = _make_openai_provider()
        caps = p.capabilities()
        assert isinstance(caps, ProviderCapabilities)
        assert caps.name == "openai"
        assert caps.supports_tool_use is True
        assert caps.supports_prompt_caching is False

    def test_model_stored(self) -> None:
        p = _make_openai_provider()
        assert p._model == "gpt-4o"


class TestOpenAIMessageConversion:
    def test_system_message_in_array(self) -> None:
        """OpenAI keeps system messages in the messages array."""
        p = _make_openai_provider()
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
        ]
        converted = p._convert_messages_openai(messages)
        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "Be helpful."

    def test_tool_calls_in_assistant_message(self) -> None:
        from agentis.types import ToolCall

        p = _make_openai_provider()
        tc = ToolCall(id="call_1", name="grep", arguments={"pattern": "foo"})
        messages = [
            Message(role="assistant", content="Searching.", tool_calls=[tc]),
        ]
        converted = p._convert_messages_openai(messages)
        assert converted[0]["tool_calls"][0]["id"] == "call_1"
        assert converted[0]["tool_calls"][0]["type"] == "function"
        assert converted[0]["tool_calls"][0]["function"]["name"] == "grep"

    def test_tool_result_message(self) -> None:
        p = _make_openai_provider()
        msg = Message(
            role="tool",
            content="5 matches found",
            metadata={"tool_call_id": "call_1"},
        )
        converted = p._convert_messages_openai([msg])
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "call_1"


class TestOpenAIToolConversion:
    def test_converts_to_function_format(self) -> None:
        p = _make_openai_provider()
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
        converted = p._convert_tools_openai(tools)
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "grep"
        assert converted[0]["function"]["parameters"]["properties"]["pattern"]["type"] == "string"


class TestOpenAIComplete:
    @pytest.mark.asyncio
    async def test_complete_parses_text_response(self) -> None:
        p = _make_openai_provider()

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5,
        )

        p._client = MagicMock()
        p._client.chat = MagicMock()
        p._client.chat.completions = MagicMock()
        p._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await p.complete([Message(role="user", content="Hi")])
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_parses_tool_calls(self) -> None:
        p = _make_openai_provider()

        mock_tc = MagicMock()
        mock_tc.id = "call_abc"
        mock_tc.type = "function"
        mock_tc.function.name = "grep"
        mock_tc.function.arguments = json.dumps({"pattern": "foo"})

        mock_choice = MagicMock()
        mock_choice.message.content = "Searching."
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=15)

        p._client = MagicMock()
        p._client.chat = MagicMock()
        p._client.chat.completions = MagicMock()
        p._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await p.complete([Message(role="user", content="search")])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "grep"
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].arguments == {"pattern": "foo"}

    @pytest.mark.asyncio
    async def test_complete_raises_provider_error(self) -> None:
        p = _make_openai_provider()

        p._client = MagicMock()
        p._client.chat = MagicMock()
        p._client.chat.completions = MagicMock()
        p._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(ProviderError, match="API error"):
            await p.complete([Message(role="user", content="Hi")])


class TestOpenAICompatibleProvider:
    def test_capabilities_conservative(self) -> None:
        p = _make_compatible_provider()
        caps = p.capabilities()
        assert caps.name == "openai_compatible"
        assert caps.supports_prompt_caching is False

    def test_base_url_set(self) -> None:
        p = _make_compatible_provider()
        assert p._base_url == "https://api.together.xyz/v1"

    def test_custom_capabilities_override(self) -> None:
        from agentis.providers.openai_compatible import OpenAICompatibleProvider

        try:
            custom_caps = ProviderCapabilities(
                name="groq",
                supports_prompt_caching=False,
                max_context_tokens=32_000,
            )
            p = OpenAICompatibleProvider(
                model="llama-3",
                api_key="sk-test",
                base_url="https://api.groq.com/v1",
                capabilities_override=custom_caps,
            )
        except ConfigError:
            pytest.skip("openai SDK not installed")

        caps = p.capabilities()
        assert caps.name == "groq"
        assert caps.max_context_tokens == 32_000
