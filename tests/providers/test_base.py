"""Tests for agentis.providers.base — BaseProvider."""

from __future__ import annotations

import pytest

from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.providers.base import BaseProvider
from agentis.types import Message, ProviderResponse, TokenUsage, ToolCall

# ── Concrete subclass for testing abstract base ─────────────


class StubProvider(BaseProvider):
    """Minimal concrete provider for testing base class behavior."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(model="stub-model", **kwargs)
        self._complete_response: ProviderResponse | None = None
        self._call_count = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        self._call_count += 1
        if self._complete_response:
            return self._complete_response
        return ProviderResponse(
            content="stub response",
            tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        yield "stub "
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="stub")


# ── Tests ───────────────────────────────────────────────────


class TestBaseProviderConstruction:
    def test_model_stored(self) -> None:
        p = StubProvider()
        assert p._model == "stub-model"

    def test_api_key_stored(self) -> None:
        p = StubProvider(api_key="sk-test123")
        assert p._api_key == "sk-test123"

    def test_base_url_default_none(self) -> None:
        p = StubProvider()
        assert p._base_url is None

    def test_base_url_custom(self) -> None:
        p = StubProvider(base_url="https://custom.api.com")
        assert p._base_url == "https://custom.api.com"


class TestBaseProviderMessageConversion:
    def test_convert_simple_messages(self) -> None:
        p = StubProvider()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        converted = p._convert_messages(messages)
        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[1]["role"] == "user"
        assert converted[2]["role"] == "assistant"
        assert converted[1]["content"] == "Hello"

    def test_convert_message_with_tool_calls(self) -> None:
        p = StubProvider()
        tc = ToolCall(id="tc1", name="grep", arguments={"pattern": "foo"})
        msg = Message(role="assistant", content="Let me search.", tool_calls=[tc])
        converted = p._convert_messages([msg])
        assert converted[0]["role"] == "assistant"
        # Tool calls should be present in the converted message
        assert "tool_calls" in converted[0]
        assert len(converted[0]["tool_calls"]) == 1


class TestBaseProviderToolConversion:
    def test_convert_tools(self) -> None:
        p = StubProvider()
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
        converted = p._convert_tools(tools)
        assert len(converted) == 1
        assert converted[0]["name"] == "grep"
        assert converted[0]["description"] == "Search files"

    def test_convert_empty_tools(self) -> None:
        p = StubProvider()
        assert p._convert_tools([]) == []

    def test_convert_none_tools(self) -> None:
        p = StubProvider()
        assert p._convert_tools(None) == []


class TestBaseProviderRetry:
    @pytest.mark.asyncio
    async def test_retry_returns_result_on_success(self) -> None:
        p = StubProvider()
        call_count = 0

        async def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await p._retry_with_backoff(succeed, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_retries_on_transient_error(self) -> None:
        p = StubProvider()
        call_count = 0

        async def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("rate limited")
            return "ok"

        result = await p._retry_with_backoff(
            fail_then_succeed, max_retries=5, base_delay=0.01
        )
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self) -> None:
        p = StubProvider()

        async def always_fail() -> str:
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            await p._retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)
