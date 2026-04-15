"""OpenAIProvider — OpenAI API integration."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from agentis.errors import (
    AuthenticationError,
    ConfigError,
    ProviderError,
    ProviderNetworkError,
    RateLimitError,
)
from agentis.protocols import ProviderCapabilities, ToolSchema
from agentis.providers.base import BaseProvider
from agentis.types import Message, ProviderResponse, TokenUsage, ToolCall

logger = logging.getLogger("agentis")

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def _map_openai_error(exc: Exception) -> ProviderError:
    """Map an openai SDK exception to the matching agentis error class."""
    if not _HAS_OPENAI:
        return ProviderError(f"OpenAI API error: {exc}")
    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(
            "Authentication failed. Check your $OPENAI_API_KEY "
            "(run `agentis doctor` to verify)."
        )
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(
            "OpenAI rate limit hit. The SDK already retried with backoff; "
            "reduce concurrency or wait before retrying."
        )
    if isinstance(exc, openai.APIConnectionError):
        return ProviderNetworkError(
            f"Network error reaching OpenAI: {exc}. Check connectivity."
        )
    return ProviderError(f"OpenAI API error: {exc}")


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models (GPT-4o, etc.).

    Supports tool use (function calling) and streaming.
    Requires the ``openai`` package: ``pip install agentis-ai[openai]``
    """

    ENV_KEY = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not _HAS_OPENAI:
            raise ConfigError(
                "The openai SDK is required for OpenAIProvider. "
                "Install it with: pip install agentis-ai[openai]"
            )
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = openai.AsyncOpenAI(**client_kwargs)

    def capabilities(self) -> ProviderCapabilities:
        """Return OpenAI provider capabilities."""
        return ProviderCapabilities(
            name="openai",
            supports_tool_use=True,
            supports_streaming=True,
            supports_prompt_caching=False,
            supports_image_input=True,
            max_context_tokens=128_000,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send a completion request to the OpenAI API."""
        converted_messages = self._convert_messages_openai(messages)
        converted_tools = self._convert_tools_openai(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise _map_openai_error(e) from e

        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream tokens from the OpenAI API."""
        converted_messages = self._convert_messages_openai(messages)
        converted_tools = self._convert_tools_openai(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise _map_openai_error(e) from e

    def _convert_messages_openai(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert agentis Messages to OpenAI chat format."""
        result: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "tool":
                result.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.metadata.get("tool_call_id", ""),
                })
            elif msg.role == "assistant" and msg.tool_calls:
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                result.append(entry)
            else:
                result.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        return result

    def _convert_tools_openai(
        self, tools: list[ToolSchema] | None
    ) -> list[dict[str, Any]]:
        """Convert tool schemas to OpenAI function calling format."""
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _parse_response(self, response: Any) -> ProviderResponse:
        """Parse an OpenAI API response into a ProviderResponse."""
        choice = response.choices[0]
        content = choice.message.content or ""

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=choice.finish_reason,
        )
