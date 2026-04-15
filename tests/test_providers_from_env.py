"""Tests for Provider.from_env() and provider-specific error mapping."""

from __future__ import annotations

import httpx
import pytest

from agentis.errors import (
    AuthenticationError,
    ConfigError,
    ProviderError,
    ProviderNetworkError,
    RateLimitError,
)
from agentis.providers.anthropic import AnthropicProvider, _map_anthropic_error
from agentis.providers.openai import OpenAIProvider, _map_openai_error


# ── from_env ────────────────────────────────────────────────


def test_anthropic_from_env_reads_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    provider = AnthropicProvider.from_env()
    assert provider._api_key == "sk-test-123"


def test_anthropic_from_env_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        AnthropicProvider.from_env()
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_anthropic_from_env_raises_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    with pytest.raises(ConfigError):
        AnthropicProvider.from_env()


def test_anthropic_from_env_forwards_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    provider = AnthropicProvider.from_env(model="claude-opus-4-6")
    assert provider._model == "claude-opus-4-6"


def test_openai_from_env_reads_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    provider = OpenAIProvider.from_env()
    assert provider._api_key == "sk-openai-test"


def test_openai_from_env_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        OpenAIProvider.from_env()
    assert "OPENAI_API_KEY" in str(exc_info.value)


# ── error mapping ──────────────────────────────────────────


def _mk_response(status: int) -> httpx.Response:
    req = httpx.Request("POST", "https://example.invalid/v1/messages")
    return httpx.Response(status_code=status, request=req)


def test_map_anthropic_authentication_error() -> None:
    import anthropic

    exc = anthropic.AuthenticationError(
        message="unauthorized", response=_mk_response(401), body=None
    )
    mapped = _map_anthropic_error(exc)
    assert isinstance(mapped, AuthenticationError)
    assert "ANTHROPIC_API_KEY" in str(mapped)


def test_map_anthropic_rate_limit_error() -> None:
    import anthropic

    exc = anthropic.RateLimitError(
        message="rate limited", response=_mk_response(429), body=None
    )
    mapped = _map_anthropic_error(exc)
    assert isinstance(mapped, RateLimitError)


def test_map_anthropic_connection_error() -> None:
    import anthropic

    req = httpx.Request("POST", "https://example.invalid/v1/messages")
    exc = anthropic.APIConnectionError(request=req)
    mapped = _map_anthropic_error(exc)
    assert isinstance(mapped, ProviderNetworkError)


def test_map_anthropic_unknown_falls_back_to_provider_error() -> None:
    exc = RuntimeError("weird failure")
    mapped = _map_anthropic_error(exc)
    assert isinstance(mapped, ProviderError)
    assert not isinstance(mapped, AuthenticationError)
    assert not isinstance(mapped, RateLimitError)
    assert not isinstance(mapped, ProviderNetworkError)


def test_map_openai_authentication_error() -> None:
    import openai

    exc = openai.AuthenticationError(
        message="unauthorized", response=_mk_response(401), body=None
    )
    mapped = _map_openai_error(exc)
    assert isinstance(mapped, AuthenticationError)
    assert "OPENAI_API_KEY" in str(mapped)


def test_map_openai_rate_limit_error() -> None:
    import openai

    exc = openai.RateLimitError(
        message="rate limited", response=_mk_response(429), body=None
    )
    mapped = _map_openai_error(exc)
    assert isinstance(mapped, RateLimitError)


def test_map_openai_connection_error() -> None:
    import openai

    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    exc = openai.APIConnectionError(request=req)
    mapped = _map_openai_error(exc)
    assert isinstance(mapped, ProviderNetworkError)
