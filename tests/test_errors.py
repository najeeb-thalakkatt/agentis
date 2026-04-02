"""Tests for agentis.errors — exception hierarchy."""

from __future__ import annotations

from agentis.errors import (
    AgentisError,
    CompactionError,
    ConfigError,
    HookDeniedError,
    ProviderError,
    SessionError,
    ToolExecutionError,
)


class TestErrorHierarchy:
    def test_all_inherit_from_agentis_error(self) -> None:
        errors = [
            ProviderError("test"),
            ToolExecutionError("test"),
            HookDeniedError("test"),
            CompactionError("test"),
            SessionError("test"),
            ConfigError("test"),
        ]
        for err in errors:
            assert isinstance(err, AgentisError)
            assert isinstance(err, Exception)

    def test_agentis_error_is_exception(self) -> None:
        err = AgentisError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"

    def test_provider_error_message(self) -> None:
        err = ProviderError("API rate limit exceeded")
        assert str(err) == "API rate limit exceeded"

    def test_hook_denied_error(self) -> None:
        err = HookDeniedError("Destructive command blocked")
        assert str(err) == "Destructive command blocked"

    def test_can_be_raised_and_caught(self) -> None:
        try:
            raise ProviderError("timeout")
        except AgentisError as e:
            assert str(e) == "timeout"
        else:
            raise AssertionError("Should have been caught")

    def test_specific_catch(self) -> None:
        try:
            raise ConfigError("missing api key")
        except ConfigError as e:
            assert "api key" in str(e)
        except AgentisError:
            raise AssertionError("Should have been caught by ConfigError")
