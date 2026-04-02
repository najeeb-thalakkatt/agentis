"""Exception hierarchy for the Agentis framework."""

from __future__ import annotations


class AgentisError(Exception):
    """Base exception for all Agentis errors."""


class ProviderError(AgentisError):
    """An LLM provider call failed."""


class ToolExecutionError(AgentisError):
    """A tool execution failed unexpectedly.

    Note: tools should normally return ToolResult(success=False) instead of
    raising. This exception is for truly unexpected failures.
    """


class HookDeniedError(AgentisError):
    """A lifecycle hook blocked an operation."""


class CompactionError(AgentisError):
    """Context compaction failed."""


class SessionError(AgentisError):
    """Session management error."""


class ConfigError(AgentisError):
    """Configuration or setup error."""
