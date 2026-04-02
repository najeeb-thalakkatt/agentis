"""OpenAICompatibleProvider — for OpenAI-compatible API endpoints."""

from __future__ import annotations

from typing import Any

from agentis.protocols import ProviderCapabilities
from agentis.providers.openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """Provider for OpenAI-compatible API endpoints (Together, Groq, etc.).

    Inherits all behavior from OpenAIProvider, but allows custom base_url
    and conservative default capabilities.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        capabilities_override: ProviderCapabilities | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self._capabilities_override = capabilities_override

    def capabilities(self) -> ProviderCapabilities:
        """Return capabilities — uses override if provided, otherwise conservative defaults."""
        if self._capabilities_override:
            return self._capabilities_override

        return ProviderCapabilities(
            name="openai_compatible",
            supports_tool_use=True,
            supports_streaming=True,
            supports_prompt_caching=False,
            supports_image_input=False,
            max_context_tokens=32_000,
        )
