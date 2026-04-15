"""Web tool pack — HTTP request tools.

Install: pip install agentis-ai[web]  (requires httpx)

Usage:
    from agentis.packs.web import TOOLS
    agent = AgentRuntime(provider=provider, tools=TOOLS)
"""

from __future__ import annotations

from typing import Any

from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool
from agentis.types import Permission


@tool()
async def http_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Make an HTTP GET request.

    Args:
        url: The URL to request.
        headers: Optional request headers.

    Returns:
        Dict with status_code, headers, and body.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for web tools: pip install agentis-ai[web]")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers or {})
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text[:10000],  # Cap body size
        }


@tool(permission=Permission.MUTATING)
async def http_post(
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make an HTTP POST request.

    Args:
        url: The URL to request.
        body: JSON body to send.
        headers: Optional request headers.

    Returns:
        Dict with status_code, headers, and body.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for web tools: pip install agentis-ai[web]")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=body, headers=headers or {})
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text[:10000],
        }


def get_web_tools() -> list[FunctionTool]:
    """Return all web tools."""
    return [http_get, http_post]


TOOLS = get_web_tools()
