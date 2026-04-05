"""Shell tool pack — bash command execution.

Install: pip install agentis[shell]  (no extra deps needed)

Usage:
    from agentis.packs.shell import TOOLS
    agent = AgentRuntime(provider=provider, tools=TOOLS)
"""

from __future__ import annotations

import asyncio
import os

from agentis.tools.base import FunctionTool
from agentis.tools.decorator import tool
from agentis.types import Permission

# Environment variables to strip from subprocess environment
_SENSITIVE_ENV_KEYS = frozenset({
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN", "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
    "DATABASE_URL", "SECRET_KEY", "PRIVATE_KEY",
})


def _safe_env() -> dict[str, str]:
    """Return a copy of the environment with sensitive keys removed."""
    return {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_KEYS}


@tool(permission=Permission.DANGEROUS)
async def bash(command: str, timeout: int = 120) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Timeout in seconds (default 120).
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_safe_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    output = stdout.decode("utf-8", errors="replace")
    err_output = stderr.decode("utf-8", errors="replace")
    combined = output + err_output

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command exited with code {proc.returncode}: {err_output or output}"
        )

    return combined


def get_shell_tools() -> list[FunctionTool]:
    """Return all shell tools."""
    return [bash]


TOOLS = get_shell_tools()
