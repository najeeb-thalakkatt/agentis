"""Built-in safety hooks for common scenarios.

These hooks are deterministic code-level enforcement — the LLM cannot
bypass them.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

from agentis.types import HookAction, HookContext, HookResponse

# Patterns for destructive bash commands
_BANNED_PATTERNS = [
    r"rm\s+-rf\s+/",            # rm -rf /
    r"DROP\s+DATABASE",          # SQL drop
    r":\(\)\{\s*:\|:&\s*\};:",   # Fork bomb
    r"mkfs\.",                   # Format disk
    r"dd\s+if=.*of=/dev/",      # Overwrite device
    r"curl.*\|\s*sh",           # Pipe to shell
    r"wget.*\|\s*sh",           # Pipe to shell
    r"chmod\s+777",             # World-writable
    r">\s*/dev/sd",             # Overwrite disk
]

# Tools that operate on file paths
_FILE_TOOLS = frozenset({"file_read", "file_write", "file_edit"})


def block_destructive_commands(ctx: HookContext) -> HookResponse:
    """Block dangerous bash commands via regex matching.

    Only applies to the 'bash' tool. Returns ALLOW for all other tools.
    """
    if ctx.tool_name != "bash":
        return HookResponse(action=HookAction.ALLOW)

    cmd = ctx.arguments.get("command", "") if ctx.arguments else ""
    for pattern in _BANNED_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return HookResponse(
                action=HookAction.DENY,
                reason=f"Destructive command blocked: matched pattern '{pattern}'",
            )
    return HookResponse(action=HookAction.ALLOW)


def create_path_sandbox(allowed_paths: list[str]) -> Callable[[HookContext], HookResponse]:
    """Create a hook that restricts file access to allowed paths.

    Args:
        allowed_paths: List of directory paths the agent is allowed to access.

    Returns:
        A hook function that DENYs file operations outside allowed paths.
    """
    resolved_allowed = [os.path.realpath(p) for p in allowed_paths]

    def sandbox_hook(ctx: HookContext) -> HookResponse:
        if ctx.tool_name not in _FILE_TOOLS:
            return HookResponse(action=HookAction.ALLOW)

        path = ctx.arguments.get("path", "") if ctx.arguments else ""
        if not path:
            return HookResponse(action=HookAction.ALLOW)

        # Resolve the path to prevent traversal attacks
        resolved = os.path.realpath(path)

        for allowed in resolved_allowed:
            if resolved.startswith(allowed + os.sep) or resolved == allowed:
                return HookResponse(action=HookAction.ALLOW)

        return HookResponse(
            action=HookAction.DENY,
            reason=f"Path '{path}' is outside allowed directories.",
        )

    return sandbox_hook


def require_approval_for_dangerous(ctx: HookContext) -> HookResponse:
    """Require human approval for tools marked as DANGEROUS.

    Checks the 'permission' key in context metadata. If it equals
    'dangerous', returns ASK_HUMAN.
    """
    permission = ctx.metadata.get("permission", "")
    if permission == "dangerous":
        return HookResponse(
            action=HookAction.ASK_HUMAN,
            reason=f"Tool '{ctx.tool_name}' requires human approval (dangerous).",
        )
    return HookResponse(action=HookAction.ALLOW)
