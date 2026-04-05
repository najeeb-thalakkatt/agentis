"""Tool packs — domain-specific tool collections as optional extras.

Use ``load_pack("filesystem")`` to get a pack's tools by name,
or ``load_packs(["filesystem", "coding"])`` for multiple packs.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentis.protocols import Tool

# Registry: pack name -> module path
_PACK_REGISTRY: dict[str, str] = {
    "filesystem": "agentis.packs.filesystem",
    "coding": "agentis.packs.coding",
    "data": "agentis.packs.data",
    "shell": "agentis.packs.shell",
    "web": "agentis.packs.web",
}


def load_pack(name: str) -> list[Tool]:
    """Load a tool pack by name.

    Args:
        name: Pack name (e.g., "filesystem", "coding", "shell").

    Returns:
        List of Tool instances from the pack.

    Raises:
        KeyError: If the pack name is not registered.
        ImportError: If the pack module cannot be imported.
    """
    if name not in _PACK_REGISTRY:
        available = ", ".join(sorted(_PACK_REGISTRY))
        raise KeyError(f"Unknown pack '{name}'. Available: {available}")

    module = importlib.import_module(_PACK_REGISTRY[name])
    return list(module.TOOLS)


def load_packs(names: list[str]) -> list[Tool]:
    """Load multiple tool packs and combine their tools.

    Args:
        names: List of pack names.

    Returns:
        Combined list of Tool instances from all requested packs.
    """
    tools: list[Tool] = []
    for name in names:
        tools.extend(load_pack(name))
    return tools


def list_packs() -> list[str]:
    """Return all registered pack names."""
    return sorted(_PACK_REGISTRY)


def register_pack(name: str, module_path: str) -> None:
    """Register a custom tool pack.

    Args:
        name: Pack name for use with load_pack().
        module_path: Dotted module path (must export a TOOLS list).
    """
    _PACK_REGISTRY[name] = module_path
