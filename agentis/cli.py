"""Agentis command-line interface.

Stdlib-only (argparse). Keeps the zero-dependency install story intact.

Commands:
  agentis new <name>   Scaffold a new project directory.
  agentis doctor       Check environment and print diagnostics.
  agentis --version    Print the installed agentis version.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from agentis import __version__

# ── Scaffold templates ──────────────────────────────────────

_PROVIDER_CHOICES = ("anthropic", "openai")

_PROVIDER_META: dict[str, dict[str, str]] = {
    "anthropic": {
        "import": "from agentis import AnthropicProvider",
        "class": "AnthropicProvider",
        "env_key": "ANTHROPIC_API_KEY",
        "extras": "anthropic",
    },
    "openai": {
        "import": "from agentis import OpenAIProvider",
        "class": "OpenAIProvider",
        "env_key": "OPENAI_API_KEY",
        "extras": "openai",
    },
}

_MAIN_PY_TEMPLATE = '''\
"""Minimal agentis agent — edit to taste.

Run:
    cp .env.example .env            # fill in your API key
    pip install -e '.[{extras}]'
    python main.py "your prompt here"
"""

from __future__ import annotations

import asyncio
import os
import sys

{provider_import}
from agentis import AgentRuntime, tool


@tool()
async def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "What is 17 + 25?"

    provider = {provider_class}.from_env()
    agent = AgentRuntime(
        provider=provider,
        tools=[add],
        system_prompt="You are a helpful assistant. Use tools when useful.",
    )
    result = await agent.run(prompt)
    print(result)


if __name__ == "__main__":
    # Best-effort .env loader (no python-dotenv dep).
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    asyncio.run(main())
'''

_ENV_EXAMPLE_TEMPLATE = """\
# Copy to .env and fill in your key.
{env_key}=
"""

_PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "An agentis-powered agent."
requires-python = ">=3.11"
dependencies = [
    "agentis-ai[{extras}]",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
"""

_README_TEMPLATE = """\
# {name}

Built with [agentis](https://github.com/).

## Quickstart

```bash
cp .env.example .env              # then fill in {env_key}
pip install -e '.[{extras}]'
python main.py "your prompt here"
```

Edit `main.py` to add tools, swap providers, or customise the system prompt.
"""

_GITIGNORE_TEMPLATE = """\
.env
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
build/
dist/
*.egg-info/
"""


# ── `agentis new` ───────────────────────────────────────────


def _scaffold(target: Path, name: str, provider: str) -> None:
    """Write scaffold files into ``target``. Caller must ensure target is empty."""
    meta = _PROVIDER_META[provider]
    target.mkdir(parents=True, exist_ok=True)

    (target / "main.py").write_text(
        _MAIN_PY_TEMPLATE.format(
            provider_import=meta["import"],
            provider_class=meta["class"],
            extras=meta["extras"],
        )
    )
    (target / ".env.example").write_text(
        _ENV_EXAMPLE_TEMPLATE.format(env_key=meta["env_key"])
    )
    (target / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(name=name, extras=meta["extras"])
    )
    (target / "README.md").write_text(
        _README_TEMPLATE.format(
            name=name, env_key=meta["env_key"], extras=meta["extras"]
        )
    )
    (target / ".gitignore").write_text(_GITIGNORE_TEMPLATE)


def cmd_new(args: argparse.Namespace) -> int:
    """Scaffold a new project directory."""
    name = args.name
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        print(
            f"error: project name must be alphanumeric (plus _ or -); got {name!r}",
            file=sys.stderr,
        )
        return 2

    target = Path.cwd() / name
    if target.exists():
        print(
            f"error: {target} already exists — choose a different name or remove it first.",
            file=sys.stderr,
        )
        return 1

    _scaffold(target, name=name, provider=args.provider)
    meta = _PROVIDER_META[args.provider]
    print(f"Created {target}")
    print()
    print("Next steps:")
    print(f"  cd {name}")
    print("  cp .env.example .env    # add your API key")
    print(f"  pip install -e '.[{meta['extras']}]'")
    print("  python main.py 'hello'")
    return 0


# ── `agentis doctor` ────────────────────────────────────────


def _check_sdk(module: str) -> bool:
    """Return True if ``module`` is importable (without actually importing)."""
    return importlib.util.find_spec(module) is not None


def cmd_doctor(args: argparse.Namespace) -> int:
    """Print environment diagnostics. Exit 0 if all checks pass, 1 otherwise."""
    lines: list[str] = []
    failures = 0

    lines.append(f"agentis    : {__version__}")
    lines.append(f"python     : {sys.version.split()[0]}")

    has_anthropic = _check_sdk("anthropic")
    has_openai = _check_sdk("openai")
    lines.append(f"anthropic  : {'installed' if has_anthropic else 'not installed'}")
    lines.append(f"openai     : {'installed' if has_openai else 'not installed'}")
    if not (has_anthropic or has_openai):
        lines.append(
            "  hint: install at least one provider — `pip install agentis-ai[anthropic]`"
        )
        failures += 1

    env_keys = {
        "ANTHROPIC_API_KEY": has_anthropic,
        "OPENAI_API_KEY": has_openai,
    }
    any_key_set = False
    for key, sdk_present in env_keys.items():
        value = os.environ.get(key, "").strip()
        present = bool(value)
        any_key_set = any_key_set or present
        status = "set" if present else "not set"
        lines.append(f"${key:<20}: {status}")
        if sdk_present and not present:
            lines.append(
                f"  hint: export {key}=... to use the matching provider via from_env()"
            )

    if (has_anthropic or has_openai) and not any_key_set:
        failures += 1

    summary = "PASS" if failures == 0 else "FAIL"
    lines.append("")
    lines.append(f"summary    : {summary}")
    print("\n".join(lines))
    return 0 if failures == 0 else 1


# ── argparse plumbing ──────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentis",
        description="Agentis CLI — scaffold and diagnose agentis projects.",
    )
    parser.add_argument(
        "--version", action="version", version=f"agentis {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="Scaffold a new agentis project.")
    p_new.add_argument("name", help="Project directory name.")
    p_new.add_argument(
        "--provider",
        choices=_PROVIDER_CHOICES,
        default="anthropic",
        help="Default provider wired into main.py (default: anthropic).",
    )
    p_new.set_defaults(func=cmd_new)

    p_doctor = sub.add_parser("doctor", help="Check environment and diagnostics.")
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
