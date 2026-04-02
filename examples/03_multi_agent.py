"""Example 03 — Multi-Agent: Fork & Teammate

Demonstrates:
  - ForkAgent: parallel read-only investigation (N forks ~ 1x cost with caching)
  - TeammateAgent: independent agents coordinating via shared mailbox

Run: python examples/03_multi_agent.py
"""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from _mock_provider import MockProvider

from agentis import (
    AgentRuntime,
    ForkAgent,
    InMemoryMailbox,
    TeammateAgent,
    tool,
)


# ── Tools ───────────────────────────────────────────────────


@tool()
async def search_module(module: str) -> str:
    """Search a module for issues."""
    # In a real app, this would grep/analyze actual code
    findings = {
        "auth": "Found unvalidated JWT expiry in auth/tokens.py:42",
        "database": "N+1 query detected in database/users.py:88",
        "api": "Missing rate limiting on api/endpoints.py:15",
    }
    return findings.get(module, f"No issues found in {module}")


@tool()
async def summarize(text: str) -> str:
    """Summarize findings into a report."""
    return f"Summary: {text[:100]}..."


# ── Part A: Parallel Investigation with ForkAgent ───────────


async def demo_fork() -> None:
    print("=== Part A: Parallel Investigation (ForkAgent) ===\n")
    print("Spawning 3 forks to investigate auth, database, and api modules...\n")

    parent = AgentRuntime(
        provider=MockProvider(default_text="Investigation complete."),
        tools=[search_module],
        system_prompt="You are a code reviewer.",
    )

    # Spawn 3 parallel forks — on providers with prompt caching,
    # this costs ~1x because all forks share the same context prefix
    results = await ForkAgent.parallel_investigate(
        parent=parent,
        tasks=[
            "Search the auth module for security issues",
            "Search the database module for performance issues",
            "Search the api module for missing validations",
        ],
    )

    for i, result in enumerate(results, 1):
        print(f"  Fork {i}: {result}")

    print()


# ── Part B: Teammate Coordination ──────────────────────────


async def demo_teammate() -> None:
    print("=== Part B: Teammate Coordination ===\n")

    # Two agents share a mailbox for coordination
    mailbox = InMemoryMailbox()

    researcher = TeammateAgent(
        name="researcher",
        runtime=AgentRuntime(
            provider=MockProvider(default_text="Research complete. Findings sent to writer."),
            tools=[search_module],
            system_prompt="You are a security researcher.",
        ),
        mailbox=mailbox,
    )

    writer = TeammateAgent(
        name="writer",
        runtime=AgentRuntime(
            provider=MockProvider(default_text="Report written based on researcher's findings."),
            tools=[summarize],
            system_prompt="You are a technical writer.",
        ),
        mailbox=mailbox,
    )

    # Researcher does work and sends findings to writer
    print("  Researcher starts work...")
    result = await researcher.run("Investigate auth module for vulnerabilities")
    print(f"  Researcher: {result}")

    await researcher.send("writer", {
        "findings": "JWT expiry not validated in auth/tokens.py:42",
        "severity": "high",
    })
    print("  Researcher sent findings to writer.")

    # Writer picks up the mail and produces a report
    mail = await writer.check_mail()
    print(f"  Writer received {len(mail)} message(s): {mail[0]['content']}")

    result = await writer.run("Write a security report based on the findings")
    print(f"  Writer: {result}")


# ── Main ────────────────────────────────────────────────────


async def main() -> None:
    print("=== 03: Multi-Agent Patterns ===\n")
    await demo_fork()
    await demo_teammate()


if __name__ == "__main__":
    asyncio.run(main())
