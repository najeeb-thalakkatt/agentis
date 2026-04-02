"""Example 04 — Persistent Memory & Provider Swap

Demonstrates:
  - MemoryIndex: remember/recall/search across sessions
  - Memory persistence: load() restores state from disk
  - Provider swap: one-line switch between Anthropic, OpenAI, or local models

Run: python examples/04_memory_and_providers.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from _mock_provider import MockProvider

from agentis import AgentRuntime, MemoryIndex

# ── Part A: Persistent Memory ──────────────────────────────


async def demo_memory() -> None:
    print("=== Part A: Persistent Memory ===\n")

    workspace = tempfile.mkdtemp(prefix="agentis-example-")

    try:
        # --- Session 1: Store findings ---
        print("Session 1: Storing findings...")
        memory = MemoryIndex(workspace=workspace)

        await memory.remember(
            "Project architecture",
            "Monorepo with 3 services: auth (Python), api (Go), web (React). "
            "Shared protobuf definitions in /proto. CI runs on GitHub Actions.",
            tags=["architecture", "infrastructure"],
        )
        tid2 = await memory.remember(
            "Auth bug: JWT expiry",
            "JWT tokens are not validated for expiry in auth/tokens.py:42. "
            "Discovered during security audit on 2026-03-15. Priority: high.",
            tags=["bug", "security", "high-priority"],
        )
        await memory.remember(
            "Team contacts",
            "Auth team lead: Alice (alice@example.com). "
            "API team lead: Bob (bob@example.com).",
            tags=["team", "contacts"],
        )

        print(f"  Stored {len(memory.entries)} topics:")
        for entry in memory.entries:
            print(f"    - {entry.topic} [{', '.join(entry.tags)}] -> {entry.topic_id}")

        # Recall full content
        content = await memory.recall(tid2)
        print(f"\n  Recalled '{memory.entries[1].topic}':")
        print(f"    {content.strip()[:80]}...")

        # --- Session 2: Load from disk ---
        print(f"\nSession 2: Loading from disk ({workspace})...")
        memory2 = MemoryIndex(workspace=workspace)
        await memory2.load()

        print(f"  Loaded {len(memory2.entries)} topics:")
        for entry in memory2.entries:
            print(f"    - {entry.topic}")

        # Search by tags
        bugs = await memory2.search(tags=["bug"])
        print(f"\n  Search for 'bug' tag: {[b.topic for b in bugs]}")

        security = await memory2.search(query="Auth")
        print(f"  Search for 'Auth' query: {[s.topic for s in security]}")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ── Part B: Provider Swap ──────────────────────────────────


async def demo_providers() -> None:
    print("\n=== Part B: Provider Swap ===\n")

    # Swap providers in one line. Uncomment the one you want:

    # Option 1: Anthropic Claude
    # from agentis import AnthropicProvider
    # provider = AnthropicProvider(
    #     model="claude-sonnet-4-20250514",
    #     api_key=os.getenv("ANTHROPIC_API_KEY"),
    # )

    # Option 2: OpenAI
    # from agentis import OpenAIProvider
    # provider = OpenAIProvider(
    #     model="gpt-4o",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )

    # Option 3: Any OpenAI-compatible endpoint (Ollama, vLLM, Together, Groq)
    # from agentis import OpenAICompatibleProvider
    # provider = OpenAICompatibleProvider(
    #     model="llama3",
    #     base_url="http://localhost:11434/v1",
    # )

    # For this example, we use MockProvider:
    provider = MockProvider(default_text="The auth module uses JWT with 15-min expiry.")
    print("  Using MockProvider (swap one line above for a real provider)\n")

    agent = AgentRuntime(
        provider=provider,
        system_prompt="You are a helpful assistant with project knowledge.",
    )

    result = await agent.run("What does the auth module use for tokens?")
    print(f"  Agent: {result}")


# ── Main ────────────────────────────────────────────────────


async def main() -> None:
    print("=== 04: Memory & Providers ===\n")
    await demo_memory()
    await demo_providers()


if __name__ == "__main__":
    asyncio.run(main())
