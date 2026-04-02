"""Example 01 — Quickstart: The Basic Agentic Loop

Demonstrates:
  - @tool decorator with READ_ONLY, MUTATING, and DANGEROUS permissions
  - AgentRuntime with run() (autonomous loop) and step() (single turn)
  - How the agent calls tools and the framework returns structured results

Run: python examples/01_quickstart.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _mock_provider import MockProvider

from agentis import (
    AgentRuntime,
    Permission,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    tool,
)

# ── Define tools with different permission levels ───────────


@tool()
async def search_docs(query: str) -> list[str]:
    """Search the knowledge base for relevant documents."""
    # In a real app, this would hit a vector DB or search API
    return [
        f"Result 1: '{query}' found in architecture.md",
        f"Result 2: '{query}' found in api-reference.md",
    ]


@tool(permission=Permission.MUTATING)
async def save_note(title: str, content: str) -> bool:
    """Save a research note to the project."""
    # In a real app, this would write to a file or database
    print(f"  [tool] Saved note: '{title}'")
    return True


@tool(permission=Permission.DANGEROUS)
async def delete_all_notes() -> str:
    """Delete all research notes. Requires human approval."""
    return "All notes deleted."


# ── Script the mock LLM responses ──────────────────────────


provider = MockProvider(responses=[
    # Turn 1: LLM decides to search
    ProviderResponse(
        content="Let me search for that.",
        tool_calls=[ToolCall(
            id="t1", name="search_docs",
            arguments={"query": "quantum computing"},
        )],
        usage=TokenUsage(input_tokens=50, output_tokens=20),
        stop_reason="tool_use",
    ),
    # Turn 2: LLM saves a note based on search results
    ProviderResponse(
        content="Found some results. Saving a note.",
        tool_calls=[ToolCall(id="t2", name="save_note", arguments={
            "title": "Quantum Computing Research",
            "content": "Key findings from architecture.md and api-reference.md",
        })],
        usage=TokenUsage(input_tokens=80, output_tokens=30),
        stop_reason="tool_use",
    ),
    # Turn 3: Final answer
    ProviderResponse(
        content="I researched quantum computing and saved a note with the key findings. "
                "The topic is covered in architecture.md and api-reference.md.",
        tool_calls=[],
        usage=TokenUsage(input_tokens=100, output_tokens=40),
        stop_reason="end_turn",
    ),
])


async def main() -> None:
    print("=== 01: Quickstart - Research Assistant ===\n")

    # Create the agent runtime
    agent = AgentRuntime(
        provider=provider,
        tools=[search_docs, save_note, delete_all_notes],
        system_prompt="You are a research assistant. Search for information and save notes.",
    )

    # --- run() — autonomous loop ---
    print("Using run() (autonomous loop):")
    result = await agent.run("Research quantum computing and save your findings.")
    print(f"  Agent: {result}\n")

    # --- step() — single turn control ---
    print("Using step() (single turn):")
    agent.reset()  # Clear conversation, start fresh
    step_provider = MockProvider(default_text="Here is a quick summary of quantum computing.")
    agent._provider = step_provider

    step_result = await agent.step("Give me a quick summary of quantum computing.")
    print(f"  Agent: {step_result.response.content}")
    print(f"  is_final={step_result.is_final}, turn={step_result.turn_number}")


if __name__ == "__main__":
    asyncio.run(main())
