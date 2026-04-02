"""Example 02 — Safety Hooks & Cost Tracking

Demonstrates:
  - HookRegistry with built-in safety hooks (block destructive commands, path sandbox)
  - CostTracker extension with budget limits
  - Human approval callback for DANGEROUS tools
  - steps() async iterator for per-step progress

Run: python examples/02_safety_and_cost.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _mock_provider import MockProvider

from agentis import (
    AgentRuntime,
    ApprovalRequest,
    CostTracker,
    HookRegistry,
    LifecycleEvent,
    Permission,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    tool,
)
from agentis.hooks.builtins import block_destructive_commands, create_path_sandbox

# ── Tools ───────────────────────────────────────────────────


@tool(name="bash", permission=Permission.DANGEROUS)
async def bash_tool(command: str) -> str:
    """Execute a shell command."""
    return f"Output of: {command}"


@tool(name="file_read")
async def file_read_tool(path: str) -> str:
    """Read a file from disk."""
    return f"Contents of {path}"


@tool()
async def safe_search(query: str) -> str:
    """Search the codebase safely."""
    return f"Found 3 matches for '{query}'"


# ── Human approval callback ────────────────────────────────


async def approval_callback(request: ApprovalRequest) -> bool:
    """Auto-deny dangerous operations for this example."""
    print(f"  [approval] Tool '{request.tool_name}' requested: {request.reason}")
    print("  [approval] DENIED (auto-deny in example)")
    return False  # In production, prompt the user


# ── Script the mock LLM ────────────────────────────────────


provider = MockProvider(responses=[
    # Turn 1: Agent tries a destructive bash command
    ProviderResponse(
        content="Let me clean up.",
        tool_calls=[ToolCall(id="t1", name="bash", arguments={"command": "rm -rf /"})],
        usage=TokenUsage(input_tokens=30, output_tokens=15),
        stop_reason="tool_use",
    ),
    # Turn 2: Agent tries to read outside sandbox
    ProviderResponse(
        content="Let me check credentials.",
        tool_calls=[ToolCall(id="t2", name="file_read", arguments={"path": "/etc/passwd"})],
        usage=TokenUsage(input_tokens=40, output_tokens=15),
        stop_reason="tool_use",
    ),
    # Turn 3: Agent tries a safe search
    ProviderResponse(
        content="Let me search safely.",
        tool_calls=[ToolCall(id="t3", name="safe_search", arguments={"query": "auth bug"})],
        usage=TokenUsage(input_tokens=50, output_tokens=15),
        stop_reason="tool_use",
    ),
    # Turn 4: Final answer
    ProviderResponse(
        content="Some operations were blocked by safety policies. "
                "I found 3 matches for the auth bug using the safe search.",
        tool_calls=[],
        usage=TokenUsage(input_tokens=60, output_tokens=30),
        stop_reason="end_turn",
    ),
])


async def main() -> None:
    print("=== 02: Safety Hooks & Cost Tracking ===\n")

    # Set up safety hooks
    hooks = HookRegistry()
    hooks.register(
        LifecycleEvent.PRE_TOOL_USE, block_destructive_commands, name="block_destructive",
    )
    hooks.register(LifecycleEvent.PRE_TOOL_USE, create_path_sandbox(["/tmp/safe"]), name="sandbox")

    # Set up cost tracking with a $1 budget
    cost_tracker = CostTracker(budget_usd=1.00)

    # Create the agent
    agent = AgentRuntime(
        provider=provider,
        tools=[bash_tool, file_read_tool, safe_search],
        hooks=hooks,
        extensions=[cost_tracker],
        approval_callback=approval_callback,
        system_prompt="You are a code maintenance agent.",
    )

    # Run with steps() to see each step
    step_num = 0
    async for step in agent.steps("Clean up the project and find auth bugs"):
        step_num += 1
        blocked = [r for r in step.tool_results if not r.success]
        succeeded = [r for r in step.tool_results if r.success]

        if blocked:
            for r in blocked:
                print(f"  Step {step_num}: BLOCKED - {r.error}")
        if succeeded:
            for r in succeeded:
                print(f"  Step {step_num}: OK - {r.summary}")
        if step.is_final:
            print(f"  Step {step_num}: FINAL - {step.response.content[:80]}...")

    # Print cost report
    print("\n--- Cost Report ---")
    report = cost_tracker.get_report()
    print(f"  Total cost:   ${report['total_cost_usd']:.6f}")
    print(f"  Input tokens:  {report['total_input_tokens']}")
    print(f"  Output tokens: {report['total_output_tokens']}")
    print(f"  Turns:         {report['num_turns']}")
    print(f"  Over budget:   {report['over_budget']}")


if __name__ == "__main__":
    asyncio.run(main())
