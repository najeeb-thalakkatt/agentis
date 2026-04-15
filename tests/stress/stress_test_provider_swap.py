"""Stress Test #5: Provider Swap — Same scenario, different backends.

Runs the same SentinelOps incident on:
  1. Anthropic Claude (Haiku — cheapest Claude)
  2. OpenAI GPT-4o
  3. Ollama llama3.2 (local, OpenAI-compatible)

All through the same AgentRuntime with the same tools, hooks, and config.
Surfaces where the provider abstraction leaks.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sential", "sentinelops-simulator"))

from agentis import AgentRuntime, CostTracker, MemoryIndex
from agentis.protocols import ProviderCapabilities
from agentis.runtime.agent import StepResult
from phase2.hooks import build_safety_hooks, reset_investigation_state
from phase2.prompts import SYSTEM_PROMPT, format_alert
from phase2.tools import build_agentis_tools
from simulator.engine import SimulatorEngine
from simulator.scenarios import get_all_scenarios


# ── Result Tracking ────────────────────────────────────────


@dataclass
class ProviderResult:
    """Result from running one provider on the scenario."""

    provider_name: str
    model: str
    completed: bool = False
    error: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    safety_blocks: int = 0
    diagnosis: str = ""
    score: float = 0.0
    cost_usd: float = 0.0
    wall_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


# ── Provider Factory ───────────────────────────────────────


def create_provider(
    provider_type: str,
    anthropic_key: str = "",
    openai_key: str = "",
) -> Any:
    """Create a provider instance by type."""
    if provider_type == "anthropic":
        from agentis.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            model="claude-haiku-4-5-20251001",
            api_key=anthropic_key,
        ), "Anthropic", "claude-haiku-4-5-20251001"

    elif provider_type == "openai":
        from agentis.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model="gpt-4o",
            api_key=openai_key,
        ), "OpenAI", "gpt-4o"

    elif provider_type == "ollama":
        from agentis.providers.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            model="llama3.2",
            api_key="ollama",  # Ollama doesn't need a real key
            base_url="http://localhost:11434/v1",
            capabilities_override=ProviderCapabilities(
                name="ollama_llama3.2",
                supports_tool_use=True,
                supports_streaming=True,
                supports_prompt_caching=False,
                supports_image_input=False,
                max_context_tokens=8_000,  # Conservative for 3B model
                model_tier="weak",
            ),
        ), "Ollama", "llama3.2"

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# ── Run One Provider ───────────────────────────────────────


async def run_with_provider(
    provider_type: str,
    scenario: Any,
    anthropic_key: str = "",
    openai_key: str = "",
    call_delay: float = 1.0,
) -> ProviderResult:
    """Run a scenario with a specific provider."""
    provider, provider_name, model_name = create_provider(
        provider_type, anthropic_key, openai_key,
    )
    result = ProviderResult(provider_name=provider_name, model=model_name)

    sim = SimulatorEngine(scenario, simulate_latency=False, error_rate=0.0)
    tools = build_agentis_tools(sim)
    hooks = build_safety_hooks()
    workspace = tempfile.mkdtemp(prefix=f"swap-{provider_type}-")
    memory = MemoryIndex(workspace=workspace)
    cost_tracker = CostTracker(budget_usd=1.0)

    async def auto_approve(request):
        return True

    runtime = AgentRuntime(
        provider=provider,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        hooks=hooks,
        extensions=[cost_tracker],
        memory=memory,
        approval_callback=auto_approve,
        max_turns=10,
        skeptical=False,
    )

    start = time.time()
    diagnosis_text = ""

    try:
        async for step in runtime.steps(format_alert(scenario.alert)):
            result.llm_calls += 1

            if call_delay > 0 and result.llm_calls > 1:
                await asyncio.sleep(call_delay)

            # Track tools
            for tc in (step.response.tool_calls or []):
                result.tools_used.append(tc.name)
                result.tool_calls += 1

            # Track safety blocks
            for tr in step.tool_results:
                if not tr.success and "blocked" in (tr.error or "").lower():
                    result.safety_blocks += 1

            if step.response.content:
                diagnosis_text += step.response.content + "\n"

            if step.is_final:
                break

        result.completed = True
        result.diagnosis = diagnosis_text.strip()

    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)[:200]}"
        result.diagnosis = diagnosis_text.strip()

    result.wall_time = time.time() - start

    # Score
    if result.diagnosis:
        score_data = sim.score_diagnosis(result.diagnosis)
        result.score = score_data.get("overall_accuracy", 0)

    # Cost
    report = cost_tracker.get_report()
    result.cost_usd = report["total_cost_usd"]
    result.input_tokens = report["total_input_tokens"]
    result.output_tokens = report["total_output_tokens"]

    # Cleanup
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)

    return result


# ── Main ───────────────────────────────────────────────────


async def run_provider_swap_test(
    anthropic_key: str,
    openai_key: str,
) -> None:
    """Run the provider swap stress test."""
    scenarios = get_all_scenarios()
    scenario = scenarios[0]  # First scenario for all providers

    print("=" * 80)
    print("PROVIDER SWAP STRESS TEST")
    print(f"  Scenario: {scenario.name}")
    print(f"  Alert: {scenario.alert.get('title', '?')}")
    print(f"  Same tools, hooks, config — different LLM backend")
    print("=" * 80)

    providers_to_test = []
    if anthropic_key:
        providers_to_test.append(("anthropic", "Anthropic Claude Haiku"))
    if openai_key:
        providers_to_test.append(("openai", "OpenAI GPT-4o"))
    # Always test Ollama (local)
    providers_to_test.append(("ollama", "Ollama llama3.2 (local)"))

    results: list[ProviderResult] = []

    for provider_type, label in providers_to_test:
        print(f"\n{'─' * 80}")
        print(f"Running: {label}")
        print(f"{'─' * 80}")

        reset_investigation_state()

        delay = 2.0 if provider_type in ("anthropic", "openai") else 0.0

        r = await run_with_provider(
            provider_type=provider_type,
            scenario=scenario,
            anthropic_key=anthropic_key,
            openai_key=openai_key,
            call_delay=delay,
        )
        results.append(r)

        status = "COMPLETED" if r.completed else f"FAILED: {r.error}"
        print(f"  Status: {status}")
        print(f"  LLM calls: {r.llm_calls}")
        print(f"  Tool calls: {r.tool_calls} ({', '.join(r.tools_used) or 'none'})")
        print(f"  Safety blocks: {r.safety_blocks}")
        print(f"  Score: {r.score:.0%}")
        print(f"  Cost: ${r.cost_usd:.4f}")
        print(f"  Tokens: {r.input_tokens} in / {r.output_tokens} out")
        print(f"  Time: {r.wall_time:.1f}s")
        if r.diagnosis:
            diag_preview = r.diagnosis[:200].replace("\n", " ")
            print(f"  Diagnosis: {diag_preview}...")

    # ── Comparison ─────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print(f"{'=' * 80}")

    # Header
    names = [r.provider_name for r in results]
    header = f"  {'Metric':<20}" + "".join(f"{n:>15}" for n in names)
    print(f"\n{header}")
    print(f"  {'─' * (20 + 15 * len(results))}")

    # Rows
    def row(label, values, fmt=">15"):
        vals = "".join(f"{v:{fmt}}" for v in values)
        print(f"  {label:<20}{vals}")

    row("Model", [r.model for r in results])
    row("Completed", ["Yes" if r.completed else "No" for r in results])
    row("LLM calls", [str(r.llm_calls) for r in results])
    row("Tool calls", [str(r.tool_calls) for r in results])
    row("Safety blocks", [str(r.safety_blocks) for r in results])
    row("Score", [f"{r.score:.0%}" for r in results])
    row("Cost", [f"${r.cost_usd:.4f}" for r in results])
    row("Time", [f"{r.wall_time:.1f}s" for r in results])
    row("Input tokens", [str(r.input_tokens) for r in results])
    row("Output tokens", [str(r.output_tokens) for r in results])

    if any(r.error for r in results):
        row("Error", [r.error[:30] if r.error else "" for r in results])

    # ── Abstraction Leak Analysis ──────────────────────
    print(f"\n{'─' * 80}")
    print("ABSTRACTION LEAK ANALYSIS")
    print(f"{'─' * 80}")

    leaks: list[str] = []
    completed = [r for r in results if r.completed]
    failed = [r for r in results if not r.completed]

    if failed:
        for r in failed:
            leaks.append(f"PROVIDER FAILURE: {r.provider_name} ({r.model}) — {r.error}")

    if completed:
        # Check if all providers used the same tools
        tool_sets = [set(r.tools_used) for r in completed]
        if len(set(frozenset(s) for s in tool_sets)) > 1:
            leaks.append(
                f"TOOL DIVERGENCE: Providers used different tool sets. "
                + " | ".join(f"{r.provider_name}: {set(r.tools_used)}" for r in completed)
            )

        # Check if any provider had safety blocks the others didn't
        block_counts = [r.safety_blocks for r in completed]
        if len(set(block_counts)) > 1:
            leaks.append(
                f"SAFETY DIVERGENCE: Different block counts. "
                + " | ".join(f"{r.provider_name}: {r.safety_blocks}" for r in completed)
            )

        # Check score variance
        scores = [r.score for r in completed if r.score > 0]
        if scores and (max(scores) - min(scores)) > 0.3:
            leaks.append(
                f"QUALITY VARIANCE: Score spread > 30pp. "
                + " | ".join(f"{r.provider_name}: {r.score:.0%}" for r in completed)
            )

    if leaks:
        print(f"\n  Leaks found: {len(leaks)}\n")
        for i, leak in enumerate(leaks):
            print(f"  [{i+1}] {leak}")
    else:
        print("\n  No abstraction leaks detected.")
        print("  All providers completed, used compatible tools, same safety behavior.")

    # ── Verdict ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    all_completed = all(r.completed for r in results)
    all_same_safety = len(set(r.safety_blocks for r in results if r.completed)) <= 1

    if all_completed and not leaks:
        print("VERDICT: PROVIDER-AGNOSTIC CLAIM VALIDATED.")
        print("  All 3 providers worked through the same AgentRuntime interface.")
    elif all_completed:
        print("VERDICT: PARTIAL — All providers work but with behavioral differences.")
    else:
        failed_names = [r.provider_name for r in failed]
        print(f"VERDICT: ABSTRACTION LEAKS FOUND — {', '.join(failed_names)} failed.")
        print("  The framework does NOT work identically across all providers.")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "sential", "sentinelops-simulator"))

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not anthropic_key and not openai_key:
        print("ERROR: Set at least one of ANTHROPIC_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    asyncio.run(run_provider_swap_test(anthropic_key, openai_key))
