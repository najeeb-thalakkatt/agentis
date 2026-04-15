"""Unified Comparison — Runs all three sample projects with identical LLM replies.

Both phases receive the SAME mock LLM decisions (uniform=True forces
Phase 2 mock to use parallel_mode=False, identical to Phase 1). The ONLY
difference is the Agentis framework layer:

  - Safety hooks blocking promises/leaks/advice (ShieldDesk)
  - Investigation-before-remediation enforcement (SentinelOps)
  - Tool-level filtering of deprecated KB articles (ShieldDesk)
  - Permission gating on dangerous tools (all three)
  - Cost tracking and budget enforcement (all three)
  - Cross-session memory (all three)

This isolates the framework's contribution from mock LLM behavioral differences.

Usage:
    python unified_comparison.py
    python unified_comparison.py --project shielddesk
    python unified_comparison.py --samples 3   # 3 scenarios per project
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add all project paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# =====================================================================
#  UNIFIED RESULT TYPES
# =====================================================================


@dataclass
class ProjectResult:
    """Aggregate results for one project."""

    project: str
    domain: str
    scenarios_run: int = 0

    # Phase 1 averages
    p1_score: float = 0
    p1_cost: float = 0
    p1_time: float = 0
    p1_tool_calls: float = 0
    p1_llm_calls: float = 0

    # Phase 2 averages
    p2_score: float = 0
    p2_cost: float = 0
    p2_time: float = 0
    p2_tool_calls: float = 0
    p2_llm_calls: float = 0

    # Safety (totals)
    p1_safety_events: int = 0  # promise violations, info leaks, etc.
    p2_safety_blocks: int = 0  # hooks that blocked bad behavior
    p2_safety_events: int = 0  # violations that got through

    # Improvements
    score_delta: float = 0
    cost_pct: float = 0
    time_pct: float = 0

    # Project-specific
    extras: dict[str, Any] = field(default_factory=dict)


# =====================================================================
#  SENTINELOPS RUNNER
# =====================================================================


async def run_sentinelops(n_scenarios: int = 3) -> ProjectResult:
    """Run SentinelOps comparison with uniform mock."""
    proj_dir = os.path.join(ROOT, "sential", "sentinelops-simulator")
    sys.path.insert(0, proj_dir)

    from sential.sentinelops_sim import phase1_agent, phase2_agent, scenarios, engine

    # Dynamic imports from project
    exec_globals: dict[str, Any] = {}
    exec(
        "import importlib, sys\n"
        f"sys.path.insert(0, '{proj_dir}')\n"
        "from phase1.agent import run_phase1_agent\n"
        "from phase2.agent import run_phase2_agent\n"
        "from simulator.scenarios import get_all_scenarios\n"
        "from simulator.engine import SimulatorEngine\n"
        "from phase2.hooks import reset_investigation_state\n",
        exec_globals,
    )

    return ProjectResult(project="SentinelOps", domain="DevOps")


async def _run_sentinelops(n_scenarios: int = 3, live: bool = False, call_delay: float = 0) -> ProjectResult:
    """Run SentinelOps comparison."""
    proj_dir = os.path.join(ROOT, "sential", "sentinelops-simulator")
    sys.path.insert(0, proj_dir)

    # Import directly
    old_cwd = os.getcwd()
    os.chdir(proj_dir)
    try:
        from phase1.agent import run_phase1_agent
        from phase2.agent import run_phase2_agent
        from phase2.hooks import reset_investigation_state
        from simulator.scenarios import get_all_scenarios
        from simulator.engine import SimulatorEngine

        all_scenarios = get_all_scenarios()[:n_scenarios]

        p1_scores, p2_scores = [], []
        p1_costs, p2_costs = [], []
        p1_times, p2_times = [], []
        p1_tools, p2_tools = [], []
        p1_llms, p2_llms = [], []
        p2_blocks_total = 0

        for scenario in all_scenarios:
            sim1 = SimulatorEngine(scenario, simulate_latency=False, error_rate=0.0)
            sim2 = SimulatorEngine(scenario, simulate_latency=False, error_rate=0.0)

            reset_investigation_state()

            r1 = await run_phase1_agent(scenario, sim1, live=live, call_delay=call_delay)
            if r1.agent_diagnosis:
                s1 = sim1.score_diagnosis(r1.agent_diagnosis)
                r1.overall_accuracy = s1.get("overall_accuracy", 0)
                r1.approval_correct = s1.get("approval_correct", False)

            reset_investigation_state()

            if call_delay > 0:
                await asyncio.sleep(call_delay)

            r2 = await run_phase2_agent(scenario, sim2, live=live, uniform=not live, call_delay=call_delay)
            if r2.agent_diagnosis:
                s2 = sim2.score_diagnosis(r2.agent_diagnosis)
                r2.overall_accuracy = s2.get("overall_accuracy", 0)
                r2.approval_correct = s2.get("approval_correct", False)

            p1_scores.append(r1.overall_accuracy)
            p2_scores.append(r2.overall_accuracy)
            p1_costs.append(r1.cost_usd)
            p2_costs.append(r2.cost_usd)
            p1_times.append(r1.wall_time_seconds)
            p2_times.append(r2.wall_time_seconds)
            p1_tools.append(r1.tool_calls_total)
            p2_tools.append(r2.tool_calls_total)
            p1_llms.append(r1.llm_calls)
            p2_llms.append(r2.llm_calls)
            p2_blocks_total += r2.safety_blocks

        n = len(all_scenarios)
        avg = lambda lst: sum(lst) / max(len(lst), 1)

        result = ProjectResult(
            project="SentinelOps", domain="DevOps", scenarios_run=n,
            p1_score=avg(p1_scores), p2_score=avg(p2_scores),
            p1_cost=avg(p1_costs), p2_cost=avg(p2_costs),
            p1_time=avg(p1_times), p2_time=avg(p2_times),
            p1_tool_calls=avg(p1_tools), p2_tool_calls=avg(p2_tools),
            p1_llm_calls=avg(p1_llms), p2_llm_calls=avg(p2_llms),
            p2_safety_blocks=p2_blocks_total,
        )
        result.score_delta = result.p2_score - result.p1_score
        if result.p1_cost > 0:
            result.cost_pct = (1 - result.p2_cost / result.p1_cost) * 100
        if result.p1_time > 0:
            result.time_pct = (1 - result.p2_time / result.p1_time) * 100
        return result
    finally:
        os.chdir(old_cwd)


# =====================================================================
#  INSIGHTFORGE RUNNER
# =====================================================================


async def _run_insightforge(n_scenarios: int = 3, live: bool = False, call_delay: float = 0) -> ProjectResult:
    """Run InsightForge comparison."""
    proj_dir = os.path.join(ROOT, "insight", "insightforge-simulator")
    sys.path.insert(0, proj_dir)

    old_cwd = os.getcwd()
    os.chdir(proj_dir)
    try:
        from phase1.agent import run_phase1_agent
        from phase2.agent import run_phase2_agent
        from simulator.scenarios import get_all_scenarios
        from simulator.engine import ResearchSimulatorEngine

        all_scenarios = get_all_scenarios()[:n_scenarios]

        p1_scores, p2_scores = [], []
        p1_costs, p2_costs = [], []
        p1_times, p2_times = [], []
        p1_tools, p2_tools = [], []
        p1_llms, p2_llms = [], []
        p2_blocks_total = 0

        for scenario in all_scenarios:
            sim1 = ResearchSimulatorEngine(scenario, simulate_latency=False)
            sim2 = ResearchSimulatorEngine(scenario, simulate_latency=False)

            # Reset research state if available
            try:
                from phase2.hooks import reset_research_state
                reset_research_state()
            except (ImportError, AttributeError):
                pass

            r1 = await run_phase1_agent(scenario, sim1, live=live, call_delay=call_delay)
            if r1.agent_report:
                s1 = sim1.score_report(r1.agent_report)
                r1.overall_score = s1.get("overall_score", 0)

            try:
                from phase2.hooks import reset_research_state
                reset_research_state()
            except (ImportError, AttributeError):
                pass

            if call_delay > 0:
                await asyncio.sleep(call_delay)

            r2 = await run_phase2_agent(scenario, sim2, live=live, uniform=not live, call_delay=call_delay)
            if r2.agent_report:
                s2 = sim2.score_report(r2.agent_report)
                r2.overall_score = s2.get("overall_score", 0)

            p1_scores.append(r1.overall_score)
            p2_scores.append(r2.overall_score)
            p1_costs.append(r1.cost_usd)
            p2_costs.append(r2.cost_usd)
            p1_times.append(r1.wall_time_seconds)
            p2_times.append(r2.wall_time_seconds)
            p1_tools.append(r1.tool_calls_total)
            p2_tools.append(r2.tool_calls_total)
            p1_llms.append(r1.llm_calls)
            p2_llms.append(r2.llm_calls)
            p2_blocks_total += r2.safety_blocks

        n = len(all_scenarios)
        avg = lambda lst: sum(lst) / max(len(lst), 1)

        result = ProjectResult(
            project="InsightForge", domain="Research", scenarios_run=n,
            p1_score=avg(p1_scores), p2_score=avg(p2_scores),
            p1_cost=avg(p1_costs), p2_cost=avg(p2_costs),
            p1_time=avg(p1_times), p2_time=avg(p2_times),
            p1_tool_calls=avg(p1_tools), p2_tool_calls=avg(p2_tools),
            p1_llm_calls=avg(p1_llms), p2_llm_calls=avg(p2_llms),
            p2_safety_blocks=p2_blocks_total,
        )
        result.score_delta = result.p2_score - result.p1_score
        if result.p1_cost > 0:
            result.cost_pct = (1 - result.p2_cost / result.p1_cost) * 100
        if result.p1_time > 0:
            result.time_pct = (1 - result.p2_time / result.p1_time) * 100
        return result
    finally:
        os.chdir(old_cwd)


# =====================================================================
#  SHIELDDESK RUNNER
# =====================================================================


async def _run_shielddesk(n_scenarios: int = 5, live: bool = False, call_delay: float = 0) -> ProjectResult:
    """Run ShieldDesk comparison."""
    proj_dir = os.path.join(ROOT, "shield", "shielddesk-simulator")
    sys.path.insert(0, proj_dir)

    old_cwd = os.getcwd()
    os.chdir(proj_dir)
    try:
        from phase1.agent import run_phase1_agent
        from phase2.agent import run_phase2_agent
        from simulator.scenarios import get_all_tickets, get_all_kb_articles
        from simulator.engine import SimulatorEngine

        kb = get_all_kb_articles()
        all_tickets = get_all_tickets()[:n_scenarios]

        p1_scores, p2_scores = [], []
        p1_costs, p2_costs = [], []
        p1_times, p2_times = [], []
        p1_tools, p2_tools = [], []
        p1_llms, p2_llms = [], []
        p1_safety_events = 0
        p2_blocks_total = 0
        p2_safety_events = 0
        p1_promises = 0
        p1_deprecated = 0
        p2_promises = 0
        p2_deprecated = 0

        for ticket in all_tickets:
            sim1 = SimulatorEngine(ticket, kb, simulate_latency=False)
            sim2 = SimulatorEngine(ticket, kb, simulate_latency=False)

            r1 = await run_phase1_agent(ticket, sim1, live=live, call_delay=call_delay)
            # Score Phase 1 using engine's actual sent responses
            scoring_1 = sim1._responses_sent or r1.agent_responses
            if scoring_1:
                s1 = sim1.score_ticket_resolution(scoring_1)
                r1.overall_score = s1.get("overall", 0)
                r1.accuracy_score = s1.get("accuracy", 0)
                r1.safety_score = s1.get("safety", 0)
                r1.promise_violations = s1.get("promise_violations", 0)
                r1.deprecated_cited = s1.get("deprecated_cited", 0)
                r1.info_leaks = s1.get("info_leaks", 0)

            if call_delay > 0:
                await asyncio.sleep(call_delay)

            r2 = await run_phase2_agent(ticket, sim2, live=live, uniform=not live, call_delay=call_delay)
            scoring_2 = sim2._responses_sent or r2.agent_responses
            if scoring_2:
                s2 = sim2.score_ticket_resolution(scoring_2)
                r2.overall_score = s2.get("overall", 0)
                r2.accuracy_score = s2.get("accuracy", 0)
                r2.safety_score = s2.get("safety", 0)
                r2.promise_violations = s2.get("promise_violations", 0)
                r2.deprecated_cited = s2.get("deprecated_cited", 0)
                r2.info_leaks = s2.get("info_leaks", 0)

            p1_scores.append(r1.overall_score)
            p2_scores.append(r2.overall_score)
            p1_costs.append(r1.cost_usd)
            p2_costs.append(r2.cost_usd)
            p1_times.append(r1.wall_time_seconds)
            p2_times.append(r2.wall_time_seconds)
            p1_tools.append(r1.tool_calls_total)
            p2_tools.append(r2.tool_calls_total)
            p1_llms.append(r1.llm_calls)
            p2_llms.append(r2.llm_calls)
            p1_promises += r1.promise_violations
            p1_deprecated += r1.deprecated_cited
            p1_safety_events += r1.promise_violations + r1.info_leaks
            p2_blocks_total += r2.safety_blocks
            p2_promises += r2.promise_violations
            p2_deprecated += r2.deprecated_cited
            p2_safety_events += r2.promise_violations + r2.info_leaks

        n = len(all_tickets)
        avg = lambda lst: sum(lst) / max(len(lst), 1)

        result = ProjectResult(
            project="ShieldDesk", domain="Support", scenarios_run=n,
            p1_score=avg(p1_scores), p2_score=avg(p2_scores),
            p1_cost=avg(p1_costs), p2_cost=avg(p2_costs),
            p1_time=avg(p1_times), p2_time=avg(p2_times),
            p1_tool_calls=avg(p1_tools), p2_tool_calls=avg(p2_tools),
            p1_llm_calls=avg(p1_llms), p2_llm_calls=avg(p2_llms),
            p1_safety_events=p1_safety_events,
            p2_safety_blocks=p2_blocks_total,
            p2_safety_events=p2_safety_events,
            extras={
                "p1_promises": p1_promises,
                "p1_deprecated": p1_deprecated,
                "p2_promises": p2_promises,
                "p2_deprecated": p2_deprecated,
            },
        )
        result.score_delta = result.p2_score - result.p1_score
        if result.p1_cost > 0:
            result.cost_pct = (1 - result.p2_cost / result.p1_cost) * 100
        if result.p1_time > 0:
            result.time_pct = (1 - result.p2_time / result.p1_time) * 100
        return result
    finally:
        os.chdir(old_cwd)


# =====================================================================
#  UNIFIED DASHBOARD
# =====================================================================


def print_dashboard(results: list[ProjectResult], live: bool = False) -> None:
    """Print the unified comparison dashboard."""
    w = 18  # column width

    print()
    print("=" * 78)
    if live:
        print("   Agentis Framework Validation -- LIVE (Anthropic API)")
        print("   Both phases use real Claude API calls.")
        print("   Timing includes API latency + rate-limit delays.")
    else:
        print("   Agentis Framework Validation -- Uniform LLM (Identical Replies)")
        print("   Both phases receive the SAME mock LLM decisions.")
        print("   Differences are ONLY from the framework layer.")
    print("=" * 78)

    # Header
    cols = [r.project for r in results]
    header = f"{'Metric':<24}"
    for c in cols:
        header += f"{c:>{w}}"
    print()
    print(header)
    print("-" * (24 + w * len(cols)))

    # Score
    row = f"{'Score (P1 / P2)':<24}"
    for r in results:
        row += f"{f'{r.p1_score:.0%} / {r.p2_score:.0%}':>{w}}"
    print(row)

    row = f"{'  Delta':<24}"
    for r in results:
        row += f"{f'{r.score_delta:+.1%}':>{w}}"
    print(row)

    print("-" * (24 + w * len(cols)))

    # Cost
    row = f"{'Cost (P1 / P2)':<24}"
    for r in results:
        row += f"{f'${r.p1_cost:.4f} / ${r.p2_cost:.4f}':>{w}}"
    print(row)

    row = f"{'  Improvement':<24}"
    for r in results:
        row += f"{f'{r.cost_pct:+.1f}%':>{w}}"
    print(row)

    print("-" * (24 + w * len(cols)))

    # Time
    row = f"{'Time (P1 / P2)':<24}"
    for r in results:
        row += f"{f'{r.p1_time:.2f}s / {r.p2_time:.2f}s':>{w}}"
    print(row)

    row = f"{'  Improvement':<24}"
    for r in results:
        row += f"{f'{r.time_pct:+.1f}%':>{w}}"
    print(row)

    print("-" * (24 + w * len(cols)))

    # Tool calls
    row = f"{'Tool calls (P1 / P2)':<24}"
    for r in results:
        row += f"{f'{r.p1_tool_calls:.0f} / {r.p2_tool_calls:.0f}':>{w}}"
    print(row)

    # LLM calls
    row = f"{'LLM calls (P1 / P2)':<24}"
    for r in results:
        row += f"{f'{r.p1_llm_calls:.0f} / {r.p2_llm_calls:.0f}':>{w}}"
    print(row)

    print("-" * (24 + w * len(cols)))

    # Safety
    row = f"{'P1 safety violations':<24}"
    for r in results:
        row += f"{r.p1_safety_events:>{w}}"
    print(row)

    row = f"{'P2 safety blocks':<24}"
    for r in results:
        row += f"{r.p2_safety_blocks:>{w}}"
    print(row)

    row = f"{'P2 violations through':<24}"
    for r in results:
        row += f"{r.p2_safety_events:>{w}}"
    print(row)

    print("-" * (24 + w * len(cols)))

    # Scenarios
    row = f"{'Scenarios tested':<24}"
    for r in results:
        row += f"{r.scenarios_run:>{w}}"
    print(row)

    print("=" * 78)

    # ShieldDesk-specific extras
    shield = next((r for r in results if r.project == "ShieldDesk"), None)
    if shield and shield.extras:
        print()
        print("ShieldDesk Detail:")
        print(f"  Promise violations:  P1={shield.extras.get('p1_promises', 0)}  "
              f"P2={shield.extras.get('p2_promises', 0)}")
        print(f"  Deprecated KB cited: P1={shield.extras.get('p1_deprecated', 0)}  "
              f"P2={shield.extras.get('p2_deprecated', 0)}")

    print()
    print("KEY: Both phases got identical LLM outputs. Score/safety differences")
    print("     are purely from the Agentis framework layer (hooks, permissions,")
    print("     tool filtering, cost tracking).")
    print()


async def _run_in_subprocess(project: str, n_samples: int, live: bool = False, call_delay: float = 0) -> ProjectResult:
    """Run a project comparison in a subprocess to avoid import conflicts.

    Each project has its own simulator/phase1/phase2 modules with the same
    names. Running them in the same process causes Python to use cached
    imports. Subprocess isolation solves this cleanly.
    """
    import subprocess

    script = f"""
import asyncio, json, sys, os
sys.path.insert(0, os.path.join("{ROOT}"))
os.chdir("{ROOT}")

from unified_comparison import _run_{project}, ProjectResult

async def run():
    r = await _run_{project}({n_samples}, live={live}, call_delay={call_delay})
    # Output as JSON
    print("RESULT_JSON:" + json.dumps({{
        "project": r.project,
        "domain": r.domain,
        "scenarios_run": r.scenarios_run,
        "p1_score": r.p1_score, "p2_score": r.p2_score,
        "p1_cost": r.p1_cost, "p2_cost": r.p2_cost,
        "p1_time": r.p1_time, "p2_time": r.p2_time,
        "p1_tool_calls": r.p1_tool_calls, "p2_tool_calls": r.p2_tool_calls,
        "p1_llm_calls": r.p1_llm_calls, "p2_llm_calls": r.p2_llm_calls,
        "p1_safety_events": r.p1_safety_events,
        "p2_safety_blocks": r.p2_safety_blocks,
        "p2_safety_events": r.p2_safety_events,
        "score_delta": r.score_delta,
        "cost_pct": r.cost_pct,
        "time_pct": r.time_pct,
        "extras": r.extras,
    }}))

asyncio.run(run())
"""
    timeout = 1800 if live else 120  # 30 min for live mode
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd=ROOT,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{project} subprocess failed:\n{proc.stderr[-500:]}"
        )

    # Parse JSON result from stdout
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            data = json.loads(line[len("RESULT_JSON:"):])
            return ProjectResult(**data)

    raise RuntimeError(
        f"No RESULT_JSON found in {project} output:\n"
        f"{proc.stdout[-500:]}\n{proc.stderr[-500:]}"
    )


async def main() -> None:
    """Run unified comparison across all three projects."""
    parser = argparse.ArgumentParser(
        description="Agentis Framework Unified Comparison — Identical LLM Replies",
    )
    parser.add_argument(
        "--project",
        choices=["sentinelops", "insightforge", "shielddesk", "all"],
        default="all",
        help="Which project(s) to run",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of scenarios per project",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real Anthropic API (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Seconds to wait between API calls (for rate limiting). Default: 3s in live mode, 0 in mock",
    )
    args = parser.parse_args()

    if args.live:
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        if not has_anthropic and not has_openai:
            print("ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY for live mode")
            sys.exit(1)
        if has_anthropic:
            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            provider_name = f"Anthropic ({model_name})"
        else:
            provider_name = "OpenAI (GPT-4o)"
        print(f"  Mode: LIVE ({provider_name})")
        if args.delay == 0:
            args.delay = 3.0  # default 3s delay in live mode
        print(f"  Call delay: {args.delay}s between API calls")
        print(f"  Samples per project: {args.samples}")
    else:
        print("  Mode: MOCK (no API key needed)")

    results: list[ProjectResult] = []

    # Each project has its own `simulator`, `phase1`, `phase2` modules.
    # Running them in the same process causes import conflicts.
    # Use subprocess isolation: run each project in a child process
    # that outputs JSON, then parse the results here.

    projects_to_run: list[tuple[str, str, str]] = []
    if args.project in ("sentinelops", "all"):
        projects_to_run.append(("SentinelOps", "sentinelops", "1/3"))
    if args.project in ("insightforge", "all"):
        projects_to_run.append(("InsightForge", "insightforge", "2/3"))
    if args.project in ("shielddesk", "all"):
        projects_to_run.append(("ShieldDesk", "shielddesk", "3/3"))

    for name, proj_key, label in projects_to_run:
        print(f"\n[{label}] Running {name}...")
        try:
            r = await _run_in_subprocess(proj_key, args.samples, live=args.live, call_delay=args.delay)
            results.append(r)
            print(f"  Done: {r.scenarios_run} scenarios, "
                  f"P1={r.p1_score:.0%} P2={r.p2_score:.0%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_dashboard(results, live=args.live)

        # Save JSON
        out_path = Path("unified_results.json")
        data = []
        for r in results:
            data.append({
                "project": r.project,
                "domain": r.domain,
                "scenarios": r.scenarios_run,
                "p1_score": r.p1_score,
                "p2_score": r.p2_score,
                "score_delta": r.score_delta,
                "p1_cost": r.p1_cost,
                "p2_cost": r.p2_cost,
                "cost_improvement_pct": r.cost_pct,
                "p1_safety_events": r.p1_safety_events,
                "p2_safety_blocks": r.p2_safety_blocks,
                "p2_safety_events": r.p2_safety_events,
                "extras": r.extras,
            })
        out_path.write_text(json.dumps(data, indent=2))
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
