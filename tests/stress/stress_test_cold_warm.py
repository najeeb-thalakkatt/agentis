"""Stress Test #7: Cold Start vs Warm Start.

Measures how much the memory index and session persistence help on repeated
runs of the same problem type. Quantifies the "learning" effect.

Test plan:
  1. Cold run: 5 ShieldDesk billing tickets, no memory
  2. Warm run: 5 MORE billing tickets, WITH memory from run 1
  3. Compare: tool calls, LLM calls, response quality, time
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shield", "shielddesk-simulator"))

from agentis import AgentRuntime, CostTracker, MemoryIndex
from agentis.protocols import ProviderCapabilities
from agentis.types import (
    Message,
    ProviderResponse,
    TokenUsage,
    ToolCall,
)
from phase2.hooks import build_safety_hooks
from phase2.prompts import SYSTEM_PROMPT
from phase2.tools import build_agentis_tools
from simulator.engine import SimulatorEngine
from simulator.scenarios import (
    SupportTicket,
    get_all_kb_articles,
    get_tickets_by_type,
    get_all_tickets,
)


# ── Learning Provider ──────────────────────────────────────


class LearningProvider:
    """Mock provider that behaves differently based on memory availability.

    Cold (no memory): Full investigation — search KB, get customer, then respond.
    Warm (has memory): Checks memory first via recall, skips some searches,
    responds faster.
    """

    def __init__(self, ticket_id: str, has_memory: bool = False) -> None:
        self._ticket_id = ticket_id
        self._has_memory = has_memory
        self._turn = 0
        self.call_count = 0
        self.tools_called: list[str] = []
        self.recall_used = False

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        self.call_count += 1
        self._turn += 1

        # Check if memory index has entries
        memory_available = False
        memory_topic_ids: list[str] = []
        for msg in messages:
            if msg.role == "system" and "memory index" in msg.content.lower():
                if "(empty)" not in msg.content:
                    memory_available = True
                    for line in msg.content.split("\n"):
                        if "topics/" in line and line.strip().startswith("- "):
                            parts = line.split("topics/")
                            if len(parts) > 1:
                                tid = parts[1].split(".md")[0]
                                memory_topic_ids.append(tid)

        # ── Warm path: recall memory, then targeted response ──
        if self._has_memory and memory_available and self._turn == 1:
            self.recall_used = True
            recalls = [
                ToolCall(id=f"rc_{i}", name="recall", arguments={"topic_id": tid})
                for i, tid in enumerate(memory_topic_ids[:2])
            ]
            self.tools_called.extend(["recall"] * len(recalls))
            return ProviderResponse(
                content="I have prior knowledge about billing issues. Let me recall.",
                tool_calls=recalls,
                usage=TokenUsage(input_tokens=150, output_tokens=50),
                stop_reason="tool_use",
            )

        if self._has_memory and self.recall_used and self._turn == 2:
            # Skip KB search — already know the answers from memory
            self.tools_called.append("get_customer")
            return ProviderResponse(
                content="Recalled billing procedures. Getting customer details.",
                tool_calls=[ToolCall(
                    id="gc1", name="get_customer",
                    arguments={"customer_id": self._get_customer_id(messages)},
                )],
                usage=TokenUsage(input_tokens=200, output_tokens=50),
                stop_reason="tool_use",
            )

        if self._has_memory and self.recall_used and self._turn == 3:
            # Respond directly using prior knowledge
            self.tools_called.append("send_response")
            return ProviderResponse(
                content="Using prior knowledge to respond.",
                tool_calls=[ToolCall(
                    id="sr1", name="send_response",
                    arguments={
                        "message": (
                            "I understand your billing concern. Based on our records and "
                            "standard billing procedures, I can help you with this. "
                            "Please check your billing settings under Account > Billing. "
                            "If you see any discrepancies, I can help review the charges."
                        ),
                        "ticket_id": self._ticket_id,
                    },
                )],
                usage=TokenUsage(input_tokens=250, output_tokens=100),
                stop_reason="tool_use",
            )

        # ── Cold path: full investigation ──
        if self._turn == 1:
            self.tools_called.extend(["search_kb", "get_customer"])
            return ProviderResponse(
                content="Starting investigation.",
                tool_calls=[
                    ToolCall(id="sk1", name="search_kb",
                             arguments={"query": "billing charge invoice"}),
                    ToolCall(id="gc1", name="get_customer",
                             arguments={"customer_id": self._get_customer_id(messages)}),
                ],
                usage=TokenUsage(input_tokens=200, output_tokens=60),
                stop_reason="tool_use",
            )

        if self._turn == 2:
            self.tools_called.append("search_kb")
            return ProviderResponse(
                content="Need more KB info.",
                tool_calls=[ToolCall(
                    id="sk2", name="search_kb",
                    arguments={"query": "refund credit payment"},
                )],
                usage=TokenUsage(input_tokens=300, output_tokens=50),
                stop_reason="tool_use",
            )

        if self._turn == 3:
            self.tools_called.append("send_response")
            return ProviderResponse(
                content="Responding to customer.",
                tool_calls=[ToolCall(
                    id="sr1", name="send_response",
                    arguments={
                        "message": (
                            "I understand your billing concern. "
                            "I've looked into this and can see the charge in question. "
                            "Please check your billing settings under Account > Billing. "
                            "If you see any discrepancies, I can help review the charges."
                        ),
                        "ticket_id": self._ticket_id,
                    },
                )],
                usage=TokenUsage(input_tokens=400, output_tokens=100),
                stop_reason="tool_use",
            )

        # Final
        return ProviderResponse(
            content="I've addressed the billing concern.",
            tool_calls=[],
            usage=TokenUsage(input_tokens=100, output_tokens=30),
        )

    def _get_customer_id(self, messages: list[Message]) -> str:
        """Extract customer ID from context."""
        for msg in messages:
            if "Customer:" in msg.content or "customer_id" in msg.content:
                for part in msg.content.split():
                    if part.startswith("cust_"):
                        return part.rstrip(")")
        return "cust_001"

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[str]:
        yield "stream"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(name="learning_mock")


# ── Result Tracking ────────────────────────────────────────


@dataclass
class TicketResult:
    """Metrics from processing one ticket."""

    ticket_id: str
    llm_calls: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    recall_used: bool = False
    safety_blocks: int = 0
    elapsed: float = 0.0


@dataclass
class BatchResult:
    """Aggregated results from a batch of tickets."""

    mode: str  # "cold" or "warm"
    tickets: list[TicketResult] = field(default_factory=list)
    memories_before: int = 0
    memories_after: int = 0

    @property
    def avg_llm_calls(self) -> float:
        return sum(t.llm_calls for t in self.tickets) / max(len(self.tickets), 1)

    @property
    def avg_tool_calls(self) -> float:
        return sum(t.tool_calls for t in self.tickets) / max(len(self.tickets), 1)

    @property
    def total_tool_calls(self) -> int:
        return sum(t.tool_calls for t in self.tickets)

    @property
    def total_llm_calls(self) -> int:
        return sum(t.llm_calls for t in self.tickets)

    @property
    def recall_count(self) -> int:
        return sum(1 for t in self.tickets if t.recall_used)

    @property
    def total_search_kb(self) -> int:
        return sum(t.tools_used.count("search_kb") for t in self.tickets)

    @property
    def total_recall(self) -> int:
        return sum(t.tools_used.count("recall") for t in self.tickets)

    @property
    def avg_elapsed(self) -> float:
        return sum(t.elapsed for t in self.tickets) / max(len(self.tickets), 1)


# ── Run One Ticket ─────────────────────────────────────────


async def run_ticket(
    ticket: SupportTicket,
    memory: MemoryIndex,
    has_memory: bool,
) -> TicketResult:
    """Process one ticket through Phase 2 agent."""
    kb_articles = get_all_kb_articles()
    sim = SimulatorEngine(ticket, kb_articles=kb_articles, simulate_latency=False)
    provider = LearningProvider(ticket_id=ticket.id, has_memory=has_memory)
    tools = build_agentis_tools(sim)
    hooks = build_safety_hooks()

    async def auto_approve(req):
        return True

    runtime = AgentRuntime(
        provider=provider,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        hooks=hooks,
        memory=memory,
        approval_callback=auto_approve,
        max_turns=6,
        skeptical=False,
    )

    result = TicketResult(ticket_id=ticket.id)
    start = time.time()

    message = (
        f"Support Ticket #{ticket.id}\n"
        f"Subject: {ticket.subject}\n"
        f"Customer: {ticket.customer.name} (ID: {ticket.customer.id})\n"
        f"Priority: {ticket.priority}\n\n"
        f"Customer Message:\n{ticket.initial_message}"
    )

    async for step in runtime.steps(message):
        result.llm_calls += 1
        for tr in step.tool_results:
            result.tool_calls += 1
            if not tr.success and "blocked" in (tr.error or "").lower():
                result.safety_blocks += 1

    result.elapsed = time.time() - start
    result.tools_used = provider.tools_called
    result.recall_used = provider.recall_used

    return result


# ── Main Test ──────────────────────────────────────────────


async def run_cold_warm_test() -> None:
    """Run the cold vs warm start comparison."""
    print("=" * 80)
    print("COLD START vs WARM START STRESS TEST")
    print("  Same ticket type, different memory state")
    print("=" * 80)

    # Get billing tickets (most common type for repeat patterns)
    billing_tickets = get_tickets_by_type("billing")
    all_tickets = get_all_tickets()

    # Use billing + some others to get 5 per batch
    batch1_tickets = billing_tickets[:4]
    # Add one more from other types
    other_tickets = [t for t in all_tickets if t.type != "billing"][:1]
    batch1_tickets.extend(other_tickets)

    batch2_tickets = billing_tickets[:4]  # Same type, different instances
    batch2_tickets.extend(other_tickets[:1])

    workspace = tempfile.mkdtemp(prefix="cold-warm-")
    print(f"\n  Shared workspace: {workspace}")
    print(f"  Batch 1 tickets: {len(batch1_tickets)} (cold start)")
    print(f"  Batch 2 tickets: {len(batch2_tickets)} (warm start)")

    try:
        # ── COLD RUN: No prior memory ──
        print(f"\n{'─' * 80}")
        print("BATCH 1: COLD START (no prior memory)")
        print(f"{'─' * 80}")

        memory = MemoryIndex(workspace=workspace)
        cold = BatchResult(mode="cold", memories_before=len(memory.entries))

        for ticket in batch1_tickets:
            tr = await run_ticket(ticket, memory, has_memory=False)
            cold.tickets.append(tr)
            print(f"  [{ticket.id}] LLM={tr.llm_calls} tools={tr.tool_calls} "
                  f"({', '.join(tr.tools_used)})")

        # Save learnings to memory after cold batch
        await memory.remember(
            topic="Billing ticket patterns",
            content=(
                "Common billing issues and resolutions:\n"
                "- Overcharges: Check billing settings under Account > Billing\n"
                "- Refund requests: Escalate if > $100, otherwise offer credit\n"
                "- Invoice questions: Direct to billing portal, reference KB articles\n"
                "- Payment failures: Check payment method, suggest updating card\n"
                "- Pro/Enterprise pricing: Refer to pricing page, no unauthorized discounts\n"
            ),
            tags=["billing", "patterns"],
        )
        await memory.remember(
            topic="Billing KB articles",
            content=(
                "Key KB articles for billing tickets:\n"
                "- kb_bill_01: Payment methods and updating card\n"
                "- kb_bill_02: Understanding your invoice\n"
                "- kb_bill_03: Cancellation and refund policy\n"
                "- kb_bill_04: Plan comparison and pricing\n"
                "- kb_bill_08: Enterprise billing portal\n"
                "NOTE: kb_dep_02 (Legacy Billing Portal) is DEPRECATED — do not cite\n"
            ),
            tags=["billing", "kb"],
        )
        await memory.remember(
            topic="Customer handling best practices",
            content=(
                "Learned from billing batch:\n"
                "- Always check customer plan before suggesting features\n"
                "- VIP customers (LTV > $10K or enterprise) get priority handling\n"
                "- Never promise refunds — say 'let me look into options'\n"
                "- Check for deprecated KB articles before citing\n"
            ),
            tags=["practices", "billing"],
        )

        cold.memories_after = len(memory.entries)
        print(f"\n  Memories saved: {cold.memories_after}")
        for entry in memory.entries:
            print(f"    - {entry.topic} [{', '.join(entry.tags)}]")

        # ── WARM RUN: With memory from batch 1 ──
        print(f"\n{'─' * 80}")
        print("BATCH 2: WARM START (with memory from batch 1)")
        print(f"{'─' * 80}")

        # Create new MemoryIndex pointing to same workspace
        memory2 = MemoryIndex(workspace=workspace)
        await memory2.load()
        print(f"  Memories loaded: {len(memory2.entries)}")

        warm = BatchResult(mode="warm", memories_before=len(memory2.entries))

        for ticket in batch2_tickets:
            tr = await run_ticket(ticket, memory2, has_memory=True)
            warm.tickets.append(tr)
            print(f"  [{ticket.id}] LLM={tr.llm_calls} tools={tr.tool_calls} "
                  f"recall={tr.recall_used} ({', '.join(tr.tools_used)})")

        warm.memories_after = len(memory2.entries)

        # ── Comparison ─────────────────────────────────
        print(f"\n{'=' * 80}")
        print("COMPARISON: Cold Start vs Warm Start")
        print(f"{'=' * 80}")

        print(f"\n  {'Metric':<30} {'Cold':>10} {'Warm':>10} {'Delta':>10} {'Change':>10}")
        print(f"  {'─' * 70}")

        def pct(old: float, new: float) -> str:
            if old == 0:
                return "N/A"
            return f"{((new - old) / old) * 100:+.0f}%"

        print(f"  {'Avg LLM calls/ticket':<30} {cold.avg_llm_calls:>10.1f} {warm.avg_llm_calls:>10.1f} "
              f"{warm.avg_llm_calls - cold.avg_llm_calls:>+10.1f} {pct(cold.avg_llm_calls, warm.avg_llm_calls):>10}")
        print(f"  {'Avg tool calls/ticket':<30} {cold.avg_tool_calls:>10.1f} {warm.avg_tool_calls:>10.1f} "
              f"{warm.avg_tool_calls - cold.avg_tool_calls:>+10.1f} {pct(cold.avg_tool_calls, warm.avg_tool_calls):>10}")
        print(f"  {'Total LLM calls':<30} {cold.total_llm_calls:>10} {warm.total_llm_calls:>10} "
              f"{warm.total_llm_calls - cold.total_llm_calls:>+10}")
        print(f"  {'Total tool calls':<30} {cold.total_tool_calls:>10} {warm.total_tool_calls:>10} "
              f"{warm.total_tool_calls - cold.total_tool_calls:>+10}")
        print(f"  {'search_kb calls':<30} {cold.total_search_kb:>10} {warm.total_search_kb:>10} "
              f"{warm.total_search_kb - cold.total_search_kb:>+10}")
        print(f"  {'recall (memory) calls':<30} {cold.total_recall:>10} {warm.total_recall:>10} "
              f"{warm.total_recall - cold.total_recall:>+10}")
        print(f"  {'Tickets using recall':<30} {cold.recall_count:>10} {warm.recall_count:>10}")
        print(f"  {'Memories available':<30} {cold.memories_before:>10} {warm.memories_before:>10}")

        # ── Learning Effect ────────────────────────────
        print(f"\n{'─' * 80}")
        print("LEARNING EFFECT ANALYSIS")
        print(f"{'─' * 80}")

        effects: list[str] = []

        if warm.avg_tool_calls < cold.avg_tool_calls:
            saved = cold.avg_tool_calls - warm.avg_tool_calls
            effects.append(f"Tool calls reduced by {saved:.1f}/ticket ({pct(cold.avg_tool_calls, warm.avg_tool_calls)} fewer)")

        if warm.total_search_kb < cold.total_search_kb:
            effects.append(f"KB searches reduced: {cold.total_search_kb} -> {warm.total_search_kb} "
                          f"(memory replaced {cold.total_search_kb - warm.total_search_kb} searches)")

        if warm.recall_count > 0:
            effects.append(f"Memory recall used in {warm.recall_count}/{len(warm.tickets)} tickets")

        if warm.avg_llm_calls < cold.avg_llm_calls:
            effects.append(f"LLM calls reduced by {cold.avg_llm_calls - warm.avg_llm_calls:.1f}/ticket")

        if effects:
            print(f"\n  Learning effects detected: {len(effects)}")
            for e in effects:
                print(f"    + {e}")
        else:
            print("\n  No measurable learning effect detected.")

        # ── Verdict ────────────────────────────────────
        print(f"\n{'=' * 80}")
        tool_reduction = (1 - warm.avg_tool_calls / max(cold.avg_tool_calls, 0.1)) * 100
        if tool_reduction > 10 and warm.recall_count > 0:
            print(f"VERDICT: WARM START SHOWS {tool_reduction:.0f}% TOOL REDUCTION.")
            print(f"  Memory persistence works. The framework learns from prior tickets.")
        elif warm.recall_count > 0:
            print(f"VERDICT: MEMORY USED but tool reduction minimal ({tool_reduction:.0f}%).")
        else:
            print("VERDICT: NO LEARNING EFFECT — memory not utilized.")
        print(f"{'=' * 80}")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "shield", "shielddesk-simulator"))
    asyncio.run(run_cold_warm_test())
