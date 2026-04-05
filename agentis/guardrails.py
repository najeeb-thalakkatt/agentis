"""ModelGuardrails — tier-adaptive instructions for different model capabilities.

Weaker models need explicit step-by-step instructions and hard constraints.
Stronger models perform better with open-ended guidance.
The guardrails are injected as system messages by the AgentRuntime.
"""

from __future__ import annotations

_WEAK_GUARDRAILS = """\
IMPORTANT — MODEL GUARDRAILS (strict mode):
You are running with limited capability. Follow these rules exactly:

1. ALWAYS investigate before acting. Call at least 2 read-only tools
   before any mutating or dangerous action.
2. NEVER conclude without taking action. If the task requires remediation
   (restart, rollback, scale, fix, send_response, write_report), you MUST
   call the appropriate tool. Do NOT just describe what you would do.
3. Follow the workflow IN ORDER. Do not skip steps or loop back.
4. Limit verification calls: at most 2 verify calls, at most 3 freshness
   checks. Do not re-check data you already have.
5. Call write_report or send_response EXACTLY ONCE as your final action.
   Do not revise or rewrite.
6. If you are unsure, escalate rather than guessing.
7. Complete in 6 turns or fewer. Every extra turn wastes budget.
8. NEVER repeat the same tool call with the same arguments.
"""

_MODERATE_GUARDRAILS = """\
GUIDELINES (moderate mode):
- Investigate before acting. Gather evidence from at least 2 sources
  before remediation.
- Always take action — do not just describe what should be done.
  Call the tool to do it.
- Avoid repeating tool calls. If you already have the data, use it.
- Aim to complete in 8 turns or fewer.
"""

_STRONG_GUARDRAILS = ""


class ModelGuardrails:
    """Generates model-tier-appropriate system prompt additions."""

    _PROMPTS = {
        "weak": _WEAK_GUARDRAILS,
        "moderate": _MODERATE_GUARDRAILS,
        "strong": _STRONG_GUARDRAILS,
    }

    @classmethod
    def get_prompt(cls, tier: str) -> str:
        """Return guardrail instructions for the given model tier.

        Args:
            tier: One of "strong", "moderate", "weak".

        Returns:
            System prompt addition (empty string for strong models).
        """
        return cls._PROMPTS.get(tier, "")
