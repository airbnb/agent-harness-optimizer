"""BFCLBenchmark — wraps score_bfcl_async from agent_harness_optimizer.benchmarks._bfcl_runner.

Task: multi-turn API tool-calling (Berkeley Function Calling Leaderboard v4).
100 train cases + 100 holdout cases.

BFCL-specific proposer logic lives here:
  - 3 variants (prompt_only / middleware_only / prompt_middleware_both) with
    detailed system prompts including middleware safety rules
  - Signal-based variant selection (dominant last tool → middleware; behavioral → prompt)
  - task.md includes VASR, category labels, full surface path descriptions
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from agent_harness_optimizer.benchmarks._bfcl_runner import (
    _DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    BFCLScore,
    _build_model,
    score_bfcl_async,
)
from agent_harness_optimizer.framework.benchmark import (
    Benchmark,
    CaseScore,
    ResourceBudget,
    SplitScore,
)

# ---------------------------------------------------------------------------
# BFCL-specific proposer system prompts
# ---------------------------------------------------------------------------

_BASE_RULES = textwrap.dedent("""
## Analysis procedure (follow in order)

1. Read task.md for the score and failure list.
2. Read ALL files in train_cases/failures/ — note tool sequences and what went wrong.
3. Read ALL files in train_cases/passing/ — note what these cases do correctly.
4. Build a frequency table: for each root cause, count (a) failing cases fixed,
   (b) passing cases at risk of regression.
5. Pick ONE pattern with the highest fix/risk ratio.
6. Before writing code, state in proposal.md:
   - Pattern: what the root cause is
   - Fix count: how many failing cases this addresses
   - Risk count: how many passing cases use the same tool/behavior
   - Verdict: safe to fix or not
7. Only if safe: apply the fix.

## Surface selection

- system_prompt.txt: behavioral patterns only (stops early, skips steps, asks for
  confirmation). NEVER name specific tools, API classes, or parameter values.
  Rules must be universal — safe for every case regardless of which APIs are active.
- middleware: tool-level corrections (wrong arg value, bad format, missing prerequisite).
  Tool names and specific logic belong here, NOT in the system prompt.

## Middleware safety

- A prerequisite block is only safe if EVERY passing case that calls the dependent tool
  also calls the prerequisite first. One counter-example means do NOT add the block.
- Error messages must enable one-step recovery: exact function name, exact args, reason.
  Bad:  "Error: call pressBrakePedal first."
  Good: "Error: startEngine requires brake pressed.
         Call pressBrakePedal(pedalPosition=1.0) then retry."

## Constraints

- One fix only. Do not address multiple patterns in one iteration.
- Add at most one or two sentences to the system prompt — do not rewrite it.
- Write real working code in middleware, not pseudocode.
- Update both custom_middleware.py and agent_setup.py if writing middleware.
""").strip()

_PROMPT_ONLY_SYSTEM = textwrap.dedent(f"""
You are Better Agent improving an inner agent's system prompt.

YOUR ONLY ALLOWED EDIT: current/system_prompt.txt
Do NOT touch current/middleware/.

System prompt rules:
- Write universal behavioral rules only: sequencing, completeness, not stopping early.
- NEVER name specific tools, functions, API classes, or parameter values.
  If a fix requires mentioning a tool name, it belongs in middleware, not here.
- Read train_cases/passing/ to confirm the rule won't break already-passing cases.
- One or two short rules max — do not rewrite the whole prompt.

{_BASE_RULES}
""").strip()

_MIDDLEWARE_ONLY_SYSTEM = textwrap.dedent(f"""
You are Better Agent improving an inner agent's middleware.

YOUR ONLY ALLOWED EDITS: current/middleware/custom_middleware.py
and current/middleware/agent_setup.py
Do NOT touch current/system_prompt.txt.

Middleware is the right surface for tool-specific fixes: wrong arg values, bad formats,
missing prerequisites. Check train_cases/passing/ before blocking any tool call.

Three middleware patterns:
1. Silent correction — mutate args before execution (use when the fix is always safe):
   new_call = {{**call, "args": {{**args, "key": fixed}}}}
   request = request.override(tool_call=new_call)

2. Block with error — model retries with correct value:
   return ToolMessage(content="Error: <exact function + args + reason>",
                      tool_call_id=call["id"], status="error")

3. Block on missing prerequisite — verify in history first:
   messages = request.state.get("messages", [])
   already = any(tc.get("name") == "prereq" for m in messages
                 if isinstance(m, AIMessage) for tc in (m.tool_calls or []))
   if name == "dependent" and not already:
       return ToolMessage(content="Error: <exact prereq call>",
                          tool_call_id=call["id"], status="error")

ALWAYS use request.override(tool_call=new_call) to mutate args. NEVER mutate call["args"] directly.

{_BASE_RULES}
""").strip()

_BOTH_SYSTEM = textwrap.dedent(f"""
You are Better Agent improving an inner agent's harness.

You have full access to BOTH current/system_prompt.txt AND current/middleware/.
Find the single highest-value fix and apply it to whichever surface fits best.

Surface decision:
- Behavioral pattern (stops early, asks confirmation, skips steps) → system_prompt.txt
  Write universal rules only. No tool names, no specific values.
- Tool-specific error (wrong arg, bad format, missing prerequisite) → middleware
  Write real working code. Tool names and specific logic belong here, not in the prompt.
- If the fix needs BOTH surfaces: apply to both in one iteration.

You MUST pick exactly one fix and apply it. If you genuinely find no safe fix,
write exactly "No safe fix found" in proposal.md and explain why.

{_BASE_RULES}
""").strip()

_BFCL_VARIANTS = {
    "prompt_only": _PROMPT_ONLY_SYSTEM,
    "middleware_only": _MIDDLEWARE_ONLY_SYSTEM,
    "prompt_middleware_both": _BOTH_SYSTEM,
}

_BEHAVIORAL_LAST_TOOLS = frozenset(
    {
        "write_todos",
        "task_complete",
        "none",
        "",
        "submit",
        "confirm",
        "ask",
    }
)

# Default middleware stubs written into workspace when no prior middleware exists
_DEFAULT_MIDDLEWARE_IMPL = textwrap.dedent("""
# Custom middleware for the BFCL inner agent.
# Write middleware classes here. The harness loads MIDDLEWARE from agent_setup.py.

from __future__ import annotations
from typing import Any, Callable, Awaitable
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware


class BFCLFixMiddleware(AgentMiddleware):

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        call = request.tool_call
        name = call["name"]
        args = call.get("args", {})
        # Add fixes here. Use request.override(tool_call=new_call) to mutate args.
        return await handler(request)
""").strip()

_DEFAULT_AGENT_SETUP = textwrap.dedent("""
# MIDDLEWARE is passed directly to create_deep_agent(middleware=MIDDLEWARE).
# Leave empty to use no middleware.

MIDDLEWARE = []

# To activate:
# from custom_middleware import BFCLFixMiddleware
# MIDDLEWARE = [BFCLFixMiddleware()]
""").strip()


class BFCLBenchmark(Benchmark):
    """BFCL v4 multi-turn tool-calling benchmark."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        budget: ResourceBudget | None = None,
        split_seed: int | None = None,
    ) -> None:
        self._model = model
        self._budget = budget or ResourceBudget(
            wall_time_s=300.0, max_steps=100, max_tokens=500_000
        )
        self._split_seed = split_seed

    @property
    def name(self) -> str:
        return "bfcl"

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def default_system_prompt(self) -> str:
        return DEFAULT_SYSTEM_PROMPT

    @property
    def resource_budget(self) -> ResourceBudget:
        return self._budget

    # --- scoring ---

    async def score_async(
        self,
        prompt: str,
        split: str,
        output_dir: Path,
        *,
        middleware_dir: Path | None = None,
        max_cases: int | None = None,
        case_indices: list[int] | None = None,
        num_trials: int = 1,  # ignored for BFCL; pass^k not applicable
    ) -> SplitScore:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Resolve integer positions → case IDs so score_bfcl_async can filter by ID.
        case_ids: list[str] | None = None
        if case_indices is not None:
            from agent_harness_optimizer.benchmarks._bfcl_runner import _load_split_cases

            all_split_cases = _load_split_cases(split, split_seed=self._split_seed)
            case_ids = [all_split_cases[i]["id"] for i in case_indices if i < len(all_split_cases)]
        bfcl_score: BFCLScore = await score_bfcl_async(
            system_prompt=prompt,
            model_name=self._model,
            split=split,
            output_dir=output_dir,
            middleware_dir=middleware_dir,
            max_cases=max_cases if case_ids is None else None,
            case_ids=case_ids,
            time_budget_s=self._budget.wall_time_s or 300.0,
            split_seed=self._split_seed,
        )
        return self._to_split_score(bfcl_score)

    def _to_split_score(self, bfcl_score: BFCLScore) -> SplitScore:
        cases = []
        for r in bfcl_score.results:
            cases.append(
                CaseScore(
                    case_id=r.case_id,
                    passed=r.passed,
                    stuck_type=r.stuck_type,
                    within_time_budget=r.within_time_budget,
                    within_step_budget=r.within_step_budget,
                    within_token_budget=r.within_token_budget,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    wall_time_s=r.wall_time_s,
                    category=r.category,
                    extra={
                        "tool_calls": r.tool_calls,
                        "state_diff": r.state_diff,
                        "error": r.error,
                        "turns": r.turns,
                        "vasr_eligible": r.vasr_eligible,
                    },
                )
            )
        n = len(cases) or 1
        tokens_per_case = sum(c.total_tokens for c in cases) / n
        prompt_per_case = sum(c.prompt_tokens for c in cases) / n
        completion_per_case = sum(c.completion_tokens for c in cases) / n
        return SplitScore(
            passed=bfcl_score.passed,
            total=bfcl_score.total,
            reliability=round(1.0 - bfcl_score.stuck_rate, 4),
            tokens_per_case=round(tokens_per_case, 1),
            prompt_tokens_per_case=round(prompt_per_case, 1),
            completion_tokens_per_case=round(completion_per_case, 1),
            cases=cases,
        )

    # --- analysis ---

    def build_asi(self, score: SplitScore, failure_matrix_cases: dict[str, str] | None) -> str:
        failures = [c for c in score.cases if not c.passed]
        if not failures:
            return "# ASI\n\nNo failures — nothing to improve.\n"

        by_last_tool: dict[str, list[CaseScore]] = {}
        for c in failures:
            tool_calls = c.extra.get("tool_calls", [])
            last = tool_calls[-1]["name"] if tool_calls else "none"
            by_last_tool.setdefault(last, []).append(c)

        by_error: dict[str, list[CaseScore]] = {}
        for c in failures:
            err = (c.extra.get("error") or c.extra.get("state_diff") or "")[:60].strip()
            by_error.setdefault(err or "(no error)", []).append(c)

        lines = [
            "# Actionable Side Information (ASI)",
            "",
            f"Total failures: {len(failures)}/{score.total}",
            "",
        ]

        lines.append("## Failures by last tool called")
        for tool, cases in sorted(by_last_tool.items(), key=lambda x: -len(x[1])):
            ids = [c.case_id for c in cases[:5]]
            note = ""
            if failure_matrix_cases:
                rec = sum(1 for c in cases if failure_matrix_cases.get(c.case_id) == "RECURRING")
                per = sum(1 for c in cases if failure_matrix_cases.get(c.case_id) == "PERSISTENT")
                note = f" [REC={rec} PER={per}]"
            lines.append(f"- last_tool={tool}: {len(cases)} cases{note} — {ids}")
        lines.append("")

        lines.append("## Failures by error message")
        for err, cases in sorted(by_error.items(), key=lambda x: -len(x[1]))[:8]:
            lines.append(f"- '{err[:60]}': {len(cases)} cases — {[c.case_id for c in cases[:4]]}")
        lines.append("")

        if failure_matrix_cases:
            lines.append("## RECURRING cases (most fixable)")
            for c in failures:
                if failure_matrix_cases.get(c.case_id) == "RECURRING":
                    calls = c.extra.get("tool_calls", [])
                    call_seq = " → ".join(tc["name"] for tc in calls[-5:]) or "none"
                    diff = (c.extra.get("state_diff") or c.extra.get("error") or "")[:120].replace(
                        "\n", " "
                    )
                    lines.append(f"- {c.case_id} [{c.category}]: [{call_seq}] {diff}")
            lines.append("")

        return "\n".join(lines)

    def extract_top_patterns(self, score: SplitScore, n: int = 3) -> list[dict]:
        failures = [c for c in score.cases if not c.passed]
        clusters: dict[str, dict] = {}
        for c in failures:
            # Skip infra_error cases — unfixable by prompt/middleware changes
            if c.stuck_type == "infra_error":
                continue
            tool_calls = c.extra.get("tool_calls", [])
            last = tool_calls[-1]["name"] if tool_calls else "none"
            err = (c.extra.get("error") or c.extra.get("state_diff") or "")[:40].strip()
            key = f"{last}:{err}"
            if key not in clusters:
                clusters[key] = {
                    "key": key,
                    "last_tool": last,
                    "error_prefix": err,
                    "count": 0,
                    "case_ids": [],
                }
            clusters[key]["count"] += 1
            clusters[key]["case_ids"].append(c.case_id)
        return sorted(clusters.values(), key=lambda x: -x["count"])[:n]

    def write_case_files(
        self, workspace: Path, score: SplitScore, target_case_ids: set[str] | None = None
    ) -> None:
        failures_dir = workspace / "train_cases" / "failures"
        passing_dir = workspace / "train_cases" / "passing"
        failures_dir.mkdir(parents=True, exist_ok=True)
        passing_dir.mkdir(parents=True, exist_ok=True)
        for c in score.cases:
            data: dict[str, Any] = {
                "case_id": c.case_id,
                "category": c.category,
                "passed": c.passed,
                "stuck_type": c.stuck_type,
                "tool_calls": c.extra.get("tool_calls", []),
                "state_diff": c.extra.get("state_diff"),
                "error": c.extra.get("error"),
            }
            if not c.passed:
                # Only write target-pattern failures — prevents dominant patterns
                # (e.g. brake-pedal) from hijacking LLM attention away from the
                # assigned target pattern.
                if target_case_ids is not None and c.case_id not in target_case_ids:
                    continue
                dest = failures_dir
            else:
                dest = passing_dir
            (dest / f"{c.case_id}.json").write_text(json.dumps(data, indent=2))

    # --- BFCL-specific proposer workspace content ---

    def get_proposer_variants(self) -> dict[str, str]:
        return _BFCL_VARIANTS

    def build_task_md(self, score: SplitScore, iteration: int) -> str:
        failures = [c for c in score.cases if not c.passed]
        passed = score.passed
        total = score.total
        vasr_eligible = sum(1 for c in score.cases if c.extra.get("vasr_eligible", c.passed))
        vasr = vasr_eligible / total if total else 0.0

        lines = [
            "# BFCL Harness Improvement Task",
            "",
            "You are improving the inner agent harness using eval feedback.",
            "",
            "Rules:",
            "- Edit only files under `current/`.",
            "- Do not edit files under `train_cases/`, `history/`, or this task file.",
            "- Read `history/history.md` FIRST — do NOT propose anything already tried.",
            "- Read `history/failure_matrix.md`: prioritise NEW (regressions) and RECURRING.",
            "  PERSISTENT cases have resisted all prior fixes — skip unless genuinely new idea.",
            "  FIXED cases are passing — do not touch behavior that fixed them.",
            "- Read train_cases/failures/<id>.json for full trajectory + tool call args.",
            "- Read train_cases/passing/<id>.json to verify a fix won't cause regression.",
            "- BEFORE adding any middleware prerequisite block: check passing/ for the same tool.",
            "  If any passing case calls that tool without the prerequisite, do NOT add the block.",
            "- One fix only. Smallest safe change.",
            "- If no safe fix: write 'No safe fix found' in proposal.md, leave current/ unchanged.",
            "",
            f"Iteration: {iteration}",
            f"Score: {passed}/{total} passed  VASR: {vasr:.1%}",
            "",
            "Editable surfaces:",
            "- `system_prompt` → `current/system_prompt.txt` (behavioral rules only)",
            "- `middleware`     → `current/middleware/`"
            " (tool-level corrections via MIDDLEWARE list)",
            "",
            "Visible failures:",
        ]
        for c in failures:
            tool_calls = c.extra.get("tool_calls", [])
            calls = ", ".join(tc["name"] for tc in tool_calls[-4:]) or "none"
            snippet = (c.extra.get("state_diff") or c.extra.get("error") or "")[:200].replace(
                "\n", " "
            )
            lines.append(f"- `{c.case_id}` [{c.category}]: last calls=[{calls}] diff={snippet}")
        lines.append("")
        return "\n".join(lines)

    def select_best_variant(self, score: SplitScore, middleware_active: bool) -> str | None:
        """Signal-based variant selection. Returns None to run all 3 in parallel."""
        failures = [c for c in score.cases if not c.passed]
        if not failures:
            return None

        last_tool_counts: dict[str, int] = {}
        for c in failures:
            tool_calls = c.extra.get("tool_calls", [])
            last = tool_calls[-1]["name"] if tool_calls else "__none__"
            last_tool_counts[last] = last_tool_counts.get(last, 0) + 1

        top_tool, top_count = max(last_tool_counts.items(), key=lambda x: x[1])
        top_fraction = top_count / len(failures)

        behavioral_fraction = sum(last_tool_counts.get(t, 0) for t in _BEHAVIORAL_LAST_TOOLS) / len(
            failures
        )

        # Strong behavioral signal → prompt_only only
        if behavioral_fraction >= 0.40:
            return "prompt_only"

        # Dominant non-behavioral tool + appears in passing cases → middleware_only only
        if top_fraction >= 0.35 and top_tool not in _BEHAVIORAL_LAST_TOOLS:
            passing_with_top = sum(
                1
                for c in score.cases
                if c.passed
                and c.extra.get("tool_calls")
                and c.extra["tool_calls"][-1]["name"] == top_tool
            )
            if passing_with_top >= 2:
                return "middleware_only"

        # Mixed signal → run all 3 variants in parallel
        return None

    def get_default_middleware_stubs(self) -> dict[str, str]:
        """Return default middleware file contents for workspace setup."""
        return {
            "custom_middleware.py": _DEFAULT_MIDDLEWARE_IMPL,
            "agent_setup.py": _DEFAULT_AGENT_SETUP,
        }

    # --- model ---

    def build_model(self, model_name: str):
        return _build_model(model_name)
