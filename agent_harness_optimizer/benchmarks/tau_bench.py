"""TauBenchmark — wraps tau2-bench (sierra-research/tau2-bench).

Task: customer service agent tasks across 5 domains:
  airline, retail, telecom, banking_knowledge, mock

Install tau2-bench:
    pip install tau2-bench   # or: uv add tau2-bench

tau2-bench requires separate LLM configs for the scored agent (inner) and
the simulated user (user_model). Both are passed at construction time.

Splits:  "train" and "test" map to tau2-bench task subsets.
         By default all tasks in a domain = "train"; pass max_cases to subsample.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from agent_harness_optimizer.benchmarks._bfcl_runner import _INFRA_ERROR_SIGNALS
from agent_harness_optimizer.framework.benchmark import (
    Benchmark,
    CaseScore,
    ResourceBudget,
    SplitScore,
)

# tau2-bench imports — optional at module load so the file can be imported
# without tau2 installed (raises only when score_async is actually called).
try:
    import tau2  # noqa: F401

    _TAU2_AVAILABLE = True
except ImportError:
    _TAU2_AVAILABLE = False


_DEFAULT_TAU_MODEL = "azure/gpt-4.1"
_DEFAULT_USER_MODEL = "azure/gpt-4.1"
_SUPPORTED_DOMAINS = ("airline", "retail", "telecom", "banking_knowledge", "mock")


def build_tau_strata(domain: str, pool: list[int], idx_to_task: dict) -> dict[int, str]:
    """Return a strata dict mapping pool_index → category label.

    telecom: category extracted from bracket-prefixed task ID e.g. "[mms_issue]task_name"
    retail:  inferred from user_scenario text — 3 buckets: exchange / return / other
    airline: no meaningful categories — returns empty dict (no stratification)
    others:  returns empty dict
    """
    if domain == "telecom":
        strata = {}
        for idx in pool:
            task = idx_to_task.get(idx)
            tid = str(task.id) if task else ""
            cat = tid.split("]")[0].lstrip("[") if tid.startswith("[") else "unknown"
            strata[idx] = cat
        return strata

    if domain == "retail":
        strata = {}
        for idx in pool:
            task = idx_to_task.get(idx)
            sc = str(getattr(task, "user_scenario", "")).lower() if task else ""
            if "exchange" in sc:
                cat = "exchange"
            elif "return" in sc:
                cat = "return"
            else:
                cat = "other"
            strata[idx] = cat
        return strata

    # airline and others: no stratification
    return {}


# ---------------------------------------------------------------------------
# Proposer system prompts
# ---------------------------------------------------------------------------

_BASE_RULES = textwrap.dedent("""
## Analysis procedure (follow in order)

1. Read task.md for the score and failure list.
2. Read ALL files in train_cases/failures/ — note conversation turns, tool calls, and what
   went wrong.
3. Read ALL files in train_cases/passing/ — note what these cases do correctly.
4. Build a frequency table: for each root cause, count (a) failing cases fixed,
   (b) passing cases at risk of regression.
5. Pick ONE pattern with the highest fix/risk ratio.
6. Before writing anything, state in proposal.md:
   - Pattern: what the root cause is
   - Fix count: how many failing cases this addresses
   - Risk count: how many passing cases use the same behavior
   - Verdict: safe to fix or not
7. Only if safe: apply the fix.

## Surface selection

- system_prompt.txt: behavioral patterns (agent stops early, skips verification, misunderstands
  customer request, fails to use correct tool sequence). Write universal rules that apply
  across all domains and tasks. NEVER name specific tool functions or API parameters.
- middleware: tool-level corrections (wrong argument values, missing prerequisite tool calls,
  incorrect response parsing). Specific logic belongs here, not in the system prompt.

## Middleware safety

- A prerequisite block is safe only if EVERY passing case that calls the dependent tool
  also calls the prerequisite first. One counter-example means do NOT add the block.
- Error messages must enable one-step recovery: exact function name, exact args, reason.

## Constraints

- One fix only. Do not address multiple patterns in one iteration.
- Add at most one or two sentences to the system prompt — do not rewrite it.
- Write real working code in middleware, not pseudocode.
- Update both custom_middleware.py and agent_setup.py if writing middleware.
""").strip()

_PROMPT_ONLY_SYSTEM = textwrap.dedent(f"""
You are Better Agent improving a customer service agent's system prompt.

YOUR ONLY ALLOWED EDIT: current/system_prompt.txt
Do NOT touch current/middleware/.

System prompt rules:
- Write universal behavioral rules only: proper verification, complete tool sequences,
  accurate information retrieval before taking actions.
- NEVER name specific tools, functions, API classes, or parameter values.
  If a fix requires mentioning a tool name, it belongs in middleware, not here.
- Read train_cases/passing/ to confirm the rule won't break already-passing cases.
- One or two short rules max — do not rewrite the whole prompt.

{_BASE_RULES}
""").strip()

_MIDDLEWARE_ONLY_SYSTEM = textwrap.dedent(f"""
You are Better Agent improving a customer service agent's middleware.

YOUR ONLY ALLOWED EDITS: current/middleware/custom_middleware.py and
current/middleware/agent_setup.py
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
You are Better Agent improving a customer service agent's harness.

You have full access to BOTH current/system_prompt.txt AND current/middleware/.
Find the single highest-value fix and apply it to whichever surface fits best.

Surface decision:
- Behavioral pattern (agent stops early, fails to verify, skips required steps) → system_prompt.txt
  Write universal rules only. No tool names, no specific values.
- Tool-specific error (wrong arg, bad format, missing prerequisite) → middleware
  Write real working code. Tool names and specific logic belong here, not in the prompt.
- If the fix needs BOTH surfaces: apply to both in one iteration.

You MUST pick exactly one fix and apply it. If you genuinely find no safe fix,
write exactly "No safe fix found" in proposal.md and explain why.

{_BASE_RULES}
""").strip()

_TAU_VARIANTS = {
    "prompt_only": _PROMPT_ONLY_SYSTEM,
    "middleware_only": _MIDDLEWARE_ONLY_SYSTEM,
    "prompt_middleware_both": _BOTH_SYSTEM,
}

_BEHAVIORAL_LAST_TOOLS = frozenset(
    {
        "task_complete",
        "none",
        "",
        "submit",
        "confirm",
        "ask",
        "respond_to_customer",
    }
)

# Default middleware stubs
_DEFAULT_MIDDLEWARE_IMPL = textwrap.dedent("""
# Custom middleware for the tau2-bench inner agent.
# Write middleware classes here. The harness loads MIDDLEWARE from agent_setup.py.

from __future__ import annotations
from typing import Any, Callable, Awaitable
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware


class TauFixMiddleware(AgentMiddleware):

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
# MIDDLEWARE is passed directly to the agent runner.
# Leave empty to use no middleware.

MIDDLEWARE = []

# To activate:
# from custom_middleware import TauFixMiddleware
# MIDDLEWARE = [TauFixMiddleware()]
""").strip()


class TauBenchmark(Benchmark):
    """tau2-bench benchmark for customer service agent tasks.

    resource_budget controls:
      wall_time_s → per-task timeout (enforced via asyncio.wait_for)
      max_steps   → max turns between agent and simulated user
      max_tokens  → max cumulative tokens (tracked post-hoc; task is stopped if exceeded)

    Args:
        domain:      tau2-bench domain ("airline", "retail", "telecom", "banking_knowledge", "mock")
        model:       inner agent LLM name (any tau2-supported model string)
        user_model:  simulated user LLM name
        budget:      ResourceBudget; defaults to wall_time_s=300, max_steps=30, max_tokens=100_000
    """

    def __init__(
        self,
        *,
        domain: str = "retail",
        model: str = _DEFAULT_TAU_MODEL,
        user_model: str = _DEFAULT_USER_MODEL,
        budget: ResourceBudget | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        if domain not in _SUPPORTED_DOMAINS:
            raise ValueError(f"domain must be one of {_SUPPORTED_DOMAINS}, got {domain!r}")
        self._domain = domain
        self._model = model
        self._user_model = user_model
        self._budget = budget or ResourceBudget(wall_time_s=300.0, max_steps=30, max_tokens=100_000)
        self._data_dir = str(data_dir) if data_dir else None

    @property
    def name(self) -> str:
        return f"tau-{self._domain}"

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def default_system_prompt(self) -> str:
        # Load the domain policy file as the seed prompt so the baseline agent
        # already knows domain rules and available actions.
        policy_candidates = [
            Path(self._data_dir) / "tau2" / "domains" / self._domain / "policy.md"
            if self._data_dir
            else None,
            Path.home()
            / "projects"
            / "tau2-bench"
            / "data"
            / "tau2"
            / "domains"
            / self._domain
            / "policy.md",
        ]
        for p in policy_candidates:
            if p and p.exists():
                return p.read_text().strip()
        return (
            "You are a helpful customer service agent. "
            "Use the available tools to resolve the customer's request accurately and efficiently. "
            "Always verify information before making changes."
        )

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
        num_trials: int = 1,
    ) -> SplitScore:
        if num_trials > 1:
            return await self._score_multi_trial(
                prompt,
                split,
                output_dir,
                middleware_dir=middleware_dir,
                max_cases=max_cases,
                case_indices=case_indices,
                num_trials=num_trials,
            )
        if not _TAU2_AVAILABLE:
            raise ImportError("tau2-bench is not installed. Run: pip install tau2-bench")

        import os

        from tau2.run import get_tasks, run_task

        # Set data directory for tau2 if not already configured
        if not os.environ.get("TAU2_DATA_DIR"):
            if self._data_dir:
                os.environ["TAU2_DATA_DIR"] = self._data_dir
            else:
                import pathlib

                os.environ["TAU2_DATA_DIR"] = str(
                    pathlib.Path.home() / "projects" / "tau2-bench" / "data"
                )

        from agent_harness_optimizer.utils.llm import _ensure_auth, build_model

        build_model(self._model)
        # tau2 calls litellm directly for the user simulator — ensure auth is initialized
        # for the user model too so azure/bedrock credentials are set before tau2 starts.
        _ensure_auth(self._user_model)

        output_dir.mkdir(parents=True, exist_ok=True)

        # tau2 only has a "base" split for most domains; map our internal split names
        # ("train", "holdout") both to None (all tasks) since tau2 splits are separate
        # task-set files, not row-level flags — the BH optimizer subsamples via max_cases.
        task_set_name = self._domain
        all_tasks = get_tasks(task_set_name)
        if case_indices is not None:
            tasks = [all_tasks[i] for i in case_indices]
        elif max_cases is not None:
            tasks = all_tasks[:max_cases]
        else:
            tasks = all_tasks

        # Register a custom LLMAgent subclass that injects our optimized prompt as domain_policy.
        agent_name = _register_custom_agent_class(prompt)

        domain = self._domain
        model = self._model
        user_model = self._user_model
        max_steps = self._budget.max_steps or 30

        cases: list[CaseScore] = []
        sem = asyncio.Semaphore(8)  # max 8 concurrent simulations

        async def _run_one(task) -> CaseScore:
            async with sem:

                def _run_in_thread():
                    from agent_harness_optimizer.utils.llm import _ensure_auth
                    from agent_harness_optimizer.utils.llm import build_model as _bm

                    _bm(model)
                    _ensure_auth(user_model)  # tau2 user sim calls litellm directly
                    return run_task(
                        domain=domain,
                        task=task,
                        agent=agent_name,
                        user="user_simulator",
                        llm_agent=model,
                        llm_user=user_model,
                        max_steps=max_steps,
                        seed=42,
                    )

                start = time.monotonic()
                task_id = str(task.id)
                traj_dir = output_dir / task_id
                traj_dir.mkdir(exist_ok=True)
                try:
                    timeout = self._budget.wall_time_s or 900.0
                    sim = await asyncio.wait_for(
                        asyncio.to_thread(_run_in_thread),
                        timeout=timeout,
                    )
                    elapsed = time.monotonic() - start

                    reward = 0.0
                    if sim.reward_info is not None:
                        reward = sim.reward_info.reward
                    passed = reward >= 1.0

                    agent_cost = sim.agent_cost or 0.0
                    user_cost = sim.user_cost or 0.0

                    # SimulationRun has no token count fields; leave as 0
                    prompt_tokens = 0
                    completion_tokens = 0

                    # Estimate turns from messages
                    raw_messages = sim.messages or []
                    msg_count = len(raw_messages)
                    turns = msg_count // 2

                    within_step = turns <= (self._budget.max_steps or 9999)
                    within_token = True  # tau2 manages its own token budget

                    stuck = ""
                    if not passed:
                        tr = sim.termination_reason
                        term_val = tr.value if hasattr(tr, "value") else str(tr)
                        if "timeout" in term_val.lower() or "max_turns" in term_val.lower():
                            stuck = "step_limit"
                        elif "error" in term_val.lower():
                            stuck = "crash"

                    # Build compact conversation transcript for the outer LLM proposer.
                    # Each entry: {"role": "assistant"|"user"|"tool", "content": ...,
                    #              "tool_calls": [{"name":..., "args":...}, ...]}
                    transcript: list[dict] = []
                    agent_tool_calls: list[dict] = []
                    for msg in raw_messages:
                        role = getattr(msg, "role", "unknown")
                        content = getattr(msg, "content", None) or ""
                        tcs = getattr(msg, "tool_calls", None) or []
                        entry: dict = {"role": role, "content": content}
                        if tcs:
                            entry["tool_calls"] = [
                                {"name": tc.name, "args": tc.arguments}
                                if hasattr(tc, "name")
                                else {"raw": str(tc)}
                                for tc in tcs
                            ]
                            if role == "assistant":
                                agent_tool_calls.extend(entry["tool_calls"])
                        transcript.append(entry)

                    (traj_dir / "result.json").write_text(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "passed": passed,
                                "stuck_type": stuck,
                                "reward": reward,
                                "turns": turns,
                                "termination_reason": str(sim.termination_reason),
                                "agent_cost": agent_cost,
                                "user_cost": user_cost,
                            },
                            indent=2,
                            default=str,
                        )
                    )

                    return CaseScore(
                        case_id=task_id,
                        passed=passed if within_step else False,
                        stuck_type=stuck,
                        within_time_budget=True,
                        within_step_budget=within_step,
                        within_token_budget=within_token,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        wall_time_s=round(elapsed, 2),
                        category=self._domain,
                        extra={
                            "turns": turns,
                            "error": stuck,
                            "tool_calls": agent_tool_calls,
                            "transcript": transcript,
                            "reward": reward,
                        },
                    )

                except TimeoutError:
                    elapsed = time.monotonic() - start
                    return CaseScore(
                        case_id=task_id,
                        passed=False,
                        stuck_type="timeout",
                        within_time_budget=False,
                        wall_time_s=round(elapsed, 2),
                        category=self._domain,
                    )
                except Exception as exc:
                    import traceback as _tb

                    elapsed = time.monotonic() - start
                    err = _tb.format_exc()
                    _is_infra = any(s in err for s in _INFRA_ERROR_SIGNALS)
                    _stuck = "infra_error" if _is_infra else "crash"
                    print(f"  [tau] {task_id}: {_stuck} — {err[:300]}")
                    return CaseScore(
                        case_id=task_id,
                        passed=False,
                        stuck_type=_stuck,
                        wall_time_s=round(elapsed, 2),
                        category=self._domain,
                        extra={"error": str(exc)},
                    )

        cases = list(await asyncio.gather(*[_run_one(t) for t in tasks]))

        passed_count = sum(1 for c in cases if c.passed)
        stuck_count = sum(1 for c in cases if c.stuck_type)
        reliability = round(1.0 - (stuck_count / len(cases) if cases else 0.0), 4)
        n = len(cases) or 1
        tokens_per_case = sum(c.total_tokens for c in cases) / n
        prompt_per_case = sum(c.prompt_tokens for c in cases) / n
        completion_per_case = sum(c.completion_tokens for c in cases) / n

        return SplitScore(
            passed=passed_count,
            total=len(cases),
            reliability=reliability,
            tokens_per_case=round(tokens_per_case, 1),
            prompt_tokens_per_case=round(prompt_per_case, 1),
            completion_tokens_per_case=round(completion_per_case, 1),
            cases=cases,
        )

    async def _score_multi_trial(
        self,
        prompt: str,
        split: str,
        output_dir: Path,
        *,
        middleware_dir: Path | None = None,
        max_cases: int | None = None,
        case_indices: list[int] | None = None,
        num_trials: int = 3,
    ) -> SplitScore:
        """Run score_async num_trials times; a case passes only if it passes in ALL trials (pass^k)."""
        trial_scores: list[SplitScore] = []
        for t in range(num_trials):
            trial_dir = output_dir.parent / f"{output_dir.name}_trial{t}"
            s = await self.score_async(
                prompt,
                split,
                trial_dir,
                middleware_dir=middleware_dir,
                max_cases=max_cases,
                case_indices=case_indices,
                num_trials=1,
            )
            trial_scores.append(s)

        # AND rule: case passes only if it passes in every trial
        # Use case_ids from first trial as the reference set
        case_results: dict[str, list[bool]] = {}
        for s in trial_scores:
            for c in s.cases:
                case_results.setdefault(c.case_id, []).append(c.passed)

        # Build combined CaseScore list with pass^k semantics
        first_by_id = {c.case_id: c for c in trial_scores[0].cases}
        combined_cases: list[CaseScore] = []
        for case_id, trial_passes in case_results.items():
            if len(trial_passes) < num_trials:
                pass_k = False  # missing trials count as failure
            else:
                pass_k = all(trial_passes)
            base = first_by_id.get(case_id)
            if base is not None:
                combined_cases.append(
                    CaseScore(
                        case_id=base.case_id,
                        passed=pass_k,
                        stuck_type=base.stuck_type if not pass_k else "",
                        within_time_budget=base.within_time_budget,
                        within_step_budget=base.within_step_budget,
                        within_token_budget=base.within_token_budget,
                        prompt_tokens=base.prompt_tokens,
                        completion_tokens=base.completion_tokens,
                        wall_time_s=base.wall_time_s,
                        category=base.category,
                        extra={
                            **base.extra,
                            "pass_k_trials": trial_passes,
                            "num_trials": num_trials,
                        },
                    )
                )

        # Write combined pass^k result.json per case into output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for c in combined_cases:
            case_dir = output_dir / c.case_id
            case_dir.mkdir(exist_ok=True)
            (case_dir / "result.json").write_text(
                json.dumps(
                    {
                        "task_id": c.case_id,
                        "passed": c.passed,
                        "num_trials": num_trials,
                        "trial_passes": c.extra.get("pass_k_trials", []),
                    },
                    indent=2,
                )
            )

        passed_count = sum(1 for c in combined_cases if c.passed)
        stuck_count = sum(1 for c in combined_cases if c.stuck_type)
        n = len(combined_cases) or 1
        reliability = round(1.0 - (stuck_count / n), 4)
        tokens_per_case = sum(c.total_tokens for c in combined_cases) / n
        prompt_per_case = sum(c.prompt_tokens for c in combined_cases) / n
        completion_per_case = sum(c.completion_tokens for c in combined_cases) / n

        return SplitScore(
            passed=passed_count,
            total=len(combined_cases),
            reliability=reliability,
            tokens_per_case=round(tokens_per_case, 1),
            prompt_tokens_per_case=round(prompt_per_case, 1),
            completion_tokens_per_case=round(completion_per_case, 1),
            cases=combined_cases,
        )

    # --- analysis ---

    def build_asi(self, score: SplitScore, failure_matrix_cases: dict[str, str] | None) -> str:
        failures = [c for c in score.cases if not c.passed]
        if not failures:
            return "# ASI\n\nNo failures.\n"

        by_last_tool: dict[str, list[CaseScore]] = {}
        for c in failures:
            tool_calls = c.extra.get("tool_calls", [])
            last = tool_calls[-1]["name"] if tool_calls else "none"
            by_last_tool.setdefault(last, []).append(c)

        by_error: dict[str, list[CaseScore]] = {}
        for c in failures:
            err = str(c.extra.get("error", ""))[:50].strip() or "(no error)"
            by_error.setdefault(err, []).append(c)

        lines = [
            "# Actionable Side Information (ASI)",
            "",
            f"Domain: {self._domain}  Failures: {len(failures)}/{score.total}",
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
        for err, cases in sorted(by_error.items(), key=lambda x: -len(x[1]))[:6]:
            lines.append(f"- '{err}': {len(cases)} cases — {[c.case_id for c in cases[:4]]}")
        lines.append("")

        if failure_matrix_cases:
            lines.append("## RECURRING cases (most fixable)")
            for c in failures:
                if failure_matrix_cases.get(c.case_id) == "RECURRING":
                    turns = c.extra.get("turns", "?")
                    err = str(c.extra.get("error", ""))[:100].replace("\n", " ")
                    lines.append(f"- {c.case_id}: turns={turns} error={err}")
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
            err = str(c.extra.get("error", ""))[:40].strip()
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
                "passed": c.passed,
                "stuck_type": c.stuck_type,
                "turns": c.extra.get("turns"),
                "error": c.extra.get("error"),
                "tool_calls": c.extra.get("tool_calls", []),
                "transcript": c.extra.get("transcript", []),
                "reward": c.extra.get("reward"),
            }
            if not c.passed:
                if target_case_ids is not None and c.case_id not in target_case_ids:
                    continue
                dest = failures_dir
            else:
                dest = passing_dir
            (dest / f"{c.case_id}.json").write_text(json.dumps(data, indent=2, default=str))

    # --- proposer workspace ---

    def get_proposer_variants(self) -> dict[str, str]:
        return _TAU_VARIANTS

    def build_task_md(self, score: SplitScore, iteration: int) -> str:
        failures = [c for c in score.cases if not c.passed]

        lines = [
            f"# tau2-bench ({self._domain}) Harness Improvement Task",
            "",
            "You are improving the inner customer service agent harness using eval feedback.",
            "",
            "Rules:",
            "- Edit only files under `current/`.",
            "- Do not edit files under `train_cases/`, `history/`, or this task file.",
            "- Read `history/history.md` FIRST — do NOT propose anything already tried.",
            "- Read `history/failure_matrix.md`: prioritise NEW and RECURRING.",
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
            f"Score: {score.passed}/{score.total} passed",
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
            turns = c.extra.get("turns", "?")
            snippet = str(c.extra.get("error") or "")[:200].replace("\n", " ")
            lines.append(f"- `{c.case_id}` [turns={turns}]: last calls=[{calls}] error={snippet}")
        lines.append("")
        return "\n".join(lines)

    def select_best_variant(self, score: SplitScore, middleware_active: bool) -> str | None:
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

        if behavioral_fraction >= 0.40:
            return "prompt_only"

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

        return None

    def get_default_middleware_stubs(self) -> dict[str, str]:
        return {
            "custom_middleware.py": _DEFAULT_MIDDLEWARE_IMPL,
            "agent_setup.py": _DEFAULT_AGENT_SETUP,
        }

    # --- model ---

    def build_model(self, model_name: str):
        from agent_harness_optimizer.utils.llm import build_model as _bm

        return _bm(model_name)


# ---------------------------------------------------------------------------
# Custom tau2 agent class — injects optimized system prompt as domain_policy
# ---------------------------------------------------------------------------


def _register_custom_agent_class(system_prompt: str) -> str:
    """Register (or re-register) a tau2 LLMAgent subclass that uses *system_prompt*.

    tau2's LLMAgent takes domain_policy in __init__ and builds its system prompt from:
        <instructions>{AGENT_INSTRUCTION}</instructions>
        <policy>{domain_policy}</policy>

    We subclass LLMAgent and override __init__ to force domain_policy = system_prompt,
    ignoring whatever the environment provides. The class is registered under a stable
    name so repeated calls simply replace the previous registration.

    Returns the registered agent name to pass to run_task().
    """
    _NAME = "harness_optimizer_agent"
    try:
        from tau2.agent.llm_agent import LLMAgent
        from tau2.registry import registry
    except ImportError:
        return "llm_agent"  # fallback to default if tau2 not available

    _prompt = system_prompt

    # Dynamically build a subclass that overrides domain_policy with our prompt
    class _HarnessAgent(LLMAgent):
        def __init__(self, tools, domain_policy: str = "", **kwargs):  # noqa: ARG002
            super().__init__(tools=tools, domain_policy=_prompt, **kwargs)

    _HarnessAgent.__name__ = _NAME
    _HarnessAgent.__qualname__ = _NAME

    # Replace existing registration if present
    if _NAME in registry._agents:  # noqa: SLF001
        del registry._agents[_NAME]  # noqa: SLF001

    registry.register_agent(_HarnessAgent, _NAME)
    return _NAME


# ---------------------------------------------------------------------------
# Middleware loader
# ---------------------------------------------------------------------------


def _load_middleware(middleware_dir: Path | None) -> list:
    """Load MIDDLEWARE list from middleware_dir/agent_setup.py, or return []."""
    if middleware_dir is None or not middleware_dir.is_dir():
        return []
    setup = middleware_dir / "agent_setup.py"
    if not setup.exists():
        return []
    import importlib.util

    mw_dir_str = str(middleware_dir)
    if mw_dir_str not in sys.path:
        sys.path.insert(0, mw_dir_str)

    # Pre-load custom_middleware.py so relative imports in agent_setup.py work
    custom_mw = middleware_dir / "custom_middleware.py"
    if custom_mw.exists():
        cm_spec = importlib.util.spec_from_file_location(
            "_tau_mw_pkg.custom_middleware",
            str(custom_mw),
        )
        if cm_spec and cm_spec.loader:
            cm_mod = importlib.util.module_from_spec(cm_spec)
            sys.modules.setdefault("_tau_mw_pkg", type(sys)("_tau_mw_pkg"))
            sys.modules["_tau_mw_pkg.custom_middleware"] = cm_mod
            sys.modules["custom_middleware"] = cm_mod
            cm_spec.loader.exec_module(cm_mod)  # type: ignore[union-attr]

    pkg_name = "_tau_mw_pkg.agent_setup"
    spec = importlib.util.spec_from_file_location(
        pkg_name, str(setup), submodule_search_locations=[]
    )
    if spec is None or spec.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "_tau_mw_pkg"
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, "MIDDLEWARE", [])
