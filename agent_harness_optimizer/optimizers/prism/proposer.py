"""PRISM proposer — benchmark-agnostic outer-agent workspace builder and runner.

All benchmark-specific analysis (ASI text, case files, pattern extraction)
is delegated to the Benchmark instance.  The proposer only handles:
  - workspace directory layout and file scaffolding
  - outer agent invocation (via deepagents FilesystemBackend)
  - reading back the edited surfaces after agent completes
"""

from __future__ import annotations

import json
import re as _re
import shutil
import textwrap
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent_harness_optimizer.framework.benchmark import Benchmark, SplitScore
from agent_harness_optimizer.optimizers.prism.types import Candidate

# ---------------------------------------------------------------------------
# Default middleware stubs
# ---------------------------------------------------------------------------

_DEFAULT_MIDDLEWARE_IMPL = textwrap.dedent("""
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
        return await handler(request)
""").strip()

_DEFAULT_AGENT_SETUP = "MIDDLEWARE = []"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_MUTATE_BASE = textwrap.dedent("""
## Procedure
1. Read asi.md — structured failure patterns extracted from this candidate's trajectories.
2. Read pareto/frontier.md — all current Pareto-frontier candidates and their scores.
3. Read history/history.md — every candidate ever attempted across all generations, with scores
   and proposals. Do NOT repeat an approach that was already tried (REJECTED) without a new angle.
4. Read history/failure_matrix.md — cross-generation persistence of each failure case.
5. Read train_cases/failures/ for the top RECURRING cases.
   Skip PERSISTENT cases unless you have a genuinely new angle not covered in asi.md.
6. Read train_cases/passing/ to verify the fix won't break passing cases.
7. Propose ONE change. Write to proposal.md:
   - Pattern: root cause
   - Fix count: estimated cases fixed
   - Risk count: passing cases at risk
   - Verdict: safe or not
8. If safe: apply fix to /current. Otherwise: write 'No safe fix found', leave current/ unchanged.

## Surface rules
- system_prompt.txt: behavioral rules only. No tool names, no specific values.
- middleware/: tool-level corrections. Specific logic goes here, not in the prompt.
- Never regress a FIXED case from failure_matrix.md.
- Never repeat a REJECTED approach from history/history.md unless you have a fundamentally new angle.
""").strip()

_MUTATE_PROMPT_ONLY = f"""You are PRISM-Mutator (prompt-only variant).

YOUR ONLY ALLOWED EDIT: current/system_prompt.txt. Do NOT touch current/middleware/.
Write universal behavioral rules only. Add at most one or two sentences.

{_MUTATE_BASE}"""

_MUTATE_MIDDLEWARE_ONLY = f"""You are PRISM-Mutator (middleware-only variant).

YOUR ONLY ALLOWED EDITS: current/middleware/custom_middleware.py and current/middleware/agent_setup.py
Do NOT touch current/system_prompt.txt.

Three middleware patterns:
1. Silent correction — mutate args before execution via request.override(tool_call=new_call)
2. Block with error (model retries): return ToolMessage(content="Error: ...", status="error")
3. Prerequisite block — verify message history before allowing a tool call

ALWAYS use request.override() to mutate args. NEVER mutate call["args"] directly.

{_MUTATE_BASE}"""

_MUTATE_BOTH = f"""You are PRISM-Mutator (full-access variant).

You have full access to both current/system_prompt.txt and current/middleware/.
Apply the single highest-value fix across either surface.

{_MUTATE_BASE}"""

_VARIANT_PROMPTS = {
    "prompt_only": _MUTATE_PROMPT_ONLY,
    "middleware_only": _MUTATE_MIDDLEWARE_ONLY,
    "prompt_middleware_both": _MUTATE_BOTH,
}

# ---------------------------------------------------------------------------
# NoConstraint ablation prompts — the three-pattern middleware vocabulary
# (silent correction / error blocking / prerequisite blocking) is removed and
# the mutator may edit middleware execution logic freely.  Everything else in
# the procedure is identical to the constrained variants.
# ---------------------------------------------------------------------------

_MUTATE_MIDDLEWARE_ONLY_UNCONSTRAINED = f"""You are PRISM-Mutator (middleware-only variant).

YOUR ONLY ALLOWED EDITS: current/middleware/custom_middleware.py and current/middleware/agent_setup.py
Do NOT touch current/system_prompt.txt.

You may implement any middleware logic you judge best — intercepting, rewriting,
reordering, injecting, or replacing tool calls and results however you see fit.

{_MUTATE_BASE}"""

_MUTATE_BOTH_UNCONSTRAINED = f"""You are PRISM-Mutator (full-access variant).

You have full access to both current/system_prompt.txt and current/middleware/.
Apply the single highest-value fix across either surface. Middleware logic is
unrestricted — implement whatever interception or rewriting you judge best.

{_MUTATE_BASE}"""

_VARIANT_PROMPTS_UNCONSTRAINED = {
    "prompt_only": _MUTATE_PROMPT_ONLY,
    "middleware_only": _MUTATE_MIDDLEWARE_ONLY_UNCONSTRAINED,
    "prompt_middleware_both": _MUTATE_BOTH_UNCONSTRAINED,
}

_CROSSOVER_ALL_PROMPT = textwrap.dedent("""
You are PRISM-Crossover, an evolutionary prompt merger.

Your job: given N pattern-focused mutation children, produce ONE merged child that
combines their complementary fixes into a single coherent prompt.

## Procedure
1. Read children/complement.md — which cases each child uniquely fixes.
2. For each child_N: read children/child_N.txt (prompt) and children/child_N_asi.md.
3. Read history/failure_matrix.md for cross-generation persistence.
4. For each child, identify the ONE key behavioral change that fixed its target pattern.
5. Compose a merged prompt incorporating ALL unique behavioral changes.
   - Compatible changes: keep both.
   - Conflicting changes: keep the one fixing more unique cases.
6. Write merged prompt to /current/system_prompt.txt.
7. Merge all middleware into /current/middleware/ (keep all classes active in MIDDLEWARE list).
8. Update proposal.md: for each child, state which unique fix you preserved.
""").strip()

_MUTATE_HUMAN = (
    "Follow your instructions: read asi.md, pareto/frontier.md, history/history.md, "
    "history/failure_matrix.md, then train_cases/failures/ for RECURRING cases, "
    "then train_cases/passing/ to verify safety. "
    "Write verdict to proposal.md. Apply fix to /current only if safe."
)
_CROSSOVER_ALL_HUMAN = (
    "Read children/complement.md, each child's prompt and ASI, and history/failure_matrix.md. "
    "Identify the key fix from each child, compose a merged prompt, "
    "write to /current/system_prompt.txt, update proposal.md."
)


# ---------------------------------------------------------------------------
# LLM-driven root-cause pattern analysis
# ---------------------------------------------------------------------------

_ANALYZE_SYSTEM = textwrap.dedent("""
You are PRISM-Analyst. Your job is to analyze failing benchmark cases and group them
by ROOT CAUSE — not by which tool happened to be last or which API field differs.

Root cause means: what behavioral mistake or missing knowledge caused the failure?
Use the full tool_sequence to understand what the agent actually did before failing —
do not infer behavior from tool names alone; look at the sequence and where it stops.
Examples of root causes:
- "Agent escalates or gives up prematurely without resolving all identified sub-tasks"
- "Agent gives up after a tool fails without retrying or satisfying prerequisites"
- "Agent posts tweet with wrong content — reads file but ignores explicit instructions"
- "Agent uses echo to write file but echo appends instead of overwrites"
- "Agent stops after first sub-task without completing remaining goals"

For each root cause cluster, also decide the best fix surface:
- "prompt_only"           — purely behavioral: instruction, reminder, ordering rule
- "middleware_only"       — tool-level: intercept/correct specific tool call args/sequence
- "prompt_middleware_both" — needs both a behavioral rule AND a tool-level correction

Output valid JSON only. No prose before or after. Schema:
{
  "clusters": [
    {
      "root_cause": "<one sentence>",
      "fix_surface": "prompt_only" | "middleware_only" | "prompt_middleware_both",
      "case_ids": ["case_id_1", ...],
      "reasoning": "<one sentence why this surface>"
    },
    ...
  ]
}

Rules:
- Produce exactly N clusters (N will be specified in the user message).
- A case_id may appear in multiple clusters if it involves both a behavioral and a tool-level issue.
- Skip infra_error cases (stuck_type=infra_error) — do not include them in any cluster.
- Order clusters by descending case count (most impactful first).
- Keep reasoning brief — one sentence.
- Prefer "prompt_only" or "middleware_only" over "prompt_middleware_both" whenever the fix
  clearly belongs to one surface. Only use "prompt_middleware_both" when both surfaces are
  genuinely needed. Aim for at least half of clusters to be surface-specific.
""").strip()


def analyze_patterns(
    score: SplitScore,
    *,
    n: int,
    outer_model: str,
    workspace_dir: Path,
    fm_cases: dict[str, str] | None = None,
) -> tuple[list[dict], int, int]:
    """Run outer LLM to group failures by root cause and recommend fix surface.

    Returns (patterns, outer_tokens_in, outer_tokens_out).
    patterns: list of dicts with keys: root_cause, fix_surface, case_ids, reasoning.
    Falls back to benchmark extract_top_patterns heuristic on any failure.
    """
    failures = [c for c in score.cases if not c.passed and c.stuck_type != "infra_error"]
    if not failures:
        return []

    # Build failure summary with full tool call sequence so the LLM can diagnose
    # actual behavior rather than inferring from empty last_tools on timeout cases.
    failure_lines = []
    for c in failures:
        tool_calls = c.extra.get("tool_calls", [])
        # Full sequence: name(args_truncated) for each call
        call_seq = []
        for tc in tool_calls:
            args = tc.get("args") or tc.get("arguments") or {}
            args_str = json.dumps(args)[:80].replace("\n", " ") if args else ""
            call_seq.append(f"{tc['name']}({args_str})" if args_str else tc["name"])
        calls_str = " → ".join(call_seq) if call_seq else "no tool calls"
        detail = (c.extra.get("state_diff") or c.extra.get("error") or "")[:200].replace("\n", " ")
        failure_lines.append(
            f"- case_id={c.case_id} stuck_type={c.stuck_type or 'none'} "
            f"tool_sequence=[{calls_str}] detail={detail}"
        )

    user_msg = (
        f"Analyze these {len(failures)} failures and group into exactly {n} root-cause clusters.\n\n"
        + "\n".join(failure_lines)
    )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "analysis_input.md").write_text(user_msg)

    tok_in = 0
    tok_out = 0
    try:
        from langchain_core.messages import HumanMessage as _HM
        from langchain_core.messages import SystemMessage as _SM

        from agent_harness_optimizer.utils.llm import build_model

        model = build_model(outer_model)
        response = model.invoke([_SM(content=_ANALYZE_SYSTEM), _HM(content=user_msg)])
        raw = response.content if hasattr(response, "content") else str(response)

        # Extract token usage from LangChain AIMessage
        usage = getattr(response, "usage_metadata", None) or getattr(
            response, "response_metadata", {}
        ).get("usage", {})
        if usage:
            tok_in = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            tok_out = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)
        clusters = result.get("clusters", [])

        # Validate — every cluster needs required fields
        valid = []
        for cl in clusters:
            if cl.get("root_cause") and cl.get("fix_surface") and cl.get("case_ids"):
                if cl["fix_surface"] not in (
                    "prompt_only",
                    "middleware_only",
                    "prompt_middleware_both",
                ):
                    cl["fix_surface"] = "prompt_middleware_both"
                valid.append(cl)

        if valid:
            (workspace_dir / "analysis_output.json").write_text(json.dumps(valid, indent=2))
            print(f"[prism] Pattern analysis: {len(valid)} clusters identified by LLM")
            for i, cl in enumerate(valid):
                print(
                    f"  cluster {i}: [{cl['fix_surface']}] {cl['root_cause'][:70]} ({len(cl['case_ids'])} cases)"
                )
            return valid, tok_in, tok_out

    except Exception as exc:
        print(f"[prism] Pattern analysis failed ({exc}), falling back to heuristic")

    return [], tok_in, tok_out


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _setup_current(workspace: Path, candidate: Candidate, prompt_only: bool = False) -> None:
    current = workspace / "current"
    current.mkdir()
    (current / "system_prompt.txt").write_text(candidate.prompt)
    if not prompt_only:
        mw = current / "middleware"
        mw.mkdir()
        if candidate.middleware_dir and candidate.middleware_dir.is_dir():
            for f in candidate.middleware_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, mw / f.name)
        else:
            (mw / "custom_middleware.py").write_text(_DEFAULT_MIDDLEWARE_IMPL)
            (mw / "agent_setup.py").write_text(_DEFAULT_AGENT_SETUP)


def _write_surface_manifest(
    workspace: Path, prompt_only: bool = False, unconstrained: bool = False
) -> None:
    manifest: dict = {
        "system_prompt": {
            "kind": "prompt",
            "target": "inner agent system prompt",
            "file": "current/system_prompt.txt",
        },
    }
    if not prompt_only:
        manifest["middleware"] = {
            "kind": "code",
            "target": "LangChain middleware via current/middleware/agent_setup.py MIDDLEWARE list",
            "file": "current/middleware/",
            "note": (
                "Middleware logic is unrestricted — implement any interception or rewriting."
                if unconstrained
                else "ALWAYS use request.override(tool_call=new_call) to mutate args."
            ),
        }
    (workspace / "surface_manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_pareto_summary(workspace: Path, frontier: list[Candidate]) -> None:
    pareto_dir = workspace / "pareto"
    pareto_dir.mkdir(exist_ok=True)
    lines = ["# Pareto Frontier", "", f"{len(frontier)} non-dominated candidates:", ""]
    for c in sorted(frontier, key=lambda x: -x.pass_rate):
        lines.append(
            f"- {c.uid} gen={c.generation} pass_rate={c.pass_rate:.3f} "
            f"reliability={c.reliability:.3f} train={c.train_passed}/{c.train_total} "
            f"holdout={c.holdout_passed}/{c.holdout_total}"
        )
        if c.proposal:
            lines.append(f"  proposal: {c.proposal[:120]}")
    (pareto_dir / "frontier.md").write_text("\n".join(lines))


def _write_generation_history(
    workspace: Path,
    all_candidates: list[Candidate],
    frontier: list[Candidate],
) -> None:
    """Write history/history.md covering every candidate ever attempted.

    For each generation, lists all candidates tried with their scores, proposals,
    and whether they ended up on the frontier — so the proposer knows what has
    already been attempted and what worked.
    """
    hist_dir = workspace / "history"
    hist_dir.mkdir(exist_ok=True)

    if not all_candidates:
        (hist_dir / "history.md").write_text("# Generation History\n\nNo history yet.\n")
        return

    frontier_uids = {c.uid for c in frontier}
    by_gen: dict[int, list[Candidate]] = {}
    for c in all_candidates:
        by_gen.setdefault(c.generation, []).append(c)

    lines = [
        "# Generation History",
        "",
        "Status: FRONTIER=currently accepted  REJECTED=tried but not on frontier",
        "",
    ]
    for gen in sorted(by_gen):
        lines.append(f"## Generation {gen}")
        for c in sorted(by_gen[gen], key=lambda x: x.uid):
            status = "FRONTIER" if c.uid in frontier_uids else "REJECTED"
            lines.append(
                f"- {c.uid} [{status}] "
                f"pass_rate={c.pass_rate:.3f} "
                f"train={c.train_passed}/{c.train_total} "
                f"holdout={c.holdout_passed}/{c.holdout_total} "
                f"reliability={c.reliability:.3f}"
            )
            if c.proposal:
                lines.append(f"  proposal: {c.proposal[:160]}")
        lines.append("")

    (hist_dir / "history.md").write_text("\n".join(lines))


def _write_failure_matrix(
    workspace: Path, fm_cases: dict[str, str] | None, score: SplitScore
) -> None:
    hist_dir = workspace / "history"
    hist_dir.mkdir(exist_ok=True)
    if not fm_cases:
        (hist_dir / "failure_matrix.md").write_text("# Failure Matrix\n\nNo history yet.\n")
        return
    failures = [c for c in score.cases if not c.passed]
    order = {"NEW": 0, "RECURRING": 1, "PERSISTENT": 2}
    lines = [
        "# Failure Matrix",
        "",
        "Labels:",
        "  NEW        = was passing on accepted prompt, now failing (regression — high priority)",
        "  RECURRING  = failing now, but a prior mutation (accepted OR rejected) did fix it (solvable)",
        "  PERSISTENT = failed across every accepted generation AND every rejected mutation attempt",
        "  FIXED      = was failing before, now passing on current prompt",
        "",
        "Priority order: NEW > RECURRING > PERSISTENT",
        "PERSISTENT cases are the hardest — every attempt so far has failed them.",
        "",
    ]
    for c in sorted(failures, key=lambda x: order.get(fm_cases.get(x.case_id, "PERSISTENT"), 2)):
        lines.append(f"- {c.case_id}: {fm_cases.get(c.case_id, '?')}")
    (hist_dir / "failure_matrix.md").write_text("\n".join(lines))


def _write_mutate_task(
    workspace: Path,
    candidate: Candidate,
    score: SplitScore,
    generation: int,
    target_pattern: dict | None,
) -> None:
    failures = [c for c in score.cases if not c.passed]
    if target_pattern:
        target_ids = set(target_pattern["case_ids"])
        display = [c for c in failures if c.case_id in target_ids]
    else:
        display = failures

    failure_lines = []
    for c in display:
        tool_calls = c.extra.get("tool_calls", [])
        calls = ", ".join(tc["name"] for tc in tool_calls[-4:]) or "none"
        detail = (c.extra.get("state_diff") or c.extra.get("error") or "")[:150].replace("\n", " ")
        failure_lines.append(f"- `{c.case_id}`: last calls=[{calls}] detail={detail}")

    pattern_section = []
    if target_pattern:
        root_cause = target_pattern.get("root_cause") or target_pattern.get("key", "")
        fix_surface = target_pattern.get("fix_surface", "")
        reasoning = target_pattern.get("reasoning", "")
        case_ids = target_pattern.get("case_ids", [])
        pattern_section = [
            "",
            "## YOUR TARGET ROOT CAUSE — fix ONLY these cases, ignore other failures",
            f"- root_cause: {root_cause}",
            f"- fix_surface: {fix_surface}" if fix_surface else "",
            f"- reasoning: {reasoning}" if reasoning else "",
            f"- cases ({len(case_ids)}): {case_ids[:10]}",
            "",
            "One precise fix for this root cause only. Do not touch other patterns.",
        ]
        pattern_section = [line for line in pattern_section if line != ""]

    (workspace / "task.md").write_text(
        "\n".join(
            [
                "# PRISM Mutation Task",
                f"Generation: {generation}  Candidate: {candidate.uid}",
                f"Score: {score.passed}/{score.total} (pass_rate={candidate.pass_rate:.3f} reliability={candidate.reliability:.3f})",
                *pattern_section,
                "",
                "Rules:",
                "- Edit only files under `current/`.",
                "- Read asi.md FIRST.",
                "- Read pareto/frontier.md, history/failure_matrix.md.",
                "- Read train_cases/failures/ for target cases, train_cases/passing/ for safety.",
                "- One fix only. Smallest safe change.",
                "- If no safe fix: write 'No safe fix found' in proposal.md.",
                "",
                "Target failures:",
                *failure_lines,
                "",
            ]
        )
        + "\n"
    )


def _build_mutate_workspace(
    *,
    workspace: Path,
    candidate: Candidate,
    score: SplitScore,
    generation: int,
    frontier: list[Candidate],
    all_candidates: list[Candidate],
    fm_cases: dict[str, str] | None,
    target_pattern: dict | None,
    benchmark: Benchmark,
    variant: str = "prompt_middleware_both",
    unconstrained: bool = False,
) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    _prompt_only = variant == "prompt_only"
    _setup_current(workspace, candidate, prompt_only=_prompt_only)
    _write_surface_manifest(workspace, prompt_only=_prompt_only, unconstrained=unconstrained)
    (workspace / "asi.md").write_text(benchmark.build_asi(score, fm_cases))
    _write_pareto_summary(workspace, frontier)
    _write_generation_history(workspace, all_candidates, frontier)
    _write_failure_matrix(workspace, fm_cases, score)
    # prompt_middleware_both sees all failures (BH behavior); surface-specific variants
    # see only their target cluster's cases.
    target_ids = (
        set(target_pattern["case_ids"])
        if target_pattern and variant != "prompt_middleware_both"
        else None
    )
    benchmark.write_case_files(workspace, score, target_case_ids=target_ids)
    _write_mutate_task(workspace, candidate, score, generation, target_pattern)
    (workspace / "variant.txt").write_text(
        (target_pattern.get("root_cause") or target_pattern.get("key", "auto"))[:60]
        if target_pattern
        else "auto"
    )
    (workspace / "proposal.md").write_text(
        "# Proposal\n\n- Pattern:\n- Fix count:\n- Risk count:\n- Verdict:\n"
    )


def _build_crossover_all_workspace(
    *,
    workspace: Path,
    base_candidate: Candidate,
    children: list[Candidate],
    scores: list[SplitScore],
    generation: int,
    frontier: list[Candidate],
    all_candidates: list[Candidate],
    fm_cases: dict[str, str] | None,
    benchmark: Benchmark,
) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    _setup_current(workspace, base_candidate)
    _write_surface_manifest(workspace)
    _write_pareto_summary(workspace, frontier)
    _write_generation_history(workspace, all_candidates, frontier)
    _write_failure_matrix(workspace, fm_cases, scores[0])

    children_dir = workspace / "children"
    children_dir.mkdir()

    all_passing = [set(c.case_id for c in s.cases if c.passed) for s in scores]
    unique_per = [
        p - set().union(*(q for j, q in enumerate(all_passing) if j != i))
        for i, p in enumerate(all_passing)
    ]

    for i, (child, score, unique) in enumerate(zip(children, scores, unique_per)):
        label = f"child_{i}"
        (children_dir / f"{label}.txt").write_text(child.prompt)
        (children_dir / f"{label}_asi.md").write_text(benchmark.build_asi(score, fm_cases))
        (children_dir / f"{label}_info.md").write_text(
            "\n".join(
                [
                    f"# {label}: {child.uid}",
                    f"pass_rate={child.pass_rate:.3f} reliability={child.reliability:.3f}",
                    f"proposal: {(child.proposal or '')[:200]}",
                    "",
                    f"Unique cases this child passes ({len(unique)}): {sorted(unique)[:15]}",
                ]
            )
        )

    (children_dir / "complement.md").write_text(
        "\n".join(
            [
                "# Crossover Complement Analysis",
                "",
                f"Base: {base_candidate.uid}",
                "",
                *[
                    f"child_{i} ({c.uid}): pass_rate={c.pass_rate:.3f} — "
                    f"{len(u)} unique passes: {sorted(u)[:10]}"
                    for i, (c, u) in enumerate(zip(children, unique_per))
                ],
                "",
                "Goal: merge ALL unique behavioral fixes into one coherent prompt.",
            ]
        )
    )

    (workspace / "task.md").write_text(
        "\n".join(
            [
                "# PRISM Crossover-All Task",
                f"Generation: {generation}  Children: {[c.uid for c in children]}",
                "",
                "Rules:",
                "- Each child fixed ONE failure pattern. Merge all their fixes into one prompt.",
                "- Read children/complement.md, each child's prompt and ASI.",
                "- Read history/failure_matrix.md.",
                "- Identify the ONE key rule each child added. Compose a merged prompt with all rules.",
                "- Write to /current/system_prompt.txt. Merge middleware. Update proposal.md.",
                "",
            ]
        )
    )
    (workspace / "proposal.md").write_text(
        "# Crossover-All Proposal\n\n"
        + "\n".join(f"- child_{i} fix preserved:" for i in range(len(children)))
        + "\n- Conflicts resolved:\n"
    )


# ---------------------------------------------------------------------------
# Surface reader
# ---------------------------------------------------------------------------


def _read_surfaces(workspace: Path) -> tuple[str, Path | None]:
    prompt = (workspace / "current" / "system_prompt.txt").read_text().strip()
    mw = workspace / "current" / "middleware"
    middleware_dir = None
    if mw.is_dir():
        setup = mw / "agent_setup.py"
        if setup.exists():
            has_nonempty = any(
                _re.search(r"MIDDLEWARE\s*=\s*\[.+\]", line)
                for line in setup.read_text().splitlines()
                if "MIDDLEWARE" in line and not line.strip().startswith("#")
            )
            if has_nonempty:
                middleware_dir = mw
    return prompt, middleware_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mutate(
    candidate: Candidate,
    score: SplitScore,
    *,
    generation: int,
    frontier: list[Candidate],
    all_candidates: list[Candidate] | None = None,
    fm_cases: dict[str, str] | None = None,
    benchmark: Benchmark,
    outer_model: str,
    workspace_dir: Path,
    max_turns: int = 200,
    variant: str | None = None,
    target_pattern: dict | None = None,
    unconstrained: bool = False,
) -> tuple[str, Path | None, str, str, int, int]:
    """Run mutation agent.  Returns (new_prompt, new_middleware_dir, proposal, variant_used, outer_tokens_in, outer_tokens_out).

    unconstrained=True (NoConstraint ablation): middleware edits are not limited
    to the three tool-boundary patterns; the mutator may edit execution logic freely.
    """
    if variant is None:
        variant = "prompt_middleware_both"

    pat_label = (
        f" pattern={(target_pattern.get('root_cause') or target_pattern.get('key', '?'))[:40]}"
        if target_pattern
        else ""
    )
    print(f"[prism]   mutate({candidate.uid}) variant={variant}{pat_label}")

    _build_mutate_workspace(
        workspace=workspace_dir,
        candidate=candidate,
        score=score,
        generation=generation,
        frontier=frontier,
        all_candidates=all_candidates or [],
        fm_cases=fm_cases,
        target_pattern=target_pattern,
        benchmark=benchmark,
        variant=variant,
        unconstrained=unconstrained,
    )

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    from agent_harness_optimizer.utils.llm import build_model

    model = build_model(outer_model)
    backend = FilesystemBackend(root_dir=str(workspace_dir), virtual_mode=True)
    _prompts = _VARIANT_PROMPTS_UNCONSTRAINED if unconstrained else _VARIANT_PROMPTS
    agent = create_deep_agent(model=model, system_prompt=_prompts[variant], backend=backend)
    result = agent.invoke(
        {"messages": [HumanMessage(content=_MUTATE_HUMAN)]},
        config={"recursion_limit": max(max_turns * 3, 900)},
    )
    prompt, mw_dir = _read_surfaces(workspace_dir)
    proposal = (
        (workspace_dir / "proposal.md").read_text().strip()
        if (workspace_dir / "proposal.md").exists()
        else ""
    )

    tok_in = 0
    tok_out = 0
    for msg in (result or {}).get("messages", []):
        usage = getattr(msg, "usage_metadata", None) or getattr(msg, "response_metadata", {}).get(
            "usage", {}
        )
        if usage:
            tok_in += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            tok_out += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

    return prompt, mw_dir, proposal, variant, tok_in, tok_out


def crossover_all_children(
    children: list[Candidate],
    scores: list[SplitScore],
    base_candidate: Candidate,
    *,
    generation: int,
    frontier: list[Candidate],
    all_candidates: list[Candidate] | None = None,
    fm_cases: dict[str, str] | None = None,
    benchmark: Benchmark,
    outer_model: str,
    workspace_dir: Path,
    max_turns: int = 200,
) -> tuple[str, Path | None, str, int, int]:
    """Merge all pattern-focused children.  Returns (new_prompt, new_middleware_dir, proposal, outer_tokens_in, outer_tokens_out)."""
    print(f"[prism]   crossover_all({len(children)} children, base={base_candidate.uid})")
    _build_crossover_all_workspace(
        workspace=workspace_dir,
        base_candidate=base_candidate,
        children=children,
        scores=scores,
        generation=generation,
        frontier=frontier,
        all_candidates=all_candidates or [],
        fm_cases=fm_cases,
        benchmark=benchmark,
    )
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    from agent_harness_optimizer.utils.llm import build_model

    model = build_model(outer_model)
    backend = FilesystemBackend(root_dir=str(workspace_dir), virtual_mode=True)
    agent = create_deep_agent(model=model, system_prompt=_CROSSOVER_ALL_PROMPT, backend=backend)
    result = agent.invoke(
        {"messages": [HumanMessage(content=_CROSSOVER_ALL_HUMAN)]},
        config={"recursion_limit": max(max_turns * 3, 900)},
    )
    prompt, mw_dir = _read_surfaces(workspace_dir)
    proposal = (
        (workspace_dir / "proposal.md").read_text().strip()
        if (workspace_dir / "proposal.md").exists()
        else ""
    )

    tok_in = 0
    tok_out = 0
    for msg in (result or {}).get("messages", []):
        usage = getattr(msg, "usage_metadata", None) or getattr(msg, "response_metadata", {}).get(
            "usage", {}
        )
        if usage:
            tok_in += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            tok_out += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

    return prompt, mw_dir, proposal, tok_in, tok_out
