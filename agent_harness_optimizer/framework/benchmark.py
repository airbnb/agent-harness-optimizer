"""Benchmark — task-specific half of the eval-optimizer framework.

Swap axis 1: replace this to change *what* is being optimized.
Implementations live in benchmarks/bfcl.py, benchmarks/tau_bench.py, etc.

## Resource budget contract

Each benchmark declares its own ResourceBudget — three independent axes:

    T_wall  (wall_time_s)  — hard wall-clock timeout per case; SIGKILL on breach
    S_steps (max_steps)    — stop agent after this many tool calls / turns
    C_tokens (max_tokens)  — stop agent after this many cumulative prompt+completion tokens

Stuck taxonomy (generic across all benchmarks):
    ""            — clean pass or clean fail, no budget violation
    "timeout"     — killed by T_wall
    "step_limit"  — stopped by S_steps
    "token_limit" — stopped by C_tokens
    "loop"        — detected infinite tool repetition
    "crash"       — unhandled exception before terminal state
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resource budget
# ---------------------------------------------------------------------------


@dataclass
class ResourceBudget:
    """Three-axis resource budget.  Set an axis to None to disable it."""

    wall_time_s: float | None = 300.0  # T_wall
    max_steps: int | None = 100  # S_steps
    max_tokens: int | None = 500_000  # C_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
        }


# ---------------------------------------------------------------------------
# Per-case result
# ---------------------------------------------------------------------------


@dataclass
class CaseScore:
    """Result of running one case.

    Benchmark-specific data (tool_calls, error text, state diffs, reward, etc.)
    lives in `extra` and is only read by the same Benchmark's analysis methods.
    The optimizer loops never read `extra` directly.
    """

    case_id: str
    passed: bool
    stuck_type: str = ""
    within_time_budget: bool = True
    within_step_budget: bool = True
    within_token_budget: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    category: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "case_id": self.case_id,
            "passed": self.passed,
            "stuck_type": self.stuck_type,
            "within_time_budget": self.within_time_budget,
            "within_step_budget": self.within_step_budget,
            "within_token_budget": self.within_token_budget,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_time_s": self.wall_time_s,
            "category": self.category,
        }
        d.update(self.extra)
        return d

    @staticmethod
    def from_dict(row: dict[str, Any]) -> CaseScore:
        known = {
            "case_id",
            "passed",
            "stuck_type",
            "within_time_budget",
            "within_step_budget",
            "within_token_budget",
            "prompt_tokens",
            "completion_tokens",
            "wall_time_s",
            "category",
        }
        return CaseScore(
            case_id=row["case_id"],
            passed=bool(row["passed"]),
            stuck_type=row.get("stuck_type", ""),
            within_time_budget=row.get("within_time_budget", True),
            within_step_budget=row.get("within_step_budget", True),
            within_token_budget=row.get("within_token_budget", True),
            prompt_tokens=int(row.get("prompt_tokens", 0)),
            completion_tokens=int(row.get("completion_tokens", 0)),
            wall_time_s=float(row.get("wall_time_s", 0.0)),
            category=row.get("category", ""),
            extra={k: v for k, v in row.items() if k not in known},
        )


# ---------------------------------------------------------------------------
# Split-level aggregated score
# ---------------------------------------------------------------------------


@dataclass
class SplitScore:
    """Aggregated score for one (prompt, split) evaluation run.

    The optimizer loops read only these fields:
      passed, total       — integer pass counts
      reliability         — 1.0 means zero stuck/crash cases
      tokens_per_case     — average tokens used (cost proxy for optimizer gates)
      cases               — full per-case list for ASI / workspace generation
    """

    passed: int
    total: int
    reliability: float
    tokens_per_case: float = 0.0
    prompt_tokens_per_case: float = 0.0
    completion_tokens_per_case: float = 0.0
    cases: list[CaseScore] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def stuck_rate(self) -> float:
        return 1.0 - self.reliability

    @property
    def stuck_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cases:
            if c.stuck_type:
                out[c.stuck_type] = out.get(c.stuck_type, 0) + 1
        return out

    @property
    def infra_error_rate(self) -> float:
        """Fraction of cases that failed due to infrastructure errors (deployment/rate limit)."""
        return (
            sum(c.stuck_type == "infra_error" for c in self.cases) / self.total
            if self.total
            else 0.0
        )

    def is_valid(self, max_infra_error_rate: float = 0.05) -> bool:
        """Returns False if infra_error_rate exceeds threshold — run should be discarded."""
        return self.infra_error_rate <= max_infra_error_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total": self.total,
            "reliability": self.reliability,
            "tokens_per_case": self.tokens_per_case,
            "prompt_tokens_per_case": self.prompt_tokens_per_case,
            "completion_tokens_per_case": self.completion_tokens_per_case,
            "infra_error_rate": round(self.infra_error_rate, 4),
            "is_valid": self.is_valid(),
            "stuck_breakdown": self.stuck_breakdown,
            "per_case": [c.to_dict() for c in self.cases],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SplitScore:
        return SplitScore(
            passed=int(d["passed"]),
            total=int(d["total"]),
            reliability=float(d.get("reliability", 1.0)),
            tokens_per_case=float(d.get("tokens_per_case", 0.0)),
            prompt_tokens_per_case=float(d.get("prompt_tokens_per_case", 0.0)),
            completion_tokens_per_case=float(d.get("completion_tokens_per_case", 0.0)),
            cases=[CaseScore.from_dict(r) for r in d.get("per_case", [])],
        )


# ---------------------------------------------------------------------------
# Benchmark ABC
# ---------------------------------------------------------------------------


class Benchmark(ABC):
    """Abstract base every benchmark must implement.

    Subclass this for BFCL, tau-bench, terminal tasks, etc.
    The optimizer loops call only the methods defined here.
    """

    # --- identity ---

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in log messages and output directory names."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default inner (scored) model name string."""

    @property
    def default_system_prompt(self) -> str:
        """Seed system prompt for the initial candidate."""
        return ""

    @property
    def resource_budget(self) -> ResourceBudget:
        """Resource budget applied to every case.

        Benchmarks own budget enforcement — the optimizer never passes budget
        params to score_async.
        """
        return ResourceBudget()

    # --- scoring ---

    @abstractmethod
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
        """Score `prompt` on `split`. Enforces self.resource_budget internally.

        max_cases: optional subsample limit for cheap screen evaluations.
                   Ignored when case_indices is provided.
        case_indices: explicit list of indices into the benchmark's task list.
                      When provided, evaluates exactly these cases (disjoint
                      train/holdout splits become possible).  When None, falls
                      back to tasks[:max_cases] for backward compatibility.
        middleware_dir: only used by benchmarks that support middleware (e.g. BFCL).
                        Non-middleware benchmarks ignore this parameter.
        """

    # --- analysis helpers ---

    @abstractmethod
    def build_asi(
        self,
        score: SplitScore,
        failure_matrix_cases: dict[str, str] | None,
    ) -> str:
        """Return Actionable Side Information markdown string.

        failure_matrix_cases: {case_id -> "NEW"|"RECURRING"|"PERSISTENT"|"FIXED"}
        computed by the optimizer from cross-iteration history. None on iteration 1.
        """

    @abstractmethod
    def extract_top_patterns(self, score: SplitScore, n: int = 3) -> list[dict]:
        """Return the top N failure clusters for pattern-focused mutations.

        Each dict must contain at minimum:
          {"key": str, "count": int, "case_ids": list[str]}
        """

    @abstractmethod
    def write_case_files(
        self, workspace: Path, score: SplitScore, target_case_ids: set[str] | None = None
    ) -> None:
        """Write per-case files into workspace/train_cases/{failures,passing}/.

        If target_case_ids is provided, only those failure cases are written to
        failures/ — all other failures are omitted so the LLM focuses on the
        target pattern rather than being distracted by the most-frequent pattern.
        Passing cases are always written in full (needed for safety checks).
        """

    # --- proposer workspace customisation ---
    # These have sensible generic defaults. Override for domain-specific guidance.

    def get_proposer_variants(self) -> dict[str, str]:
        """Return {variant_name: system_prompt_text} for the outer proposal agents.

        Default: one generic prompt-only variant.
        Override to add domain-specific instructions or multiple variants
        (e.g. BFCL adds middleware_only and prompt_middleware_both variants).
        """
        return {"default": _GENERIC_PROPOSER_SYSTEM}

    def build_task_md(self, score: SplitScore, iteration: int) -> str:
        """Return task.md content shown to the outer proposal agent.

        The generic implementation is prompt-surface-only with no benchmark-specific
        fields. Override to add domain context (metrics, surface paths, etc.).
        """
        failures = [c for c in score.cases if not c.passed]
        lines = [
            "# Harness Improvement Task",
            "",
            f"Iteration: {iteration}",
            f"Score: {score.passed}/{score.total} passed  reliability={score.reliability:.3f}",
            "",
            "Rules:",
            "- Edit only files under `current/`.",
            "- Read train_cases/failures/ and train_cases/passing/ before proposing.",
            "- Read history/history.md — do NOT repeat anything already tried.",
            "- Read history/failure_matrix.md — focus on RECURRING, skip PERSISTENT.",
            "- One fix only. Smallest safe change.",
            "- Write verdict to proposal.md before touching current/.",
            "- If no safe fix: write 'No safe fix found', leave current/ unchanged.",
            "",
            "Failures:",
        ]
        for c in failures:
            detail = (c.extra.get("error") or "")[:200].replace("\n", " ")
            lines.append(f"- `{c.case_id}` [{c.category}]: {detail}")
        lines.append("")
        return "\n".join(lines)

    def select_best_variant(
        self,
        score: SplitScore,
        middleware_active: bool,
    ) -> str | None:
        """Return the variant name to run, or None to run all variants in parallel.

        Default: always run all variants (let the optimizer pick best by train score).
        Override for signal-based selection (e.g. BFCL checks dominant last tool).

        Benchmarks with only one variant should return that variant's name to avoid
        running unnecessary parallel jobs.
        """
        variants = self.get_proposer_variants()
        if len(variants) == 1:
            return next(iter(variants))
        return None

    def write_case_rollout(
        self,
        output_dir: Path,
        split: SplitScore,
        *,
        split_role: str = "",
        candidate_id: str = "baseline",
        experiment_id: str | None = None,
        repeat_id: int = 0,
        search_seed: int = 0,
    ) -> None:
        """Append per-case rows to {output_dir}/case_rollout.jsonl.

        Called by benchmark subclasses at the end of score_async() when output_dir is set.
        """
        if not split.cases:
            return
        rollout_path = output_dir / "case_rollout.jsonl"
        with rollout_path.open("a") as f:
            for c in split.cases:
                row = {
                    "experiment_id": experiment_id,
                    "optimizer_run_id": output_dir.name,
                    "benchmark": self.name,
                    "task_id": c.case_id,
                    "split_role": split_role,
                    "repeat_id": repeat_id,
                    "search_seed": search_seed,
                    "candidate_id": candidate_id,
                    "inner_model": self.default_model,
                    "pass_bool": c.passed,
                    "quality_score": 1.0 if c.passed else 0.0,
                    "validity_indicator": not bool(c.stuck_type),
                    "stuck_type": c.stuck_type,
                    "input_tokens": c.prompt_tokens,
                    "output_tokens": c.completion_tokens,
                    "wall_clock_seconds": c.wall_time_s,
                }
                f.write(json.dumps(row) + "\n")

    def get_default_middleware_stubs(self) -> dict[str, str] | None:
        """Return {filename: content} for default middleware scaffolding, or None.

        None (default) means this benchmark does not use middleware — the optimizer
        will not create a middleware/ directory in the workspace.

        Override in benchmarks that run agents on a middleware-capable stack
        (e.g. BFCLBenchmark returns custom_middleware.py + agent_setup.py stubs).
        """
        return None

    # --- model factory ---

    @abstractmethod
    def build_model(self, model_name: str):
        """Return a LangChain BaseChatModel for the given model_name string."""


# ---------------------------------------------------------------------------
# Generic proposer system prompt (prompt-only, no middleware references)
# ---------------------------------------------------------------------------

_GENERIC_PROPOSER_SYSTEM = """You are an expert agent optimizer.

Given a set of failing cases, propose ONE improvement to the system prompt
that will fix the most cases without breaking passing ones.

## Analysis procedure
1. Read task.md for the score and failure list.
2. Read ALL files in train_cases/failures/ — understand what went wrong.
3. Read ALL files in train_cases/passing/ — understand what succeeds.
4. Build a frequency table: for each root cause, count (a) failing cases fixed,
   (b) passing cases at risk of regression.
5. Pick ONE pattern with the highest fix/risk ratio.
6. Write to proposal.md: Pattern, Fix count, Risk count, Verdict (safe or not).
7. Only if safe: edit current/system_prompt.txt.

## Rules
- Edit only current/system_prompt.txt.
- Write universal behavioral rules. No task-specific values or identifiers.
- Add at most one or two sentences — do not rewrite the whole prompt.
- If no safe fix: write 'No safe fix found' in proposal.md, leave current/ unchanged."""
