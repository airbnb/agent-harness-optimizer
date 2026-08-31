"""MockBenchmark — deterministic fake benchmark for testing optimizers end-to-end.

Scoring is instant (no real agent runs).  Cases always pass or fail based on
whether the prompt contains a magic keyword for each case.

Use this to test optimizer loop logic (iteration counts, accept/reject gates,
Pareto updates, workspace layout) without spending money or time on real runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_harness_optimizer.framework.benchmark import Benchmark, CaseScore, ResourceBudget, SplitScore

# If prompt contains "fix_case_N", case N passes.
_TRAIN_CASES = [f"case_{i}" for i in range(10)]
_HOLDOUT_CASES = [f"holdout_{i}" for i in range(5)]


class MockBenchmark(Benchmark):
    """Deterministic benchmark for testing.

    Scoring rules:
      - Each case "case_N" or "holdout_N" passes if the prompt contains "fix_N"
        OR if the prompt contains "fix_all".
      - stuck_type="loop" for case_2 and case_7 unless they pass.
    """

    def __init__(self, budget: ResourceBudget | None = None) -> None:
        self._budget = budget or ResourceBudget(wall_time_s=1.0, max_steps=5, max_tokens=1000)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-model"

    @property
    def default_system_prompt(self) -> str:
        return "You are a test agent."

    @property
    def resource_budget(self) -> ResourceBudget:
        return self._budget

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
        await asyncio.sleep(0)  # yield to event loop
        output_dir.mkdir(parents=True, exist_ok=True)
        case_ids = _TRAIN_CASES if split == "train" else _HOLDOUT_CASES
        if case_indices is not None:
            case_ids = [case_ids[i] for i in case_indices if i < len(case_ids)]
        elif max_cases is not None:
            case_ids = case_ids[:max_cases]

        cases = []
        for cid in case_ids:
            num = cid.split("_")[-1]
            passed = f"fix_{num}" in prompt or "fix_all" in prompt
            stuck = "loop" if not passed and num in ("2", "7") else ""
            cases.append(
                CaseScore(
                    case_id=cid,
                    passed=passed,
                    stuck_type=stuck,
                    prompt_tokens=50,
                    completion_tokens=20,
                    extra={"tool_calls": [], "error": "" if passed else f"failed_{cid}"},
                )
            )

        passed_count = sum(1 for c in cases if c.passed)
        stuck_count = sum(1 for c in cases if c.stuck_type)
        reliability = round(1.0 - (stuck_count / len(cases) if cases else 0.0), 4)
        return SplitScore(
            passed=passed_count,
            total=len(cases),
            reliability=reliability,
            tokens_per_case=70.0,
            cases=cases,
        )

    def build_asi(self, score: SplitScore, failure_matrix_cases) -> str:
        failures = [c for c in score.cases if not c.passed]
        lines = ["# Mock ASI", f"Failures: {len(failures)}/{score.total}", ""]
        for c in failures:
            lines.append(f"- {c.case_id}: stuck={c.stuck_type} error={c.extra.get('error', '')}")
        return "\n".join(lines)

    def extract_top_patterns(self, score: SplitScore, n: int = 3) -> list[dict]:
        failures = [c for c in score.cases if not c.passed]
        clusters: dict[str, dict] = {}
        for c in failures:
            key = c.stuck_type or "clean_fail"
            if key not in clusters:
                clusters[key] = {"key": key, "count": 0, "case_ids": []}
            clusters[key]["count"] += 1
            clusters[key]["case_ids"].append(c.case_id)
        return sorted(clusters.values(), key=lambda x: -x["count"])[:n]

    def write_case_files(self, workspace: Path, score: SplitScore) -> None:
        for d in ("failures", "passing"):
            (workspace / "train_cases" / d).mkdir(parents=True, exist_ok=True)
        for c in score.cases:
            dest = workspace / "train_cases" / ("passing" if c.passed else "failures")
            (dest / f"{c.case_id}.json").write_text(json.dumps(c.to_dict(), indent=2))

    def build_model(self, model_name: str):
        return None  # no-op; optimizers call this for auth init, not model creation
