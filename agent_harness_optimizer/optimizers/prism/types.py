"""PRISM candidate types — benchmark-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Candidate:
    """One prompt (+ optional middleware) on the Pareto frontier."""

    uid: str
    generation: int
    prompt: str
    middleware_dir: Path | None

    train_passed: int = 0
    train_total: int = 0
    holdout_passed: int = 0
    holdout_total: int = 0
    screen_passed: int = 0
    screen_total: int = 0

    pass_rate: float = 0.0
    reliability: float = 0.0  # 1 - stuck_rate

    # Token counts (separate for cost estimation)
    prompt_tokens_per_case: float = 0.0
    completion_tokens_per_case: float = 0.0

    # Wall-clock timing
    eval_wall_time_s: float = 0.0  # seconds to score train+holdout
    proposal_wall_time_s: float = 0.0  # seconds for outer LLM proposal

    parent_uids: list[str] = field(default_factory=list)
    proposal: str = ""
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def holdout_pass_rate(self) -> float:
        return self.holdout_passed / self.holdout_total if self.holdout_total else 0.0

    @property
    def train_pass_rate(self) -> float:
        return self.train_passed / self.train_total if self.train_total else 0.0

    def dominates(self, other: Candidate, objective: str = "holdout") -> bool:
        # Pareto dominance on pass rate + reliability.
        # objective="holdout" (default): holdout-only pass rate, so train-holdout
        # tradeoffs don't discard candidates that improve holdout at the cost of train.
        # objective="train": used by the NoGate ablation, where the gate (holdout)
        # split is not consulted during search — dominance runs on train only.
        if objective == "train":
            a, b = self.train_pass_rate, other.train_pass_rate
        else:
            a, b = self.holdout_pass_rate, other.holdout_pass_rate
        return (
            a >= b
            and self.reliability >= other.reliability
            and (a > b or self.reliability > other.reliability)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "generation": self.generation,
            "prompt": self.prompt,
            "middleware_dir": str(self.middleware_dir) if self.middleware_dir else None,
            "train_passed": self.train_passed,
            "train_total": self.train_total,
            "holdout_passed": self.holdout_passed,
            "holdout_total": self.holdout_total,
            "screen_passed": self.screen_passed,
            "screen_total": self.screen_total,
            "pass_rate": self.pass_rate,
            "reliability": self.reliability,
            "prompt_tokens_per_case": self.prompt_tokens_per_case,
            "completion_tokens_per_case": self.completion_tokens_per_case,
            "eval_wall_time_s": self.eval_wall_time_s,
            "proposal_wall_time_s": self.proposal_wall_time_s,
            "parent_uids": self.parent_uids,
            "proposal": self.proposal,
            "per_case": self.per_case,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Candidate:
        c = Candidate(
            uid=d["uid"],
            generation=d["generation"],
            prompt=d["prompt"],
            middleware_dir=Path(d["middleware_dir"]) if d.get("middleware_dir") else None,
        )
        for k in (
            "train_passed",
            "train_total",
            "holdout_passed",
            "holdout_total",
            "screen_passed",
            "screen_total",
            "pass_rate",
            "reliability",
            "prompt_tokens_per_case",
            "completion_tokens_per_case",
            "eval_wall_time_s",
            "proposal_wall_time_s",
            "parent_uids",
            "proposal",
            "per_case",
        ):
            if k in d:
                setattr(c, k, d[k])
        return c


def pareto_frontier(candidates: list[Candidate], objective: str = "holdout") -> list[Candidate]:
    return [
        c
        for c in candidates
        if not any(
            other.dominates(c, objective=objective) for other in candidates if other is not c
        )
    ]
