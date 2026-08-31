"""Optimizer — loop-strategy half of the eval-optimizer framework.

Swap axis 2: replace this to change *how* prompts are optimized.
Concrete implementations: optimizers/prism/ and optimizers/better_harness/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent_harness_optimizer.framework.benchmark import Benchmark

if TYPE_CHECKING:
    from agent_harness_optimizer.framework.acceptance import AcceptanceCriterion
    from agent_harness_optimizer.utils.splits import CaseSplit


@dataclass
class OptimizeConfig:
    """Configuration shared by all optimizers."""

    output_dir: Path
    inner_model: str  # model used to score cases (inner agent)
    outer_model: str  # model used to propose changes (outer agent)
    train_split: str = "train"
    holdout_split: str = "holdout"
    outer_max_turns: int = 300
    resume: bool = False
    human_approval: bool = False
    # When set, all baseline + final-eval score_async calls use these pinned
    # case indices instead of tasks[:max_cases], ensuring disjoint and
    # apple-to-apple train/holdout sets across all four optimizer runs.
    case_split: CaseSplit | None = field(default=None, repr=False)
    # When set (0-3), a CV fold is active and all optimizers run a final
    # out-of-sample scorecard eval on the 600 held-out cases.
    split_seed: int | None = None

    @property
    def scorecard_case_indices(self) -> list[int] | None:
        """Case indices for the scorecard split, or None (score all cases).

        For BFCL the scorecard cases are determined by split_seed inside
        score_async via _load_split_cases; case_indices is left as None.
        For tau benchmarks the scorecard is stored in case_split.scorecard
        (populated by stratified make_split); pass those explicitly.
        """
        cs = self.case_split
        if cs is not None and cs.scorecard:
            return cs.scorecard
        return None

    # Acceptance criterion plug-in. Default: HoldoutPassRate (strict holdout improvement).
    # Pass HoldoutPareto() to use (holdout_pass_rate, reliability) Pareto dominance.
    acceptance: AcceptanceCriterion | None = field(default=None, repr=False)
    # When set, all optimizers load the pre-scored baseline from this directory
    # (baseline/train.json and baseline/holdout.json) instead of re-scoring.
    # This ensures all optimizers in a fold share the exact same baseline numbers.
    shared_baseline_dir: Path | None = None

    # EMNLP experiment identity fields (written to report.json)
    experiment_id: str | None = None  # e.g. "bfcl-bh-s0-r2"
    condition_id: str | None = None  # e.g. "bfcl_random_s0"
    repeat_id: int = 0  # search repeat index 0,1,2,…
    search_seed: int = 0  # stochastic seed for proposer LLM
    num_scorecard_trials: int = 1  # k for pass^k final scorecard eval (tau benchmarks only)


class Optimizer(ABC):
    """Abstract base for all optimizer loop strategies.

    Usage::

        benchmark = BFCLBenchmark(resource_budget=ResourceBudget(wall_time_s=120))
        config = OptimizeConfig(
            output_dir=Path("runs/exp1"),
            inner_model="azure/gpt-5.4-mini",
            outer_model="bedrock/claude-opus-4-5",
        )
        PRISMOptimizer(benchmark, config, generations=10).run()
    """

    def __init__(self, benchmark: Benchmark, config: OptimizeConfig) -> None:
        self.benchmark = benchmark
        self.config = config

    @property
    def acceptance(self) -> AcceptanceCriterion:
        """Resolved acceptance criterion — falls back to HoldoutPassRate if not set."""
        if self.config.acceptance is not None:
            return self.config.acceptance
        from agent_harness_optimizer.framework.acceptance import HoldoutPassRate

        return HoldoutPassRate()

    @abstractmethod
    def run(self) -> None:
        """Run the full optimization loop. Blocking. Writes all output to config.output_dir."""
