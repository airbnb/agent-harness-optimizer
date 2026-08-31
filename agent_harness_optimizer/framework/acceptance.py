"""Acceptance criteria plug-ins for all AHO optimizers.

All criteria compare a *candidate* SplitScore against the *current best*
SplitScore and return (accepted: bool, reason: str).

Three built-ins are provided:

  HoldoutPassRate   — strict holdout pass count improvement (default)
                      accept iff candidate.holdout_passed > current.holdout_passed
                      No secondary metrics, no threshold, no hyperparameters.

  HoldoutPareto     — holdout pass_rate first, then reliability tie-break
                      accept iff candidate Pareto-dominates current on
                      (holdout_pass_rate, reliability).  Useful when you want
                      the optimizer to avoid accepting a prompt that improves
                      pass rate at the cost of reliability.

  CombinedPassRate  — simple average of train and holdout pass_rate
                      accept iff (candidate_train + candidate_holdout) / 2
                      > (current_train + current_holdout) / 2
                      Penalizes train regression, rewards generalization from both splits.

New criteria can be added by subclassing AcceptanceCriterion and implementing
__call__.  Pass the instance as `acceptance` in OptimizeConfig.

Usage::

    from agent_harness_optimizer.framework.acceptance import HoldoutPassRate, HoldoutPareto, CombinedPassRate

    config = OptimizeConfig(..., acceptance=CombinedPassRate())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent_harness_optimizer.framework.benchmark import SplitScore


class AcceptanceCriterion(ABC):
    """Base class.  Subclass and implement __call__."""

    @abstractmethod
    def __call__(
        self,
        candidate_train: SplitScore,
        candidate_holdout: SplitScore,
        current_train: SplitScore,
        current_holdout: SplitScore,
    ) -> tuple[bool, str]:
        """Return (accepted, reason_string)."""


@dataclass
class HoldoutPassRate(AcceptanceCriterion):
    """Default: accept iff candidate passes strictly more holdout cases.

    No threshold, no secondary metrics — one hyperparameter-free rule.
    At n=100 holdout this means +1 case = +1pp minimum improvement.
    """

    def __call__(
        self,
        candidate_train: SplitScore,
        candidate_holdout: SplitScore,
        current_train: SplitScore,
        current_holdout: SplitScore,
    ) -> tuple[bool, str]:
        c_pass = candidate_holdout.passed
        cur_pass = current_holdout.passed
        if c_pass > cur_pass:
            return True, f"holdout {cur_pass}→{c_pass} (+{c_pass - cur_pass})"
        if c_pass < cur_pass:
            return False, f"holdout {cur_pass}→{c_pass} ({c_pass - cur_pass})"
        return False, f"holdout tie at {c_pass}/{candidate_holdout.total}"


@dataclass
class HoldoutPareto(AcceptanceCriterion):
    """Accept iff candidate Pareto-dominates current on (holdout_pass_rate, reliability).

    Dominance: candidate must be >= on both axes and strictly better on at least one.
    Tie on both → reject (conservative, keeps current).

    holdout_pass_rate  = holdout.passed / holdout.total
    reliability        = train.reliability  (1 - stuck_rate from train split)
    """

    def __call__(
        self,
        candidate_train: SplitScore,
        candidate_holdout: SplitScore,
        current_train: SplitScore,
        current_holdout: SplitScore,
    ) -> tuple[bool, str]:
        c_ho = (
            candidate_holdout.passed / candidate_holdout.total if candidate_holdout.total else 0.0
        )
        cur_ho = current_holdout.passed / current_holdout.total if current_holdout.total else 0.0
        c_rel = candidate_train.reliability
        cur_rel = current_train.reliability

        better_ho = c_ho > cur_ho
        better_rel = c_rel > cur_rel
        worse_ho = c_ho < cur_ho
        worse_rel = c_rel < cur_rel

        if worse_ho or worse_rel:
            return False, (
                f"dominated: holdout {cur_ho:.3f}→{c_ho:.3f}  reliability {cur_rel:.3f}→{c_rel:.3f}"
            )
        if better_ho or better_rel:
            axes = []
            if better_ho:
                axes.append(
                    f"holdout {cur_ho:.3f}→{c_ho:.3f}"
                    f" (+{candidate_holdout.passed - current_holdout.passed})"
                )
            if better_rel:
                axes.append(f"reliability {cur_rel:.3f}→{c_rel:.3f}")
            return True, "Pareto improvement: " + ", ".join(axes)
        return False, f"Pareto tie: holdout={c_ho:.3f} reliability={c_rel:.3f}"


@dataclass
class CombinedPassRate(AcceptanceCriterion):
    """Accept iff candidate improves the simple average of train and holdout pass_rate.

    combined = (train.pass_rate + holdout.pass_rate) / 2

    Stricter than HoldoutPassRate alone: a holdout gain can be vetoed by a train
    regression of equal magnitude.  Equally-weighted — no hyperparameters.
    """

    def __call__(
        self,
        candidate_train: SplitScore,
        candidate_holdout: SplitScore,
        current_train: SplitScore,
        current_holdout: SplitScore,
    ) -> tuple[bool, str]:
        c_tr = candidate_train.pass_rate
        c_ho = candidate_holdout.pass_rate
        cur_tr = current_train.pass_rate
        cur_ho = current_holdout.pass_rate
        c_combined = (c_tr + c_ho) / 2
        cur_combined = (cur_tr + cur_ho) / 2
        delta = c_combined - cur_combined
        if c_combined > cur_combined:
            return True, (
                f"combined {cur_combined:.3f}→{c_combined:.3f} (+{delta:.3f})  "
                f"[train {cur_tr:.3f}→{c_tr:.3f}  holdout {cur_ho:.3f}→{c_ho:.3f}]"
            )
        if c_combined < cur_combined:
            return False, (
                f"combined {cur_combined:.3f}→{c_combined:.3f} ({delta:.3f})  "
                f"[train {cur_tr:.3f}→{c_tr:.3f}  holdout {cur_ho:.3f}→{c_ho:.3f}]"
            )
        return False, f"combined tie at {c_combined:.3f}"
