"""Tests for the acceptance criteria (framework/acceptance.py).

These rules decide whether an optimizer keeps a candidate, so each branch
(accept / reject / tie) is exercised for all three criteria.
"""

from agent_harness_optimizer.framework.acceptance import (
    CombinedPassRate,
    HoldoutPareto,
    HoldoutPassRate,
)
from agent_harness_optimizer.framework.benchmark import SplitScore


def _s(passed, total=100, reliability=1.0):
    return SplitScore(passed=passed, total=total, reliability=reliability)


class TestHoldoutPassRate:
    def test_accepts_strict_improvement(self):
        ok, reason = HoldoutPassRate()(_s(50), _s(61), _s(50), _s(60))
        assert ok and "+1" in reason

    def test_rejects_regression(self):
        ok, _ = HoldoutPassRate()(_s(50), _s(59), _s(50), _s(60))
        assert not ok

    def test_rejects_tie(self):
        ok, reason = HoldoutPassRate()(_s(70), _s(60), _s(50), _s(60))
        assert not ok and "tie" in reason

    def test_train_is_ignored(self):
        # train collapse does not veto a holdout gain under this rule
        ok, _ = HoldoutPassRate()(_s(0), _s(61), _s(90), _s(60))
        assert ok


class TestHoldoutPareto:
    def test_accepts_dominating_candidate(self):
        ok, reason = HoldoutPareto()(
            _s(50, reliability=1.0), _s(65), _s(50, reliability=0.9), _s(60)
        )
        assert ok and "Pareto improvement" in reason

    def test_rejects_when_worse_on_one_axis(self):
        # better holdout but worse reliability -> not dominating
        ok, reason = HoldoutPareto()(
            _s(50, reliability=0.8), _s(65), _s(50, reliability=0.9), _s(60)
        )
        assert not ok and "dominated" in reason

    def test_accepts_single_axis_gain_with_tie_on_other(self):
        ok, _ = HoldoutPareto()(_s(50, reliability=0.9), _s(61), _s(50, reliability=0.9), _s(60))
        assert ok

    def test_rejects_full_tie(self):
        ok, reason = HoldoutPareto()(
            _s(50, reliability=0.9), _s(60), _s(50, reliability=0.9), _s(60)
        )
        assert not ok and "tie" in reason

    def test_empty_holdout_treated_as_zero(self):
        ok, _ = HoldoutPareto()(
            _s(0, total=0, reliability=1.0),
            _s(0, total=0),
            _s(0, total=0, reliability=1.0),
            _s(0, total=0),
        )
        assert not ok


class TestCombinedPassRate:
    def test_accepts_combined_improvement(self):
        ok, _ = CombinedPassRate()(_s(60), _s(60), _s(50), _s(60))
        assert ok

    def test_train_regression_vetoes_equal_holdout_gain(self):
        # +10 holdout cancelled by -10 train -> tie -> reject
        ok, reason = CombinedPassRate()(_s(40), _s(70), _s(50), _s(60))
        assert not ok and "tie" in reason

    def test_rejects_combined_regression(self):
        ok, _ = CombinedPassRate()(_s(40), _s(60), _s(50), _s(60))
        assert not ok
