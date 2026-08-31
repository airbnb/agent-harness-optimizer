"""Tests for the RelLift95 selection-replay estimator (scripts/rellift.py)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE / "scripts"))

from rellift import (  # noqa: E402
    extract_gate_stat,
    load_observations,
    rellift_with_ci,
    selection_replay,
    subsampling_analysis,
)


def _report(delta_pp: float, gate_pr: float, reliability: float = 1.0) -> dict:
    return {
        "final_scorecard": {"delta_scorecard_pp": delta_pp, "pass_rate": 0.5},
        "final_holdout": {"pass_rate": gate_pr, "passed": int(gate_pr * 40), "total": 40},
        "final_train": {"reliability": reliability},
    }


def test_extract_gate_stat():
    g = extract_gate_stat(_report(5.0, 0.6, 0.9))
    assert g == (0.6, 0.9)


def test_low_reliability_runs_excluded():
    reports = [_report(5.0, 0.6, 0.9), _report(2.0, 0.5, 0.2)]
    obs = load_observations(reports, min_reliability=0.5)
    assert len(obs) == 1 and obs[0]["delta"] == 5.0


def test_selection_replay_prefers_high_gate_runs():
    # One run has a much better gate stat AND a good delta; with budget >= 4
    # nearly every draw contains it, so the selected-lift tail should sit near
    # its delta, far above the plain 5th percentile of raw deltas (-10).
    obs = load_observations(
        [
            _report(-10.0, 0.30),
            _report(1.0, 0.35),
            _report(2.0, 0.40),
            _report(12.0, 0.80),
        ]
    )
    res = selection_replay(obs, budget=6, draws=4000, rng=random.Random(0))
    assert res["rellift_95"] > 0.0
    assert res["mean_selected"] > 6.0


def test_selection_replay_budget_one_equals_raw_distribution():
    # With budget 1 there is no selection: the tail equals the raw delta tail.
    obs = load_observations([_report(d, 0.5) for d in (-5.0, 0.0, 5.0, 10.0)])
    res = selection_replay(obs, budget=1, draws=8000, rng=random.Random(1))
    assert res["rellift_95"] <= 0.0  # worst raw delta is -5


def test_ci_brackets_point_estimate():
    obs = load_observations(
        [
            _report(d, 0.4 + 0.01 * i)
            for i, d in enumerate((-3.0, 1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 11.0))
        ]
    )
    res = rellift_with_ci(obs, budget=4, draws=2000, ci_draws=200, seed=0)
    assert res["ci_low"] <= res["rellift_95"] <= res["ci_high"]
    assert res["n_runs"] == 8


def test_reliability_breaks_gate_ties():
    # Same gate rate everywhere; the reliable run must win selection.
    obs = load_observations(
        [
            _report(-8.0, 0.5, reliability=0.6),
            _report(9.0, 0.5, reliability=1.0),
        ]
    )
    res = selection_replay(obs, budget=4, draws=2000, rng=random.Random(2))
    assert res["mean_selected"] > 5.0


def test_subsampling_rows():
    obs = load_observations([_report(float(i), 0.4 + 0.02 * i) for i in range(8)])
    rows = subsampling_analysis(
        obs, budget=4, sizes=[8, 4, 16], seed=0, draws=800, reps_per_size=50
    )
    by_size = {r["size"]: r for r in rows}
    assert "mean" in by_size[8]
    assert "mean" in by_size[4]
    assert "note" in by_size[16]  # more than available -> skipped


def test_deterministic_given_seed():
    obs = load_observations([_report(float(i), 0.4 + 0.02 * i) for i in range(6)])
    a = rellift_with_ci(obs, budget=3, draws=1000, ci_draws=100, seed=7)
    b = rellift_with_ci(obs, budget=3, draws=1000, ci_draws=100, seed=7)
    assert a == b
