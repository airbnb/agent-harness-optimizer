"""Tests for the seeded disjoint split generator (utils/splits.py).

These splits define the paper's repair/gate/scorecard protocol
(BFCL 100/100/600, tau 20/20/74), so the invariants tested here —
disjointness, determinism, stratification, 4-fold CV pairing — are
correctness requirements for every reported number.
"""

import pytest

from agent_harness_optimizer.utils import splits as splits_mod
from agent_harness_optimizer.utils.splits import make_split


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(splits_mod, "_SPLITS_ROOT", tmp_path / "_splits")


def _assert_disjoint(split):
    assert not set(split.train) & set(split.holdout)
    assert not set(split.train) & set(split.scorecard)
    assert not set(split.holdout) & set(split.scorecard)


def test_basic_split_sizes_and_disjointness():
    s = make_split("bench", total_cases=800, train_cases=100, holdout_cases=100, seed=0)
    assert len(s.train) == 100
    assert len(s.holdout) == 100
    assert len(s.scorecard) == 600
    _assert_disjoint(s)


def test_same_seed_is_deterministic():
    a = make_split("bench", 200, 50, 50, seed=7)
    b = make_split("bench", 200, 50, 50, seed=7)
    assert (a.train, a.holdout, a.scorecard) == (b.train, b.holdout, b.scorecard)


def test_different_seeds_differ():
    a = make_split("bench", 200, 50, 50, seed=0)
    b = make_split("bench", 200, 50, 50, seed=1)
    assert a.train != b.train


def test_cache_roundtrip(tmp_path):
    a = make_split("bench", 100, 20, 20, seed=3)
    b = make_split("bench", 100, 20, 20, seed=3)  # served from cache
    assert a.train == b.train
    assert a.scorecard == b.scorecard


def test_pool_constrained_split_stays_in_pool():
    pool = list(range(100, 214))  # 114-style curated pool
    s = make_split("tau", total_cases=500, train_cases=20, holdout_cases=20, seed=0, pool=pool)
    universe = set(pool)
    assert set(s.train) <= universe
    assert set(s.holdout) <= universe
    assert set(s.scorecard) <= universe
    assert len(s.train) == 20 and len(s.holdout) == 20 and len(s.scorecard) == 74
    _assert_disjoint(s)


def test_oversized_request_raises():
    with pytest.raises(ValueError):
        make_split("bench", total_cases=30, train_cases=20, holdout_cases=20, seed=0)


def _strata_pool():
    # 3 categories with tau-telecom-like imbalance: 36 + 29 + 49 = 114
    pool = list(range(114))
    strata = {}
    for i in pool:
        strata[i] = "a" if i < 36 else ("b" if i < 65 else "c")
    return pool, strata


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stratified_4fold_disjoint_and_proportional(seed):
    pool, strata = _strata_pool()
    s = make_split("tau", 114, 20, 20, seed=seed, pool=pool, strata=strata)
    assert len(s.train) + len(s.holdout) + len(s.scorecard) == 114
    _assert_disjoint(s)
    # every split contains all three strata (proportional representation)
    for part in (s.train, s.holdout, s.scorecard):
        assert {strata[i] for i in part} == {"a", "b", "c"}


def test_stratified_4fold_cv_train_holdout_rotation():
    """Across seeds 0-3, train folds are pairwise disjoint (true 4-fold CV)."""
    pool, strata = _strata_pool()
    trains = [
        set(make_split("tau", 114, 20, 20, seed=k, pool=pool, strata=strata).train)
        for k in range(4)
    ]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not trains[i] & trains[j], f"seeds {i},{j} share train cases"


def test_stratified_fallback_seed_outside_cv_range():
    pool, strata = _strata_pool()
    s = make_split("tau", 114, 20, 20, seed=42, pool=pool, strata=strata)
    assert len(s.train) == 20 and len(s.holdout) == 20
    _assert_disjoint(s)
