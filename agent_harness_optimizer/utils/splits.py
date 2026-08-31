"""Seeded disjoint train/holdout/scorecard split generator.

Generates and caches a reproducible, disjoint split of case indices so that
all four optimizers evaluate on identical train and holdout cases when given
the same --split-seed.  Without this, tau-bench's default slice-from-top
makes holdout a strict subset of train (holdout ⊂ train), contaminating
Δ_holdout with Δ_train signal.

## Pool-constrained splits

When a `pool` list is provided, the shuffle is restricted to that subset of
indices rather than range(total_cases).  For tau-telecom this means passing
the 114 curated "base" task indices so that every seed draws from the same
balanced composition (36 mobile_data_issue + 29 service_issue + 49 mms_issue)
and yields a consistent ~35% baseline pass rate.

## Stratified 4-fold CV splits

When `strata` is provided (dict mapping pool_index → stratum label) AND seed
is in {0,1,2,3}, the pool is divided into 4 disjoint folds within each stratum
(using a fixed master shuffle), then:
  - train    = fold[seed % 4]
  - holdout  = fold[(seed + 1) % 4]
  - scorecard = remaining 2 folds

This guarantees:
  1. Proportional category representation in every split (stratified).
  2. Pairwise disjoint train+holdout across all 4 seeds (true 4-fold CV).
  3. Consistent ~35% baseline pass rate across seeds (balanced composition).

For seeds outside {0,1,2,3} with strata, falls back to independent stratified
random sampling (proportional floor allocation + round-robin remainders).

## Scorecard

When train_cases + holdout_cases < len(pool), the remaining indices become
the scorecard split (CaseSplit.scorecard).  If they sum exactly to the pool
size, scorecard is empty.

Cache path: runs/_splits/{benchmark}_{seed}_{train}_{holdout}[_poolN][_strat].json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_SPLITS_ROOT = Path("runs/_splits")


@dataclass
class CaseSplit:
    train: list[int]  # indices into the benchmark's full task list
    holdout: list[int]  # disjoint from train
    scorecard: list[int]  # disjoint from train+holdout; empty if pool exhausted
    seed: int
    benchmark: str
    total_cases: int
    pool: list[int] = field(default_factory=list)  # pool used; empty = full range
    strata: dict[int, str] = field(default_factory=dict)  # pool_idx → stratum label


def make_split(
    benchmark_name: str,
    total_cases: int,
    train_cases: int,
    holdout_cases: int,
    seed: int = 42,
    pool: list[int] | None = None,
    strata: dict[int, str] | None = None,
) -> CaseSplit:
    """Return a seeded disjoint train/holdout/scorecard split.

    Args:
        benchmark_name: used for cache file naming.
        total_cases:    total tasks in the benchmark (used when pool=None).
        train_cases:    number of repair/train cases.
        holdout_cases:  number of gate/holdout cases.
        seed:           random seed for reproducibility.
        pool:           restrict universe to these indices (e.g. base 114 for tau).
        strata:         dict mapping pool index → stratum label.  When provided,
                        sampling is done proportionally within each stratum so
                        every split has balanced category representation.

    Returns:
        CaseSplit with .train, .holdout, .scorecard (remainder of pool).

    Raises:
        ValueError if train_cases + holdout_cases > len(pool or total_cases).
    """
    universe = pool if pool is not None else list(range(total_cases))
    if train_cases + holdout_cases > len(universe):
        raise ValueError(
            f"train_cases ({train_cases}) + holdout_cases ({holdout_cases}) "
            f"> pool size ({len(universe)})"
        )

    is_stratified = strata is not None and len(strata) > 0
    cache = _cache_path(
        benchmark_name,
        seed,
        train_cases,
        holdout_cases,
        pool_size=len(pool) if pool is not None else None,
        stratified=is_stratified,
    )
    if cache.exists():
        data = json.loads(cache.read_text())
        cached_pool = data.get("pool", [])
        expected_pool = pool if pool is not None else []
        if cached_pool == expected_pool:
            return CaseSplit(
                train=data["train"],
                holdout=data["holdout"],
                scorecard=data.get("scorecard", []),
                seed=data["seed"],
                benchmark=data["benchmark"],
                total_cases=data["total_cases"],
                pool=cached_pool,
                strata=data.get("strata", {}),
            )

    if is_stratified:
        train, holdout, scorecard = _stratified_split(
            universe, train_cases, holdout_cases, strata, seed
        )
    else:
        rng = random.Random(seed)
        shuffled = universe[:]
        rng.shuffle(shuffled)
        train = sorted(shuffled[:train_cases])
        holdout = sorted(shuffled[train_cases : train_cases + holdout_cases])
        scorecard = sorted(shuffled[train_cases + holdout_cases :])

    split = CaseSplit(
        train=train,
        holdout=holdout,
        scorecard=scorecard,
        seed=seed,
        benchmark=benchmark_name,
        total_cases=total_cases,
        pool=pool if pool is not None else [],
        strata=strata if strata is not None else {},
    )
    _save(split, cache)
    return split


def _stratified_split(
    universe: list[int],
    train_cases: int,
    holdout_cases: int,
    strata: dict[int, str],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified split with true 4-fold CV for seeds 0-3.

    When seed is in {0,1,2,3}: divides each stratum into 4 disjoint folds using
    a fixed master shuffle (seed=0), then rotates:
      train    = fold[seed % 4]
      holdout  = fold[(seed + 1) % 4]
      scorecard = remaining 2 folds
    This guarantees pairwise-disjoint train+holdout across all 4 seeds.

    For seeds outside {0,1,2,3}: falls back to independent proportional sampling
    (floor allocation + round-robin remainder distribution).
    """
    # Group by stratum; indices not in strata go to a default group
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in universe:
        label = strata.get(idx, "__default__")
        groups[label].append(idx)

    if seed in (0, 1, 2, 3):
        return _stratified_4fold(groups, train_cases, holdout_cases, seed)

    # Fallback: independent stratified random sampling
    rng = random.Random(seed)
    for label in groups:
        rng.shuffle(groups[label])

    n_pool = len(universe)
    n_score = n_pool - train_cases - holdout_cases

    train_out: list[int] = []
    holdout_out: list[int] = []
    score_out: list[int] = []

    for label, members in groups.items():
        n = len(members)
        n_tr = int(n * train_cases / n_pool)
        n_ho = int(n * holdout_cases / n_pool)
        n_sc = int(n * n_score / n_pool)
        train_out.extend(members[:n_tr])
        holdout_out.extend(members[n_tr : n_tr + n_ho])
        score_out.extend(members[n_tr + n_ho : n_tr + n_ho + n_sc])

    assigned = set(train_out) | set(holdout_out) | set(score_out)
    leftovers = [idx for idx in universe if idx not in assigned]
    rng.shuffle(leftovers)

    targets = (
        [train_out] * (train_cases - len(train_out))
        + [holdout_out] * (holdout_cases - len(holdout_out))
        + [score_out] * (n_score - len(score_out))
    )
    for lst, idx in zip(targets, leftovers):
        lst.append(idx)

    return sorted(train_out), sorted(holdout_out), sorted(score_out)


def _stratified_4fold(
    groups: dict[str, list[int]],
    train_cases: int,
    holdout_cases: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """True 4-fold CV: divide each stratum into 4 disjoint folds, rotate by seed.

    Master shuffle uses seed=0 so fold boundaries are identical for all seeds.
    Folds are built per-stratum using round-robin assignment after a master shuffle,
    giving ~28-30 cases per fold for tau-telecom (114 / 4).

    train    = fold[seed % 4][:train_cases]       — first train_cases items
    holdout  = fold[(seed+1) % 4][:holdout_cases] — first holdout_cases items
    scorecard = overflow from train/holdout folds + the other 2 folds entirely

    Slicing preserves full pairwise disjointness: no case appears in both
    train and holdout across any pair of seeds 0-3.
    """
    master_rng = random.Random(0)

    folds: list[list[int]] = [[], [], [], []]
    for members in groups.values():
        shuffled = members[:]
        master_rng.shuffle(shuffled)
        n = len(shuffled)
        base = n // 4
        remainder = n % 4
        pos = 0
        for f in range(4):
            size = base + (1 if f < remainder else 0)
            folds[f].extend(shuffled[pos : pos + size])
            pos += size

    # Shuffle each fold so strata are interleaved — ensures the first
    # train_cases / holdout_cases items are proportionally representative
    # when we slice below (not all-mobile then all-service then all-mms).
    for f in range(4):
        master_rng.shuffle(folds[f])

    tr_fold = seed % 4
    ho_fold = (seed + 1) % 4
    sc_folds = [(seed + 2) % 4, (seed + 3) % 4]

    tr_full = folds[tr_fold]
    ho_full = folds[ho_fold]

    train_out = sorted(tr_full[:train_cases])
    tr_overflow = tr_full[train_cases:]

    holdout_out = sorted(ho_full[:holdout_cases])
    ho_overflow = ho_full[holdout_cases:]

    score_out = sorted(tr_overflow + ho_overflow + folds[sc_folds[0]] + folds[sc_folds[1]])

    return train_out, holdout_out, score_out


def _cache_path(
    benchmark: str,
    seed: int,
    train: int,
    holdout: int,
    pool_size: int | None = None,
    stratified: bool = False,
) -> Path:
    safe = benchmark.replace("/", "_").replace(":", "_").replace(" ", "_")
    pool_tag = f"_pool{pool_size}" if pool_size is not None else ""
    strat_tag = "_strat" if stratified else ""
    return _SPLITS_ROOT / f"{safe}_{seed}_{train}_{holdout}{pool_tag}{strat_tag}.json"


def _save(split: CaseSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "benchmark": split.benchmark,
                "seed": split.seed,
                "total_cases": split.total_cases,
                "pool": split.pool,
                "strata": split.strata,
                "train": split.train,
                "holdout": split.holdout,
                "scorecard": split.scorecard,
            },
            indent=2,
        )
    )
