#!/usr/bin/env python3
"""RelLift95(B) — selection-replay estimator with CIs and subsampling analysis.

This implements the estimator as defined in the paper's revision (§3/§4.5):
RelLift95(B) is NOT the plain 5th percentile of per-run deltas. It replays,
many times, the decision a deployer would face under budget B:

    1. Draw N_B runs (with replacement) from the observed runs of one
       benchmark-optimizer arm (N_B = how many runs the budget affords).
    2. Let the pre-scorecard statistic G_i = (gate pass rate, 1 - Stuck)
       pick the best of those N_B runs — exactly as a deployer would,
       without seeing scorecard (test) data.
    3. Record the held-out scorecard lift of the selected run.
    4. Repeat `draws` times (default 5000). RelLift95 is the 5th percentile
       of the selected-lift distribution.

Because resampling cannot create information beyond the underlying runs, the
script also reports:
    - a confidence interval on the RelLift95 estimate itself, from an outer
      bootstrap over runs (resample the run set with replacement, recompute
      RelLift95, take percentile bounds across `ci_draws` replicates); and
    - a subsampling analysis (e.g. 16 -> 12 -> 8 runs) showing how the
      estimate moves as the number of underlying runs shrinks.

Inputs are the same report.json files consumed by scripts/compute_metrics.py.
The pre-scorecard statistic uses final_holdout.pass_rate (gate) and
final_train.reliability (1 - Stuck); the recorded outcome is the scorecard
delta in percentage points (delta_scorecard_pp).

Usage:
    python scripts/rellift.py --runs-dir runs/ --pattern "tau-telecom-prism-*" \
        --budget 4 --draws 5000 --ci-draws 1000 --subsample-sizes 16,12,8
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from compute_metrics import (  # noqa: E402
    extract_delta_pp,
    find_reports,
    is_valid_run,
)

# ---------------------------------------------------------------------------
# Per-run observation
# ---------------------------------------------------------------------------


def extract_gate_stat(report: dict) -> tuple[float, float] | None:
    """Pre-scorecard selection statistic G_i = (gate pass rate, 1 - Stuck).

    gate pass rate  — final_holdout.pass_rate (the gate split score of the
                      returned harness)
    1 - Stuck       — final_train.reliability (fraction of cases completing
                      without timeout/crash/loop)

    Both are observable before the scorecard is touched, so selecting on G_i
    replays exactly the information a deployer has at selection time.
    """
    fh = report.get("final_holdout", {}) or {}
    ft = report.get("final_train", {}) or {}
    gate_pr = fh.get("pass_rate")
    if gate_pr is None and fh.get("total"):
        gate_pr = fh.get("passed", 0) / fh["total"]
    reliability = ft.get("reliability")
    if gate_pr is None:
        return None
    return (float(gate_pr), float(reliability) if reliability is not None else 1.0)


def load_observations(reports: list[dict], min_reliability: float = 0.5) -> list[dict]:
    """One observation per valid run: {delta, gate_stat}."""
    obs = []
    for r in reports:
        if not is_valid_run(r, min_reliability):
            continue
        delta = extract_delta_pp(r)
        g = extract_gate_stat(r)
        if delta is None or g is None:
            continue
        obs.append({"delta": float(delta), "g": g})
    return obs


# ---------------------------------------------------------------------------
# Selection-replay estimator
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0,1]) on pre-sorted values."""
    if not sorted_vals:
        raise ValueError("empty sample")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def selection_replay(
    obs: list[dict],
    budget: int,
    draws: int,
    rng: random.Random,
    gamma: float = 0.05,
) -> dict[str, float]:
    """Replay budgeted selection `draws` times; return tail stats of selected lift.

    Each draw samples `budget` runs with replacement, selects the run with the
    lexicographically largest G_i = (gate pass rate, 1 - Stuck), and records
    its held-out delta.
    """
    if not obs:
        raise ValueError("no observations")
    selected: list[float] = []
    for _ in range(draws):
        pick = max(
            (obs[rng.randrange(len(obs))] for _ in range(budget)),
            key=lambda o: o["g"],
        )
        selected.append(pick["delta"])
    selected.sort()
    return {
        "rellift_95": _percentile(selected, gamma),
        "median_selected": _percentile(selected, 0.5),
        "mean_selected": statistics.fmean(selected),
    }


def rellift_with_ci(
    obs: list[dict],
    budget: int,
    draws: int,
    ci_draws: int,
    seed: int,
    gamma: float = 0.05,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Point estimate + outer-bootstrap CI on the RelLift95 estimate itself.

    The outer bootstrap resamples the *run set* with replacement (same size),
    recomputes RelLift95 on each replicate with a smaller inner draw count,
    and reports percentile bounds across replicates. This quantifies the
    sampling variance the estimator carries at the observed number of runs —
    resampling cannot create information beyond the underlying runs.
    """
    rng = random.Random(seed)
    point = selection_replay(obs, budget, draws, rng, gamma)

    inner_draws = max(500, draws // 5)
    reps: list[float] = []
    for _ in range(ci_draws):
        resampled = [obs[rng.randrange(len(obs))] for _ in range(len(obs))]
        reps.append(selection_replay(resampled, budget, inner_draws, rng, gamma)["rellift_95"])
    reps.sort()
    alpha = (1.0 - ci_level) / 2.0
    return {
        **point,
        "n_runs": len(obs),
        "budget": budget,
        "ci_low": _percentile(reps, alpha),
        "ci_high": _percentile(reps, 1.0 - alpha),
        "ci_level": ci_level,
    }


def subsampling_analysis(
    obs: list[dict],
    budget: int,
    sizes: list[int],
    seed: int,
    draws: int = 2000,
    reps_per_size: int = 200,
    gamma: float = 0.05,
) -> list[dict[str, Any]]:
    """Recompute RelLift95 on random subsamples of the run set (16 -> 12 -> 8).

    For each size, draws `reps_per_size` subsamples WITHOUT replacement,
    computes RelLift95 on each, and reports the mean and spread across
    subsamples. A stable estimator moves little as runs are removed.
    """
    rng = random.Random(seed)
    rows = []
    for size in sizes:
        if size > len(obs):
            rows.append({"size": size, "note": f"skipped (only {len(obs)} runs)"})
            continue
        if size == len(obs):
            vals = [selection_replay(obs, budget, draws, rng, gamma)["rellift_95"]]
        else:
            vals = []
            for _ in range(reps_per_size):
                sub = rng.sample(obs, size)
                vals.append(
                    selection_replay(sub, budget, max(500, draws // 4), rng, gamma)["rellift_95"]
                )
        vals.sort()
        rows.append(
            {
                "size": size,
                "mean": statistics.fmean(vals),
                "p5": _percentile(vals, 0.05),
                "p95": _percentile(vals, 0.95),
                "n_reps": len(vals),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern selecting one benchmark-optimizer arm's run dirs",
    )
    ap.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional JSON mapping {dir_name: label}; runs are grouped by label",
    )
    ap.add_argument(
        "--budget", type=int, default=4, help="N_B: runs the deployment budget affords (default 4)"
    )
    ap.add_argument("--draws", type=int, default=5000, help="Selection-replay draws (default 5000)")
    ap.add_argument(
        "--ci-draws",
        type=int,
        default=1000,
        help="Outer-bootstrap replicates for the CI (default 1000)",
    )
    ap.add_argument(
        "--subsample-sizes",
        default="16,12,8",
        help="Comma-separated run counts for the subsampling analysis",
    )
    ap.add_argument(
        "--gamma", type=float, default=0.05, help="Tail quantile (default 0.05 -> RelLift95)"
    )
    ap.add_argument("--min-reliability", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None, help="Write full results as JSON")
    args = ap.parse_args()

    pairs = find_reports(args.runs_dir, args.pattern)
    if not pairs:
        print("No reports found.")
        sys.exit(1)

    name_map: dict[str, str] = {}
    if args.mapping and args.mapping.exists():
        name_map = json.loads(args.mapping.read_text())

    groups: dict[str, list[dict]] = {}
    for run_dir, report in pairs:
        label = name_map.get(run_dir.name, "arm")
        groups.setdefault(label, []).append(report)

    sizes = [int(s) for s in args.subsample_sizes.split(",") if s.strip()]
    results: dict[str, Any] = {}
    for label, reports in sorted(groups.items()):
        obs = load_observations(reports, args.min_reliability)
        print(f"\n=== {label}: {len(obs)} valid runs (of {len(reports)}) ===")
        if len(obs) < 2:
            print("  too few runs — skipping")
            continue
        res = rellift_with_ci(obs, args.budget, args.draws, args.ci_draws, args.seed, args.gamma)
        print(
            f"  RelLift95(B={args.budget}) = {res['rellift_95']:+.1f}pp  "
            f"[{res['ci_low']:+.1f}, {res['ci_high']:+.1f}] "
            f"({int(res['ci_level'] * 100)}% bootstrap CI on the estimate)"
        )
        print(
            f"  selected-lift mean = {res['mean_selected']:+.1f}pp  "
            f"median = {res['median_selected']:+.1f}pp"
        )
        sub = subsampling_analysis(obs, args.budget, sizes, args.seed, gamma=args.gamma)
        for row in sub:
            if "note" in row:
                print(f"  subsample n={row['size']}: {row['note']}")
            else:
                print(
                    f"  subsample n={row['size']}: mean={row['mean']:+.1f}pp  "
                    f"spread [{row['p5']:+.1f}, {row['p95']:+.1f}] over {row['n_reps']} draws"
                )
        results[label] = {"point": res, "subsampling": sub, "deltas": [o["delta"] for o in obs]}

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
