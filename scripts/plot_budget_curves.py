#!/usr/bin/env python3
"""Regenerate Figure 2: budgeted reliable-lift curves, one panel per benchmark.

Each curve is the empirical lower-tail (gamma-quantile, default 5th percentile)
held-out lift of the harness selected under a total search budget B, using
pre-scorecard information only — the same selection replay as RelLift95(B)
(scripts/rellift.py): at each budget point, N_B = max(1, floor(B / cost_per_run))
runs are drawn with replacement, the best is picked by G = (gate pass rate,
1 - Stuck), and the selected run's scorecard lift is recorded; the curve value
is the gamma-quantile over `draws` replays. Curve heights at B_ref equal the
RelLift95 column of Table 2.

Every curve is computed from real run directories (report.json files). Arms
with no runs on disk are omitted and reported — this script never invents data.

Arm spec file (JSON): maps panel -> arm label -> {pattern, cost_per_run?}:

{
  "BFCL": {
    "PRISM":    {"pattern": "emnlp-bfcl-prism-*-opus-*", "cost_per_run": 103},
    "PRISM-PO": {"pattern": "emnlp-bfcl-prism_prompt_only-*", "cost_per_run": 79},
    "GEPA-MW":  {"pattern": "bfcl-gepa-mw-*", "cost_per_run": 27}
  },
  "τ-Retail":  {...},
  "τ-Telecom": {...}
}

cost_per_run (dollars) is required for the budget axis; if omitted the script
tries the mean of `estimated_cost_usd` in the reports and fails the arm if
neither is available.

Usage:
    python scripts/plot_budget_curves.py --runs-dir runs/ --arms figure2_arms.json \
        --b-ref 400 --out figure2.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from compute_metrics import find_reports  # noqa: E402
from rellift import _percentile, load_observations  # noqa: E402

# Figure 2 styling: color = optimizer family, line style = surface.
_FAMILY_COLORS = {
    "PRISM": "tab:red",
    "BH": "tab:blue",
    "MIPROV2": "tab:green",
    "GEPA": "tab:purple",
}


def _style(label: str) -> tuple[str, str]:
    fam = label.upper().replace("-PO", "").replace("-MW", "").strip()
    color = _FAMILY_COLORS.get(fam, "tab:gray")
    dashed = label.upper().endswith("-PO")
    return color, ("--" if dashed else "-")


def curve_for_arm(
    obs: list[dict],
    cost_per_run: float,
    budgets: list[float],
    draws: int,
    rng: random.Random,
    gamma: float = 0.05,
) -> list[float]:
    """Lower-tail selected lift at each budget point (selection replay)."""
    vals = []
    for b in budgets:
        n_b = max(1, math.floor(b / cost_per_run)) if cost_per_run > 0 else 1
        selected = []
        for _ in range(draws):
            pick = max(
                (obs[rng.randrange(len(obs))] for _ in range(n_b)),
                key=lambda o: o["g"],
            )
            selected.append(pick["delta"])
        selected.sort()
        vals.append(_percentile(selected, gamma))
    return vals


def extract_cost(reports: list[dict]) -> float | None:
    costs = [
        float(r["estimated_cost_usd"])
        for r in reports
        if isinstance(r.get("estimated_cost_usd"), (int, float))
    ]
    return statistics.fmean(costs) if costs else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument(
        "--arms",
        type=Path,
        required=True,
        help="JSON spec: panel -> arm label -> {pattern, cost_per_run?}",
    )
    ap.add_argument("--b-max", type=float, default=500.0)
    ap.add_argument("--b-step", type=float, default=10.0)
    ap.add_argument(
        "--b-ref",
        type=float,
        default=None,
        help="Reference budget marked with a vertical dashed line",
    )
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--min-reliability", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("figure2.pdf"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = json.loads(args.arms.read_text())
    budgets = [
        b for b in (args.b_step * i for i in range(int(args.b_max / args.b_step) + 1)) if b > 0
    ]
    rng = random.Random(args.seed)

    fig, axes = plt.subplots(1, len(spec), figsize=(4.2 * len(spec), 3.2), sharex=True)
    if len(spec) == 1:
        axes = [axes]

    missing: list[str] = []
    for ax, (panel, arms) in zip(axes, spec.items()):
        for label, cfg in arms.items():
            pairs = find_reports(args.runs_dir, cfg["pattern"])
            reports = [r for _, r in pairs]
            obs = load_observations(reports, args.min_reliability)
            if len(obs) < 2:
                missing.append(
                    f"{panel}/{label} (pattern={cfg['pattern']!r}: {len(obs)} valid runs)"
                )
                continue
            cost = cfg.get("cost_per_run") or extract_cost(reports)
            if not cost:
                missing.append(
                    f"{panel}/{label}: no cost_per_run and no estimated_cost_usd in reports"
                )
                continue
            color, ls = _style(label)
            ax.plot(
                budgets,
                curve_for_arm(obs, cost, budgets, args.draws, rng, args.gamma),
                color=color,
                linestyle=ls,
                linewidth=1.4,
                label=label,
            )
        if args.b_ref:
            ax.axvline(args.b_ref, color="gray", linestyle=":", linewidth=1)
        ax.set_title(panel)
        ax.set_xlabel("Total search budget ($)")
        ax.legend(fontsize=6, loc="best")
    axes[0].set_ylabel(f"RelLift$_{{{int((1 - args.gamma) * 100)}}}$ (pp)")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    if missing:
        print(
            "\nARMS OMITTED (no usable run data — the figure must not show them "
            "until real runs exist):"
        )
        for m in missing:
            print(f"  - {m}")
        sys.exit(2)


if __name__ == "__main__":
    main()
