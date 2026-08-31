#!/usr/bin/env python3
"""Compute all paper metrics from a directory of report.json files.

Computes: MeanLift, WorstLift, StdLift, RepRate_0, RepRate_1pp, RepRate_2pp,
          RelLift_95 (95th-percentile Reliable Lift), and MeanRollouts.
Outputs a LaTeX-ready table for each group of runs found.

Usage:
    # Single run directory
    python scripts/compute_metrics.py --runs-dir runs/bfcl-prism-s0

    # Multiple seeds (standard paper setup: seeds 0-3)
    python scripts/compute_metrics.py --runs-dir runs/ --pattern "bfcl-prism-*"

    # Full final_report_runs/ directory with a mapping file
    python scripts/compute_metrics.py \\
        --runs-dir final_report_runs/ \\
        --mapping final_report_runs_mapping.json \\
        --latex

    # Group by optimizer for Table 1 comparison
    python scripts/compute_metrics.py \\
        --runs-dir runs/ \\
        --group-by optimizer

Definitions of paper metrics (paper Sec 3 / Sec 4.5 conventions):
    MeanLift    = mean(delta_scorecard_pp) over runs, raw (NOT floored)  [pp]
    WorstLift   = lowest per-condition mean lift (condition = split seed);
                  falls back to min per-run delta when condition labels
                  are absent                                             [pp]
    StdLift     = std of per-condition mean lifts (swing between batches);
                  same fallback                                          [pp]
    RepRate_0   = % of runs with delta >= 0 (any improvement)            [%]
    RepRate_1pp = % of runs with delta >= 1pp                            [%]
    RepRate_2pp = % of runs with delta >= 2pp                            [%]
    RelLift_95  = budgeted selection-replay estimator (paper Sec 4.5),
                  delegated to scripts/rellift.py (budget=4, draws=5000,
                  seed=42); falls back to the plain 5th percentile of
                  per-run deltas when gate statistics are unavailable    [pp]
    MeanRollouts = mean total inner-model eval rollouts per run

    MeanLift_clipped = mean(max(0, delta)) is additionally reported as a
    supplementary field ("Helpful Effect": expected gain if a regressed
    harness is never deployed). It is NOT the paper's MeanLift.
    For RelLift_95 confidence intervals and the subsampling analysis
    (paper App. C.6), run scripts/rellift.py directly.

Scorecard delta is defined as:
    delta_scorecard_pp = (final_scorecard.pass_rate - baseline_scorecard.pass_rate) * 100
    or equivalently: final_scorecard.delta_scorecard_pp (stored in report.json)

Exclusion criterion:
    Runs with final_train.reliability < 0.5 are excluded as infrastructure failures.
    These indicate systemic API/deployment errors, not optimizer behavior.

Reproducibility note:
    The --search-seed / repeat_id parameters in agent_harness_optimizer.cli are LABELS only.
    They do NOT set a random seed for LLM sampling. LLM outputs are stochastic
    and runs with the same configuration will produce different (but statistically
    similar) results. "Reproducibility" in this codebase means same protocol, not
    identical outputs.

    Token cost estimates for runs completed before 2026-05-14 use imputed token counts
    based on per-call reference rates (token tracking was added after that date).
    Rollout counts are exact for all runs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------


def load_report(report_path: Path) -> dict | None:
    """Load report.json, truncating per_candidate_history for large files."""
    if not report_path.exists():
        return None
    try:
        content = report_path.read_text(encoding="utf-8")
        # Truncate before per_candidate_history (can be many MB)
        idx = content.find('"per_candidate_history"')
        if idx > 0:
            content = content[:idx].rstrip().rstrip(",") + "\n}"
        return json.loads(content)
    except Exception as e:
        print(f"  WARNING: could not load {report_path}: {e}", file=sys.stderr)
        return None


def find_reports(runs_dir: Path, pattern: str = "*") -> list[tuple[Path, dict]]:
    """Find all report.json files matching pattern, return (dir, report) pairs."""
    results = []
    for subdir in sorted(runs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if not fnmatch.fnmatch(subdir.name, pattern):
            continue
        rp = subdir / "report.json"
        if not rp.exists():
            continue
        report = load_report(rp)
        if report is not None:
            results.append((subdir, report))
    return results


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def extract_delta_pp(report: dict) -> float | None:
    """Extract scorecard delta in percentage points."""
    sc = report.get("final_scorecard", {})
    if not sc:
        return None
    if "delta_scorecard_pp" in sc:
        return float(sc["delta_scorecard_pp"])
    if "delta_scorecard" in sc:
        return float(sc["delta_scorecard"]) * 100.0
    # Fall back to holdout delta (less out-of-sample, but available for all runs)
    delta_ho = report.get("delta_holdout")
    if delta_ho is not None:
        return float(delta_ho) * 100.0
    # Compute from pass rates
    bscore = report.get("baseline_scorecard", {}) or report.get("baseline_holdout", {})
    fscore = sc if sc.get("pass_rate") else report.get("final_holdout", {})
    b_pr = bscore.get("pass_rate")
    f_pr = fscore.get("pass_rate")
    if b_pr is not None and f_pr is not None:
        return (float(f_pr) - float(b_pr)) * 100.0
    return None


def extract_rollouts(report: dict) -> int | None:
    val = report.get("optimization_rollouts") or report.get("total_rollouts")
    return int(val) if val is not None else None


def extract_outer_calls(report: dict) -> int | None:
    val = report.get("search_outer_calls")
    return int(val) if val is not None else None


def extract_baseline_pass_rate(report: dict) -> float | None:
    sc = report.get("baseline_scorecard", {}) or {}
    if sc.get("pass_rate") is not None:
        return float(sc["pass_rate"])
    bh = report.get("baseline_holdout", {}) or {}
    if bh.get("pass_rate") is not None:
        return float(bh["pass_rate"])
    return None


def extract_final_pass_rate(report: dict) -> float | None:
    sc = report.get("final_scorecard", {}) or {}
    if sc.get("pass_rate") is not None:
        return float(sc["pass_rate"])
    fh = report.get("final_holdout", {}) or {}
    if fh.get("pass_rate") is not None:
        return float(fh["pass_rate"])
    return None


def extract_pass_k(report: dict, k: int) -> float | None:
    sc = report.get("final_scorecard", {}) or {}
    val = sc.get(f"pass_rate_k{k}")
    return float(val) if val is not None else None


def is_valid_run(report: dict, min_reliability: float = 0.5) -> bool:
    """Filter runs with very low reliability (infra failure / deployment error).

    Paper exclusion criterion: exclude runs where final_train.reliability < 0.5.
    Reliability = fraction of cases that completed without infrastructure errors
    (timeout/rate-limit/API crash).  A run with < 50% clean completions indicates
    a systemic infrastructure problem, not an optimizer outcome.

    Note: the default threshold is 0.5, matching the paper's exclusion criterion.
    Override via --min-reliability if needed.
    """
    ft = report.get("final_train", {}) or {}
    return float(ft.get("reliability", 1.0)) >= min_reliability


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    reports: list[dict],
    min_reliability: float = 0.3,
) -> dict[str, Any]:
    """Compute all paper metrics from a list of reports.

    Returns a dict with:
        n, deltas, mean_lift, worst_lift, std_lift,
        rep_rate_0, rep_rate_1pp, rep_rate_2pp,
        rel_lift_95, mean_rollouts, mean_outer_calls,
        mean_baseline_pr, mean_final_pr
    """
    valid = [r for r in reports if is_valid_run(r, min_reliability)]
    n_total = len(reports)
    n_valid = len(valid)

    deltas = [d for r in valid if (d := extract_delta_pp(r)) is not None]
    rollouts = [x for r in valid if (x := extract_rollouts(r)) is not None]
    outer_calls = [x for r in valid if (x := extract_outer_calls(r)) is not None]
    baselines = [x for r in valid if (x := extract_baseline_pass_rate(r)) is not None]
    finals = [x for r in valid if (x := extract_final_pass_rate(r)) is not None]

    if not deltas:
        return {
            "n": n_valid,
            "n_total": n_total,
            "deltas": [],
            "mean_lift": None,
            "worst_lift": None,
            "std_lift": None,
            "rep_rate_0": None,
            "rep_rate_1pp": None,
            "rep_rate_2pp": None,
            "rel_lift_95": None,
            "mean_rollouts": None,
            "mean_outer_calls": None,
            "mean_baseline_pr": None,
            "mean_final_pr": None,
        }

    n = len(deltas)
    # MeanLift (paper Sec 3): raw mean paired lift, negatives NOT floored.
    mean_lift = statistics.mean(deltas)
    # Supplementary "Helpful Effect" mean (clipped at 0); not the paper metric.
    mean_lift_clipped = statistics.mean([max(0.0, d) for d in deltas])
    # WorstLift / StdLift (paper Sec 3): computed over per-condition mean
    # lifts (condition = split seed batch). Fallback: per-run deltas when
    # condition labels are absent from the reports.
    cond_map: dict[Any, list[float]] = {}
    for r in valid:
        d = extract_delta_pp(r)
        if d is None:
            continue
        cond = r.get("condition_id")
        if cond is None and r.get("split_seed") is not None:
            cond = f"s{r['split_seed']}"
        cond_map.setdefault(cond, []).append(d)
    if None not in cond_map and len(cond_map) > 1:
        cond_means = [statistics.mean(v) for v in cond_map.values()]
        worst_lift = min(cond_means)
        std_lift = statistics.stdev(cond_means) if len(cond_means) > 1 else 0.0
    else:
        worst_lift = min(deltas)
        std_lift = statistics.stdev(deltas) if n > 1 else 0.0
    rep_rate_0 = 100.0 * sum(1 for d in deltas if d >= 0) / n
    rep_rate_1pp = 100.0 * sum(1 for d in deltas if d >= 1.0) / n
    rep_rate_2pp = 100.0 * sum(1 for d in deltas if d >= 2.0) / n
    # RelLift_95 (paper Sec 4.5): budgeted selection replay, delegated to
    # scripts/rellift.py. Fallback: plain 5th percentile of per-run deltas
    # when gate statistics are unavailable in the reports.
    rel_lift_95 = None
    try:
        import random as _random

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import rellift as _rellift

        _obs = _rellift.load_observations(valid, min_reliability)
        if _obs:
            rel_lift_95 = _rellift.selection_replay(
                _obs, budget=4, draws=5000, rng=_random.Random(42)
            )["rellift_95"]
    except Exception as _e:  # pragma: no cover - fallback path
        print(
            f"  WARNING: selection-replay RelLift_95 unavailable ({_e}); "
            "falling back to plain 5th percentile",
            file=sys.stderr,
        )
    if rel_lift_95 is None:
        sorted_deltas = sorted(deltas)
        p5_idx = max(0, math.ceil(0.05 * n) - 1)
        rel_lift_95 = sorted_deltas[p5_idx]

    return {
        "n": n_valid,
        "n_total": n_total,
        "deltas": deltas,
        "mean_lift": mean_lift,
        "mean_lift_clipped": mean_lift_clipped,
        "worst_lift": worst_lift,
        "std_lift": std_lift,
        "rep_rate_0": rep_rate_0,
        "rep_rate_1pp": rep_rate_1pp,
        "rep_rate_2pp": rep_rate_2pp,
        "rel_lift_95": rel_lift_95,
        "mean_rollouts": statistics.mean(rollouts) if rollouts else None,
        "mean_outer_calls": statistics.mean(outer_calls) if outer_calls else None,
        "mean_baseline_pr": statistics.mean(baselines) * 100 if baselines else None,
        "mean_final_pr": statistics.mean(finals) * 100 if finals else None,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def fmt(val: Any, decimals: int = 1, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return str(val)


def print_metrics_table(
    groups: dict[str, dict],
    title: str = "Results",
    latex: bool = False,
) -> None:
    """Print a formatted table of metrics."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    col_order = [
        ("Group", 22),
        ("N", 4),
        ("Baseline%", 10),
        ("Final%", 8),
        ("MeanLift", 9),
        ("WorstLift", 10),
        ("StdLift", 8),
        ("RepRate0%", 10),
        ("RepRate2%", 10),
        ("RelLift95", 10),
        ("Rollouts", 10),
    ]

    header = "  ".join(
        name.ljust(w) if i == 0 else name.rjust(w) for i, (name, w) in enumerate(col_order)
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for group_name, m in sorted(groups.items()):
        row_vals = [
            group_name,
            str(m.get("n", "?")),
            fmt(m.get("mean_baseline_pr")),
            fmt(m.get("mean_final_pr")),
            fmt(m.get("mean_lift"), suffix="pp"),
            fmt(m.get("worst_lift"), suffix="pp"),
            fmt(m.get("std_lift"), suffix="pp"),
            fmt(m.get("rep_rate_0"), suffix="%"),
            fmt(m.get("rep_rate_2pp"), suffix="%"),
            fmt(m.get("rel_lift_95"), suffix="pp"),
            fmt(m.get("mean_rollouts"), decimals=0),
        ]
        row = "  ".join(
            row_vals[0].ljust(col_order[0][1]) if i == 0 else row_vals[i].rjust(col_order[i][1])
            for i in range(len(col_order))
        )
        print(row)
        if m.get("deltas"):
            delta_str = ", ".join(f"{d:+.1f}" for d in sorted(m["deltas"]))
            print(f"    deltas: [{delta_str}]")

    if latex:
        print("\n--- LaTeX table ---")
        print(r"\begin{tabular}{lccccccc}")
        print(r"\toprule")
        print(
            r"Method & N & Baseline\% & MeanLift & WorstLift & StdLift & RepRate$_0$ & MeanRollouts \\"
        )
        print(r"\midrule")
        for group_name, m in sorted(groups.items()):
            cells = [
                group_name.replace("_", r"\_"),
                str(m.get("n", "?")),
                fmt(m.get("mean_baseline_pr")),
                fmt(m.get("mean_lift")),
                fmt(m.get("worst_lift")),
                fmt(m.get("std_lift")),
                fmt(m.get("rep_rate_0")),
                fmt(m.get("mean_rollouts"), decimals=0),
            ]
            print(" & ".join(cells) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paper metrics from report.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing run subdirectories with report.json files",
    )
    parser.add_argument(
        "--pattern", default="*", help="Glob pattern to filter run directories (default: *)"
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional JSON mapping {dir_name: label} for run labeling",
    )
    parser.add_argument(
        "--group-by",
        default="run",
        choices=["run", "optimizer", "benchmark", "all"],
        help="How to group runs for metric computation (default: run)",
    )
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX table")
    parser.add_argument(
        "--min-reliability",
        type=float,
        default=0.5,
        help="Minimum final_train reliability to include a run (default 0.5, "
        "paper exclusion criterion). Runs below this threshold are "
        "excluded as infrastructure failures (not optimizer failures).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-run details")
    args = parser.parse_args()

    if not args.runs_dir.is_dir():
        # Handle single run directory
        if (args.runs_dir / "report.json").exists():
            args.runs_dir = args.runs_dir.parent
            args.pattern = args.runs_dir.name

    print(f"Scanning {args.runs_dir} for report.json files (pattern={args.pattern!r}) ...")
    run_pairs = find_reports(args.runs_dir, args.pattern)
    print(f"Found {len(run_pairs)} reports")

    if not run_pairs:
        print("No reports found. Check --runs-dir and --pattern.")
        sys.exit(1)

    # Load optional name mapping
    name_map: dict[str, str] = {}
    if args.mapping and args.mapping.exists():
        name_map = json.loads(args.mapping.read_text())

    if args.verbose:
        print("\nRun details:")
        for run_dir, report in run_pairs:
            label = name_map.get(run_dir.name, run_dir.name)
            delta = extract_delta_pp(report)
            bl = extract_baseline_pass_rate(report)
            fn = extract_final_pass_rate(report)
            valid = is_valid_run(report, args.min_reliability)
            flag = "" if valid else " [LOW-RELIABILITY]"
            print(
                f"  {label:<40} delta={fmt(delta, suffix='pp'):>8}  "
                f"baseline={fmt(bl):>6}  final={fmt(fn):>6}{flag}"
            )

    # Group runs
    if args.group_by == "run":
        groups: dict[str, list[dict]] = {}
        for run_dir, report in run_pairs:
            label = name_map.get(run_dir.name, run_dir.name)
            groups.setdefault(label, []).append(report)
    elif args.group_by == "all":
        groups = {"all_runs": [r for _, r in run_pairs]}
    elif args.group_by == "optimizer":
        groups = {}
        for run_dir, report in run_pairs:
            dn = run_dir.name
            opt = "unknown"
            for key in ["prism", "bh", "miprov2", "gepa"]:
                if key in dn:
                    opt = key
                    break
            groups.setdefault(opt, []).append(report)
    elif args.group_by == "benchmark":
        groups = {}
        for run_dir, report in run_pairs:
            bench = report.get("benchmark_name") or "unknown"
            groups.setdefault(bench, []).append(report)
    else:
        groups = {"all": [r for _, r in run_pairs]}

    # Compute metrics per group
    metrics: dict[str, dict] = {}
    for group_name, reports in groups.items():
        metrics[group_name] = compute_metrics(reports, args.min_reliability)

    title = f"Paper Metrics ({args.runs_dir.name})"
    print_metrics_table(metrics, title=title, latex=args.latex)

    # Summary for single-run case
    if len(run_pairs) == 1 and args.group_by == "run":
        run_dir, report = run_pairs[0]
        print(f"\nSingle run: {run_dir.name}")
        m = list(metrics.values())[0]
        delta = m["deltas"][0] if m["deltas"] else None
        print(f"  delta_scorecard_pp: {fmt(delta, suffix='pp')}")
        print(f"  baseline pass rate: {fmt(m['mean_baseline_pr'])}%")
        print(f"  final pass rate:    {fmt(m['mean_final_pr'])}%")
        print(f"  optimization rollouts: {fmt(m['mean_rollouts'], decimals=0)}")


if __name__ == "__main__":
    main()
