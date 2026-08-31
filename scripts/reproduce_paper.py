#!/usr/bin/env python3
"""Reproduce paper experiments end-to-end.

Usage:
    python scripts/reproduce_paper.py \\
        --benchmark bfcl \\
        --optimizer prism \\
        --inner-model openai/gpt-4o-mini \\
        --outer-model anthropic/claude-opus-4-7 \\
        --output-dir runs/bfcl-prism-s0 \\
        --split-seed 0

    # Run all 4 seeds sequentially (full paper experiment):
    for seed in 0 1 2 3; do
        python scripts/reproduce_paper.py \\
            --benchmark bfcl --optimizer prism \\
            --inner-model openai/gpt-4o-mini \\
            --outer-model anthropic/claude-opus-4-7 \\
            --output-dir runs/bfcl-prism-s$seed \\
            --split-seed $seed
    done

This script runs the full pipeline:
  1. Score the default prompt baseline
  2. Run the optimizer
  3. Optionally compute metrics (--compute-metrics flag)

Output format matches final_report_runs/ used in paper analysis.
Results can be fed into scripts/compute_metrics.py to reproduce paper tables.

Model credential setup:
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  # See agent_harness_optimizer/utils/llm_public.py for full credential instructions
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce paper experiments end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Required
    parser.add_argument(
        "--benchmark", required=True, help="bfcl | tau-retail | tau-telecom | tau-airline"
    )
    parser.add_argument(
        "--optimizer", required=True, help="prism | better-harness | miprov2 | gepa"
    )
    parser.add_argument(
        "--inner-model",
        required=True,
        help="litellm model string for scored agent (e.g. openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--outer-model",
        required=True,
        help="litellm model string for proposer (e.g. anthropic/claude-opus-4-7)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Output directory for this run"
    )

    # CV split
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Train/holdout split seed (0-3 for BFCL 4-fold CV; default 0)",
    )

    # Optimizer settings
    parser.add_argument(
        "--generations", type=int, default=10, help="PRISM: number of generations (default 10)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10, help="BH: max iterations (default 10)"
    )
    parser.add_argument(
        "--mutations-per-gen",
        type=int,
        default=3,
        help="PRISM: mutations per generation (default 3)",
    )
    parser.add_argument(
        "--prism-prompt-only",
        action="store_true",
        default=False,
        help="PRISM-PO variant: never touch middleware (prompt-only ablation)",
    )
    parser.add_argument(
        "--prism-ablation",
        default="none",
        choices=["none", "no_route", "no_gate", "no_matrix", "no_constraint", "no_crossover"],
        help="PRISM §6.3 component ablation: reverts one Table 1 attribute (default: none)",
    )
    parser.add_argument(
        "--bh-prompt-only",
        action="store_true",
        default=False,
        help="BH-PO variant: never touch middleware (prompt-only ablation)",
    )
    parser.add_argument(
        "--miprov2-num-candidates",
        type=int,
        default=5,
        help="MIPROv2: instruction candidates to generate (default 5)",
    )
    parser.add_argument(
        "--miprov2-num-trials",
        type=int,
        default=10,
        help="MIPROv2: Bayesian optimization trials (default 10)",
    )
    parser.add_argument(
        "--miprov2-minibatch-size",
        type=int,
        default=25,
        help="MIPROv2: cases per trial evaluation (default 25)",
    )
    parser.add_argument(
        "--miprov2-middleware",
        action="store_true",
        default=False,
        help="MIPROv2-MW variant: pattern-guided middleware proposal at final acceptance",
    )
    parser.add_argument(
        "--gepa-max-metric-calls",
        type=int,
        default=200,
        help="GEPA: total evaluate() calls budget (default 200)",
    )
    parser.add_argument(
        "--gepa-reflection-minibatch-size",
        type=int,
        default=5,
        help="GEPA: cases per reflection minibatch (default 5)",
    )
    parser.add_argument(
        "--gepa-middleware",
        action="store_true",
        default=False,
        help="GEPA-MW variant: optimize a pattern-guided middleware component alongside the prompt",
    )
    parser.add_argument(
        "--prism-population-cap",
        type=int,
        default=None,
        help="PRISM frontier-retention ablation: Pareto population cap "
        "(binds under --acceptance holdout_pareto)",
    )
    parser.add_argument(
        "--train-cases",
        type=int,
        default=None,
        help="Number of training cases (default: 100 for BFCL, 74 for tau)",
    )
    parser.add_argument(
        "--holdout-cases",
        type=int,
        default=None,
        help="Number of holdout cases (default: 100 for BFCL, 40 for tau)",
    )

    # Resource budget
    parser.add_argument(
        "--wall-time-s",
        type=float,
        default=None,
        help="Per-case wall-clock timeout (default: 300s BFCL / 900s tau)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max tool calls per case (default: 100 BFCL / 200 tau)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per case (default: 500000 BFCL / 100000 tau)",
    )

    # Reproducibility
    parser.add_argument(
        "--repeat-id",
        type=int,
        default=0,
        help="Repeat index for multiple runs per seed (default 0)",
    )
    parser.add_argument(
        "--search-seed", type=int, default=0, help="Stochastic seed for proposer LLM (default 0)"
    )
    parser.add_argument(
        "--scorecard-trials", type=int, default=None, help="k for pass^k (default: 1 BFCL / 4 tau)"
    )

    # Pipeline control
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline scoring (reuse existing baseline if present)",
    )
    parser.add_argument(
        "--shared-baseline-dir",
        type=Path,
        default=None,
        help="Shared baseline dir — if multiple optimizers share the same baseline, "
        "point all to the same directory",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="After optimization, compute paper metrics and print summary",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume an interrupted optimization run"
    )

    # tau-bench
    parser.add_argument(
        "--tau-data-dir", default=None, help="Path to tau2-bench data dir (sets TAU2_DATA_DIR)"
    )

    return parser.parse_args()


def set_defaults_by_benchmark(args: argparse.Namespace) -> None:
    """Fill in benchmark-specific defaults."""
    is_tau = args.benchmark.startswith("tau-")
    if args.wall_time_s is None:
        args.wall_time_s = 900.0 if is_tau else 300.0
    if args.max_steps is None:
        args.max_steps = 200 if is_tau else 100
    if args.train_cases is None:
        args.train_cases = 74 if is_tau else 100
    if args.holdout_cases is None:
        args.holdout_cases = 40 if is_tau else 100
    if args.scorecard_trials is None:
        args.scorecard_trials = 4 if is_tau else 1
    if args.max_tokens is None:
        args.max_tokens = 100_000 if is_tau else 500_000


def score_baseline(args: argparse.Namespace, baseline_dir: Path) -> None:
    """Run agent_harness_optimizer.score_baseline to establish shared baseline."""
    train_file = baseline_dir / "baseline" / "train.json"
    holdout_file = baseline_dir / "baseline" / "holdout.json"
    if train_file.exists() and holdout_file.exists():
        print(f"[reproduce] Baseline already exists at {baseline_dir} — reusing")
        return

    print(f"[reproduce] Scoring baseline into {baseline_dir} ...")
    cmd = [
        sys.executable,
        "-m",
        "agent_harness_optimizer.score_baseline",
        "--benchmark",
        args.benchmark,
        "--inner-model",
        args.inner_model,
        "--split-seed",
        str(args.split_seed),
        "--train-cases",
        str(args.train_cases),
        "--holdout-cases",
        str(args.holdout_cases),
        "--output-dir",
        str(baseline_dir),
        "--wall-time-s",
        str(args.wall_time_s),
        "--max-steps",
        str(args.max_steps),
        "--max-tokens",
        str(args.max_tokens),
        "--score-scorecard",
        "--scorecard-trials",
        str(args.scorecard_trials),
    ]
    if args.tau_data_dir:
        cmd.extend(["--tau-data-dir", args.tau_data_dir])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[reproduce] ERROR: baseline scoring failed", file=sys.stderr)
        sys.exit(1)


def run_optimizer(args: argparse.Namespace, shared_baseline_dir: Path | None) -> None:
    """Launch the optimizer via agent_harness_optimizer.cli."""
    print(f"\n[reproduce] Running {args.optimizer} on {args.benchmark} ...")
    cmd = [
        sys.executable,
        "-m",
        "agent_harness_optimizer.cli",
        "--benchmark",
        args.benchmark,
        "--optimizer",
        args.optimizer,
        "--inner-model",
        args.inner_model,
        "--outer-model",
        args.outer_model,
        "--output-dir",
        str(args.output_dir),
        "--split-seed",
        str(args.split_seed),
        "--train-cases",
        str(args.train_cases),
        "--holdout-cases",
        str(args.holdout_cases),
        "--wall-time-s",
        str(args.wall_time_s),
        "--max-steps",
        str(args.max_steps),
        "--max-tokens",
        str(args.max_tokens),
        "--outer-max-turns",
        "300",
        "--repeat-id",
        str(args.repeat_id),
        "--search-seed",
        str(args.search_seed),
        "--scorecard-trials",
        str(args.scorecard_trials),
        "--acceptance",
        "holdout_pass_rate",
    ]
    if args.optimizer == "prism":
        cmd.extend(
            [
                "--generations",
                str(args.generations),
                "--mutations-per-gen",
                str(args.mutations_per_gen),
            ]
        )
        if getattr(args, "prism_prompt_only", False):
            cmd.append("--prism-prompt-only")
        if getattr(args, "prism_ablation", "none") != "none":
            # map ablation name to the real CLI flag (--prism-no-route, ...)
            cmd.append("--prism-" + args.prism_ablation.replace("_", "-"))
        if getattr(args, "prism_population_cap", None) is not None:
            cmd.extend(["--prism-population-cap", str(args.prism_population_cap)])
    elif args.optimizer == "better-harness":
        cmd.extend(["--max-iterations", str(args.max_iterations)])
        if getattr(args, "bh_prompt_only", False):
            cmd.append("--bh-prompt-only")
    elif args.optimizer == "miprov2":
        cmd.extend(
            [
                "--miprov2-num-candidates",
                str(args.miprov2_num_candidates),
                "--miprov2-num-trials",
                str(args.miprov2_num_trials),
                "--miprov2-minibatch-size",
                str(args.miprov2_minibatch_size),
            ]
        )
        if getattr(args, "miprov2_middleware", False):
            cmd.append("--miprov2-middleware")
    elif args.optimizer == "gepa":
        cmd.extend(
            [
                "--gepa-max-metric-calls",
                str(args.gepa_max_metric_calls),
                "--gepa-reflection-minibatch-size",
                str(args.gepa_reflection_minibatch_size),
            ]
        )
        if getattr(args, "gepa_middleware", False):
            cmd.append("--gepa-middleware")

    if shared_baseline_dir:
        cmd.extend(["--shared-baseline-dir", str(shared_baseline_dir)])
    if args.resume:
        cmd.append("--resume")
    if args.tau_data_dir:
        cmd.extend(["--tau-data-dir", args.tau_data_dir])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[reproduce] ERROR: optimizer run failed", file=sys.stderr)
        sys.exit(1)


def print_quick_summary(output_dir: Path) -> None:
    """Print a quick summary of the run's key metrics."""
    report_path = output_dir / "report.json"
    if not report_path.exists():
        print(f"[reproduce] No report.json found at {report_path}")
        return

    try:
        with open(report_path) as f:
            # Read carefully — large files have per_candidate_history
            content = f.read(200_000)
            idx = content.find('"per_candidate_history"')
            if idx > 0:
                content = content[:idx].rstrip().rstrip(",") + "\n}"
            report = json.loads(content)

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE — QUICK SUMMARY")
        print("=" * 60)
        bh = report.get("baseline_holdout", {})
        fh = report.get("final_holdout", {})
        sc = report.get("final_scorecard", {})
        print(
            f"Baseline holdout:  {bh.get('pass_rate', 'N/A'):.3f}"
            if isinstance(bh.get("pass_rate"), float)
            else f"Baseline holdout: {bh.get('pass_rate', 'N/A')}"
        )
        print(
            f"Final holdout:     {fh.get('pass_rate', 'N/A'):.3f}"
            if isinstance(fh.get("pass_rate"), float)
            else f"Final holdout: {fh.get('pass_rate', 'N/A')}"
        )
        if sc and sc.get("delta_scorecard_pp") is not None:
            print(f"Scorecard delta:   {sc['delta_scorecard_pp']:+.1f}pp")
        if sc and sc.get("pass_rate") is not None:
            print(f"Scorecard pass:    {sc['pass_rate']:.3f}")
        opt_rollouts = report.get("optimization_rollouts")
        if opt_rollouts:
            print(f"Optimization rollouts: {opt_rollouts}")
        print(f"\nFull report: {report_path}")
        diff_path = output_dir / "final_diff.md"
        if diff_path.exists():
            print(f"Changes applied: {diff_path}")
        print("=" * 60)
    except Exception as e:
        print(f"[reproduce] Could not parse report: {e}")


def compute_and_print_metrics(output_dir: Path) -> None:
    """Run scripts/compute_metrics.py on the output directory."""
    metrics_script = Path(__file__).parent / "compute_metrics.py"
    if not metrics_script.exists():
        print(f"[reproduce] compute_metrics.py not found at {metrics_script}")
        return
    result = subprocess.run(
        [sys.executable, str(metrics_script), "--runs-dir", str(output_dir.parent)]
    )
    if result.returncode != 0:
        print("[reproduce] WARNING: metric computation failed (non-fatal)")


def main() -> None:
    args = parse_args()
    set_defaults_by_benchmark(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine shared baseline directory
    if args.shared_baseline_dir:
        shared_baseline_dir = args.shared_baseline_dir
    else:
        # Default: put baseline alongside the run directory for easy sharing
        shared_baseline_dir = (
            args.output_dir.parent
            / f"shared-baseline-{args.benchmark.replace('-', '')}-s{args.split_seed}"
        )

    if not args.skip_baseline:
        score_baseline(args, shared_baseline_dir)

    run_optimizer(args, shared_baseline_dir)
    print_quick_summary(args.output_dir)

    if args.compute_metrics:
        compute_and_print_metrics(args.output_dir)


if __name__ == "__main__":
    main()
