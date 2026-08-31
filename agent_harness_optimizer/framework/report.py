"""Canonical report schema shared across all AHO optimizers.

Every optimizer calls build_report() and writes the result to report.json.
This guarantees identical top-level keys so downstream metrics scripts
can read any optimizer's report.json with the same code.

Canonical schema
────────────────
optimizer                            str   "bh" | "prism" | "miprov2" | "gepa"
benchmark                            str
inner_model                          str
outer_model                          str
split_seed                           int | null
acceptance_criterion                 str   class name of AcceptanceCriterion

baseline_train
  passed                             int
  total                              int
  pass_rate                          float
  reliability                        float   1 - stuck_rate
  prompt_tokens_per_case             float
  completion_tokens_per_case         float
baseline_holdout
  passed / total / pass_rate
baseline_combined_pass_rate          float

final_train
  passed / total / pass_rate / reliability
  prompt_tokens_per_case / completion_tokens_per_case
  stuck_breakdown                    dict
final_holdout
  passed / total / pass_rate
final_combined_pass_rate             float

delta_train                          float   final_train.pass_rate  - baseline_train.pass_rate
delta_holdout                        float   final_holdout.pass_rate - baseline_holdout.pass_rate
delta_combined                       float   final_combined_pass_rate - baseline_combined_pass_rate

final_scorecard (present only when split_seed is not None)
  passed / total / pass_rate

optimization_rollouts                int    case evals spent during search (excl. baseline + final bookend)
total_rollouts                       int    optimization_rollouts + baseline(200) + final(200)

optimizer_config                     dict   optimizer-specific hyperparameters
"""

from __future__ import annotations

import math

from agent_harness_optimizer.framework.benchmark import SplitScore


def _wilson_ci(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval."""
    if total == 0:
        return 0.0, 0.0
    p = passed / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, round(centre - half, 4)), min(1.0, round(centre + half, 4))


def _split_dict(s: SplitScore) -> dict:
    return {
        "passed": s.passed,
        "total": s.total,
        "pass_rate": round(s.pass_rate, 4),
        "reliability": round(s.reliability, 4),
        "prompt_tokens_per_case": round(s.prompt_tokens_per_case, 2),
        "completion_tokens_per_case": round(s.completion_tokens_per_case, 2),
        "stuck_breakdown": s.stuck_breakdown,
    }


def _holdout_dict(s: SplitScore) -> dict:
    return {
        "passed": s.passed,
        "total": s.total,
        "pass_rate": round(s.pass_rate, 4),
    }


def build_report(
    *,
    optimizer: str,
    benchmark_name: str,
    inner_model: str,
    outer_model: str,
    split_seed: int | None,
    acceptance_criterion: str,
    base_train: SplitScore,
    base_holdout: SplitScore,
    final_train: SplitScore,
    final_holdout: SplitScore,
    final_scorecard: SplitScore | None = None,
    baseline_scorecard: SplitScore | None = None,
    optimization_rollouts: int = 0,
    optimizer_config: dict | None = None,
    # EMNLP identity fields
    experiment_id: str | None = None,
    condition_id: str | None = None,
    repeat_id: int = 0,
    search_seed: int = 0,
    num_scorecard_trials: int = 1,
    # EMNLP outer token fields
    search_outer_calls: int = 0,
    search_outer_tokens_in: int = 0,
    search_outer_tokens_out: int = 0,
) -> dict:
    """Return canonical report dict. Write to report.json in output_dir."""
    base_combined_total = base_train.total + base_holdout.total
    final_combined_total = final_train.total + final_holdout.total
    base_combined_pass_rate = (
        (base_train.passed + base_holdout.passed) / base_combined_total
        if base_combined_total
        else 0.0
    )
    final_combined_pass_rate = (
        (final_train.passed + final_holdout.passed) / final_combined_total
        if final_combined_total
        else 0.0
    )

    report: dict = {
        "optimizer": optimizer,
        "benchmark": benchmark_name,
        "inner_model": inner_model,
        "outer_model": outer_model,
        "split_seed": split_seed,
        "acceptance_criterion": acceptance_criterion,
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "repeat_id": repeat_id,
        "search_seed": search_seed,
        "num_scorecard_trials": num_scorecard_trials,
        "baseline_train": _split_dict(base_train),
        "baseline_holdout": _holdout_dict(base_holdout),
        "baseline_combined_pass_rate": round(base_combined_pass_rate, 4),
        "final_train": _split_dict(final_train),
        "final_holdout": _holdout_dict(final_holdout),
        "final_combined_pass_rate": round(final_combined_pass_rate, 4),
        "delta_train": round(final_train.pass_rate - base_train.pass_rate, 4),
        "delta_holdout": round(final_holdout.pass_rate - base_holdout.pass_rate, 4),
        "delta_combined": round(final_combined_pass_rate - base_combined_pass_rate, 4),
        "optimization_rollouts": optimization_rollouts,
        "total_rollouts": optimization_rollouts
        + base_train.total
        + base_holdout.total
        + final_train.total
        + final_holdout.total,
    }

    if final_scorecard is not None:
        ci_lo, ci_hi = _wilson_ci(final_scorecard.passed, final_scorecard.total)
        sc_block: dict = {
            "passed": final_scorecard.passed,
            "total": final_scorecard.total,
            "pass_rate": round(final_scorecard.pass_rate, 4),
            "reliability": round(final_scorecard.reliability, 4),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
        if baseline_scorecard is not None:
            delta_sc = final_scorecard.pass_rate - baseline_scorecard.pass_rate
            sc_block["baseline_passed"] = baseline_scorecard.passed
            sc_block["baseline_total"] = baseline_scorecard.total
            sc_block["baseline_pass_rate"] = round(baseline_scorecard.pass_rate, 4)
            sc_block["baseline_reliability"] = round(baseline_scorecard.reliability, 4)
            sc_block["delta_scorecard"] = round(delta_sc, 4)
            sc_block["delta_scorecard_pp"] = round(delta_sc * 100, 2)
            sc_block["improving_run"] = delta_sc > 0.0
            sc_block["rep_success_1pp"] = delta_sc >= 0.01
            sc_block["rep_success_2pp"] = delta_sc >= 0.02
        report["final_scorecard"] = sc_block

    report["search_outer_calls"] = search_outer_calls
    report["search_outer_tokens_in"] = search_outer_tokens_in
    report["search_outer_tokens_out"] = search_outer_tokens_out

    if optimizer_config:
        report["optimizer_config"] = optimizer_config

    return report


def print_report(report: dict) -> None:
    """Print canonical summary line for any optimizer."""
    opt = report.get("optimizer", "?")
    bt = report["baseline_train"]
    bh = report["baseline_holdout"]
    ft = report["final_train"]
    fh = report["final_holdout"]
    sc = report.get("final_scorecard")

    print(f"\n=== {opt.upper()} Final Report ===")
    print(
        f"  baseline  train:   {bt['passed']}/{bt['total']}  ({bt['pass_rate']:.4f})  "
        f"reliability={bt['reliability']:.4f}"
    )
    print(f"  baseline  holdout: {bh['passed']}/{bh['total']}  ({bh['pass_rate']:.4f})")
    print(f"  baseline  combined: {report['baseline_combined_pass_rate']:.4f}")
    print(
        f"  final     train:   {ft['passed']}/{ft['total']}  ({ft['pass_rate']:.4f})  "
        f"reliability={ft['reliability']:.4f}  "
        f"Δ={report['delta_train']:+.4f}"
    )
    print(
        f"  final     holdout: {fh['passed']}/{fh['total']}  ({fh['pass_rate']:.4f})  "
        f"Δ={report['delta_holdout']:+.4f}"
    )
    print(
        f"  final     combined: {report['final_combined_pass_rate']:.4f}  "
        f"Δ={report['delta_combined']:+.4f}"
    )
    if sc:
        delta_str = f"  Δ={sc['delta_scorecard']:+.4f}" if "delta_scorecard" in sc else ""
        print(
            f"  scorecard:         {sc['passed']}/{sc['total']}  ({sc['pass_rate']:.4f}){delta_str}"
        )
    print(f"  rollouts: opt={report['optimization_rollouts']}  total={report['total_rollouts']}")
