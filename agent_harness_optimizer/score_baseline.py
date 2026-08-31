"""Score the default prompt on train+holdout and write shared baseline files.

Usage (called by run scripts before launching 4 parallel optimizers):
    python -m agent_harness_optimizer.score_baseline \\
        --benchmark bfcl \\
        --inner-model azure/gpt-5.4-mini \\
        --split-seed 0 \\
        --train-cases 100 --holdout-cases 100 \\
        --output-dir runs/cv-baseline-seed0-2026-05-09

Writes:
    <output-dir>/baseline/train.json    — SplitScore.to_dict()
    <output-dir>/baseline/holdout.json  — SplitScore.to_dict()

All four optimizers launched with --shared-baseline-dir <output-dir> will
load these files instead of re-scoring, giving them an identical baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent_harness_optimizer.score_baseline")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--inner-model", required=True)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-cases", type=int, default=100)
    parser.add_argument("--holdout-cases", type=int, default=100)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wall-time-s", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--split-pool", default="base", choices=["base", "full"])
    parser.add_argument("--tau-data-dir", default=None)
    parser.add_argument(
        "--score-scorecard",
        action="store_true",
        help="Also score the scorecard split and write baseline/scorecard.json",
    )
    parser.add_argument(
        "--scorecard-trials",
        type=int,
        default=1,
        help="k for pass^k: run scorecard k times; case passes only if all k pass",
    )
    args = parser.parse_args()

    from agent_harness_optimizer.framework.benchmark import ResourceBudget

    budget = ResourceBudget(
        wall_time_s=args.wall_time_s,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
    )

    if args.benchmark == "bfcl":
        from agent_harness_optimizer.benchmarks.bfcl import BFCLBenchmark

        benchmark = BFCLBenchmark(model=args.inner_model, budget=budget, split_seed=args.split_seed)
        case_split = None
    elif args.benchmark.startswith("tau-"):
        import os

        from agent_harness_optimizer.benchmarks.tau_bench import TauBenchmark

        if args.tau_data_dir:
            os.environ["TAU2_DATA_DIR"] = args.tau_data_dir
        else:
            os.environ.setdefault(
                "TAU2_DATA_DIR", str(Path.home() / "projects" / "tau2-bench" / "data")
            )
        domain = args.benchmark[4:]
        benchmark = TauBenchmark(domain=domain, model=args.inner_model, budget=budget)
        from tau2.run import get_tasks

        from agent_harness_optimizer.utils.splits import make_split

        all_tasks = get_tasks(domain, task_split_name=None)
        id_to_idx = {t.id: i for i, t in enumerate(all_tasks)}
        pool = None
        strata: dict | None = None
        if args.split_pool == "base":
            try:
                from agent_harness_optimizer.benchmarks.tau_bench import build_tau_strata

                base_tasks = get_tasks(domain, task_split_name="base")
                pool = sorted(id_to_idx[t.id] for t in base_tasks if t.id in id_to_idx)
                idx_to_task = {id_to_idx[t.id]: t for t in base_tasks if t.id in id_to_idx}
                raw_strata = build_tau_strata(domain, pool, idx_to_task)
                strata = raw_strata if raw_strata else None
            except Exception:
                pass
        case_split = (
            make_split(
                benchmark_name=benchmark.name,
                total_cases=len(all_tasks),
                train_cases=args.train_cases,
                holdout_cases=args.holdout_cases,
                seed=args.split_seed,
                pool=pool,
                strata=strata,
            )
            if args.split_seed is not None
            else None
        )
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark!r}")

    _train_idx = case_split.train if case_split else None
    _holdout_idx = case_split.holdout if case_split else None
    _scorecard_idx = (
        case_split.scorecard if case_split and hasattr(case_split, "scorecard") else None
    )

    out = args.output_dir
    train_file = out / "baseline" / "train.json"
    holdout_file = out / "baseline" / "holdout.json"
    scorecard_file = out / "baseline" / "scorecard.json"

    if train_file.exists() and holdout_file.exists():
        import json as _json

        t = _json.loads(train_file.read_text())
        h = _json.loads(holdout_file.read_text())
        print(
            f"[score_baseline] already exists — train={t['passed']}/{t['total']} "
            f"holdout={h['passed']}/{h['total']}"
        )
        if args.score_scorecard and not scorecard_file.exists():
            pass  # fall through to score scorecard below
        else:
            return

    (out / "baseline").mkdir(parents=True, exist_ok=True)
    prompt = benchmark.default_system_prompt

    need_train_holdout = not (train_file.exists() and holdout_file.exists())
    need_scorecard = args.score_scorecard and not scorecard_file.exists()

    async def _run():
        tasks = []
        if need_train_holdout:
            tasks.append(
                benchmark.score_async(
                    prompt,
                    "train",
                    out / "baseline" / "train",
                    max_cases=args.train_cases,
                    case_indices=_train_idx,
                )
            )
            tasks.append(
                benchmark.score_async(
                    prompt,
                    "holdout",
                    out / "baseline" / "holdout",
                    max_cases=args.holdout_cases,
                    case_indices=_holdout_idx,
                )
            )
        if need_scorecard:
            tasks.append(
                benchmark.score_async(
                    prompt,
                    "scorecard",
                    out / "baseline" / "scorecard",
                    case_indices=_scorecard_idx,
                    num_trials=args.scorecard_trials,
                )
            )
        results = await asyncio.gather(*tasks)
        idx = 0
        base_tr = base_ho = base_sc = None
        if need_train_holdout:
            base_tr, base_ho = results[idx], results[idx + 1]
            idx += 2
        if need_scorecard:
            base_sc = results[idx]
        return base_tr, base_ho, base_sc

    print(
        f"[score_baseline] scoring {args.benchmark} seed={args.split_seed} "
        f"train={args.train_cases} holdout={args.holdout_cases}"
        f"{' scorecard' if need_scorecard else ''}…"
    )
    base_train, base_holdout, base_scorecard = asyncio.run(_run())
    if base_train is not None:
        train_file.write_text(json.dumps(base_train.to_dict(), indent=2))
        holdout_file.write_text(json.dumps(base_holdout.to_dict(), indent=2))
        print(
            f"[score_baseline] done — train={base_train.passed}/{base_train.total} "
            f"holdout={base_holdout.passed}/{base_holdout.total}"
        )
        print(f"[score_baseline] wrote {train_file} and {holdout_file}")
    if base_scorecard is not None:
        scorecard_file.write_text(json.dumps(base_scorecard.to_dict(), indent=2))
        print(f"[score_baseline] scorecard={base_scorecard.passed}/{base_scorecard.total}")
        print(f"[score_baseline] wrote {scorecard_file}")


if __name__ == "__main__":
    main()
