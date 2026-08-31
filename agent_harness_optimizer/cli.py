"""AHO CLI — benchmark-agnostic agent harness optimizer.

Usage:
    python -m agent_harness_optimizer.cli --benchmark bfcl --optimizer prism \\
        --inner-model openai/gpt-4o-mini \\
        --outer-model anthropic/claude-opus-4-7 \\
        --output-dir runs/bfcl-prism-001 \\
        --generations 10

    python -m agent_harness_optimizer.cli --benchmark tau-retail --optimizer better-harness \\
        --inner-model openai/gpt-4o-mini \\
        --outer-model anthropic/claude-opus-4-7 \\
        --output-dir runs/tau-bh-001 \\
        --max-iterations 10

Benchmark names:
    bfcl                     BFCLBenchmark (BFCL v4 multi-turn tool-calling)
    tau-<domain>             TauBenchmark, domain = airline|retail|telecom|banking_knowledge|mock

Optimizer names:
    prism                    PRISMOptimizer (evolutionary Pareto)
    better-harness           BetterHarnessOptimizer (linear iterate→propose→accept)
    miprov2                  MIPROv2Optimizer (DSPy Bayesian optimization, requires dspy-ai)
    gepa                     GEPAOptimizer (official GEPA reflective evolution, requires gepa)

Model strings use litellm format:
    openai/gpt-4o-mini, anthropic/claude-opus-4-7,
    bedrock/global.anthropic.claude-sonnet-4-6, azure/gpt-4o
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_harness_optimizer.framework.benchmark import ResourceBudget
from agent_harness_optimizer.framework.optimizer import OptimizeConfig


def _build_benchmark(args: argparse.Namespace):
    budget = ResourceBudget(
        wall_time_s=args.wall_time_s,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
    )

    if args.benchmark == "bfcl":
        from agent_harness_optimizer.benchmarks.bfcl import BFCLBenchmark

        return BFCLBenchmark(model=args.inner_model, budget=budget, split_seed=args.split_seed)

    if args.benchmark.startswith("tau-"):
        domain = args.benchmark[4:]
        from agent_harness_optimizer.benchmarks.tau_bench import TauBenchmark

        return TauBenchmark(
            domain=domain,
            model=args.inner_model,
            budget=budget,
            data_dir=args.tau_data_dir,
        )

    raise ValueError(f"Unknown benchmark: {args.benchmark!r}")


def _build_tau_case_split(args: argparse.Namespace, benchmark):
    """Build a seeded train/holdout/scorecard CaseSplit for tau benchmarks.

    Uses the curated base pool (114 tasks) by default so any seed yields a
    balanced task composition.  Stratified sampling ensures proportional
    category representation across all three splits (train, holdout, scorecard).
    """
    if args.split_seed is None:
        return None

    import os
    from pathlib import Path as _Path

    os.environ.setdefault("TAU2_DATA_DIR", str(_Path.home() / "projects" / "tau2-bench" / "data"))

    try:
        from tau2.run import get_tasks

        from agent_harness_optimizer.utils.splits import make_split

        domain = args.benchmark[4:]
        all_tasks = get_tasks(domain, task_split_name=None)
        total = len(all_tasks)
        id_to_idx = {t.id: i for i, t in enumerate(all_tasks)}

        pool = None
        strata: dict[int, str] | None = None
        pool_label = "full"

        if getattr(args, "split_pool", "base") == "base":
            try:
                from agent_harness_optimizer.benchmarks.tau_bench import build_tau_strata

                base_tasks = get_tasks(domain, task_split_name="base")
                pool = sorted(id_to_idx[t.id] for t in base_tasks if t.id in id_to_idx)
                pool_label = f"base({len(pool)})"
                idx_to_task = {id_to_idx[t.id]: t for t in base_tasks if t.id in id_to_idx}
                raw_strata = build_tau_strata(domain, pool, idx_to_task)
                strata = raw_strata if raw_strata else None
            except Exception:
                pass

        train_n = args.train_cases or total
        holdout_n = args.holdout_cases or 0
        case_split = make_split(
            benchmark_name=benchmark.name,
            total_cases=total,
            train_cases=train_n,
            holdout_cases=holdout_n,
            seed=args.split_seed,
            pool=pool,
            strata=strata,
        )

        sc_n = len(case_split.scorecard)
        strat_label = "stratified" if strata else "random"
        print(
            f"[cli] split_seed={args.split_seed} pool={pool_label} {strat_label}: "
            f"train={len(case_split.train)} holdout={len(case_split.holdout)} "
            f"scorecard={sc_n}"
        )
        return case_split
    except Exception as e:
        print(f"[cli] WARNING: could not build tau split ({e}); falling back to tasks[:N] slice")
        return None


def _ensure_shared_baseline(args: argparse.Namespace, benchmark, case_split) -> Path | None:
    """Score the default prompt once and cache results in shared_baseline_dir.

    Returns the directory path, or None if --shared-baseline-dir was not set.
    All four optimizers then load from this directory instead of re-scoring,
    giving them an identical baseline (eliminates stochastic noise in Δ comparisons).
    """
    if not getattr(args, "shared_baseline_dir", None):
        return None

    import asyncio
    import json

    bd = args.shared_baseline_dir
    train_file = bd / "baseline" / "train.json"
    holdout_file = bd / "baseline" / "holdout.json"

    if train_file.exists() and holdout_file.exists():
        print(f"[cli] shared baseline already exists at {bd} — reusing")
        return bd

    print(f"[cli] scoring shared baseline into {bd} …")
    bd.mkdir(parents=True, exist_ok=True)
    (bd / "baseline").mkdir(exist_ok=True)

    _cs = case_split
    _train_idx = _cs.train if _cs else None
    _holdout_idx = _cs.holdout if _cs else None

    train_cases = args.train_cases
    holdout_cases = args.holdout_cases

    async def _score_both():
        base_train, base_holdout = await asyncio.gather(
            benchmark.score_async(
                benchmark.default_system_prompt,
                "train",
                bd / "baseline" / "train",
                max_cases=train_cases,
                case_indices=_train_idx,
            ),
            benchmark.score_async(
                benchmark.default_system_prompt,
                "holdout",
                bd / "baseline" / "holdout",
                max_cases=holdout_cases,
                case_indices=_holdout_idx,
            ),
        )
        return base_train, base_holdout

    base_train, base_holdout = asyncio.run(_score_both())
    train_file.write_text(json.dumps(base_train.to_dict(), indent=2))
    holdout_file.write_text(json.dumps(base_holdout.to_dict(), indent=2))
    print(
        f"[cli] shared baseline: train={base_train.passed}/{base_train.total}  "
        f"holdout={base_holdout.passed}/{base_holdout.total}"
    )
    return bd


def _build_optimizer(args: argparse.Namespace, benchmark, config: OptimizeConfig):
    if args.optimizer == "prism":
        from agent_harness_optimizer.optimizers.prism.loop import PRISMConfig, PRISMOptimizer

        gc = PRISMConfig(
            train_cases=args.train_cases,
            holdout_cases=args.holdout_cases,
            mutations_per_gen=args.mutations_per_gen,
            seed=args.seed,
            pass_rate_metric=args.prism_pass_rate_metric,
            prompt_only=args.prism_prompt_only,
            no_route=args.prism_no_route,
            no_gate=args.prism_no_gate,
            no_matrix=args.prism_no_matrix,
            no_constraint=args.prism_no_constraint,
            no_crossover=args.prism_no_crossover,
            population_cap=args.prism_population_cap,
        )
        return PRISMOptimizer(benchmark, config, generations=args.generations, prism_config=gc)

    if args.optimizer == "better-harness":
        from agent_harness_optimizer.optimizers.better_harness.loop import (
            BetterHarnessOptimizer,
            BHConfig,
        )

        bh = BHConfig(
            max_iterations=args.max_iterations,
            train_cases=args.train_cases,
            holdout_cases=args.holdout_cases,
            seed=args.seed,
            prompt_only=args.bh_prompt_only,
        )
        return BetterHarnessOptimizer(benchmark, config, bh_config=bh)

    if args.optimizer == "miprov2":
        from agent_harness_optimizer.optimizers.miprov2.optimizer import (
            MIPROv2Config,
            MIPROv2Optimizer,
        )

        mc = MIPROv2Config(
            num_candidates=args.miprov2_num_candidates,
            num_trials=args.miprov2_num_trials,
            train_cases=args.train_cases,
            holdout_cases=args.holdout_cases,
            minibatch_size=args.miprov2_minibatch_size,
            minibatch_full_eval_steps=args.miprov2_full_eval_steps,
            seed=args.seed,
            middleware=args.miprov2_middleware,
        )
        return MIPROv2Optimizer(benchmark, config, mipro_config=mc)

    if args.optimizer == "gepa":
        from agent_harness_optimizer.optimizers.gepa.optimizer import GEPAConfig, GEPAOptimizer

        gc = GEPAConfig(
            max_metric_calls=args.gepa_max_metric_calls,
            reflection_minibatch_size=args.gepa_reflection_minibatch_size,
            train_cases=args.train_cases,
            holdout_cases=args.holdout_cases,
            seed=args.seed,
            middleware=args.gepa_middleware,
        )
        return GEPAOptimizer(benchmark, config, gepa_config=gc)

    raise ValueError(f"Unknown optimizer: {args.optimizer!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agent-harness-optimizer",
        description="AHO: iterative agent harness optimizer",
    )

    # Required
    parser.add_argument("--benchmark", required=True, help="bfcl | tau-<domain>")
    parser.add_argument(
        "--optimizer", required=True, help="prism | better-harness | miprov2 | gepa"
    )
    parser.add_argument(
        "--inner-model",
        required=True,
        help="litellm model string for the scored agent (e.g. openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--outer-model",
        required=True,
        help="litellm model string for the proposer (e.g. anthropic/claude-opus-4-7)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)

    # Resource budget
    parser.add_argument(
        "--wall-time-s",
        type=float,
        default=300.0,
        help="Per-case wall-clock timeout in seconds (default 300)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Max tool calls/turns per case (default 100)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500_000, help="Max tokens per case (default 500k)"
    )

    # PRISM
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--train-cases", type=int, default=100)
    parser.add_argument("--holdout-cases", type=int, default=100)
    parser.add_argument("--mutations-per-gen", type=int, default=3)
    parser.add_argument(
        "--prism-pass-rate-metric",
        default="combined",
        choices=["combined", "holdout"],
        help="pass_rate used for frontier ranking: combined=(train+holdout)/total "
        "(default, original behavior) or holdout=holdout-only",
    )
    parser.add_argument(
        "--prism-prompt-only",
        action="store_true",
        default=False,
        help="PRISM variant: p0 always uses prompt_only (never touches middleware). "
        "Equivalent to prism_prompt_only in the EMNLP experiment plan.",
    )

    # PRISM §6.3 component ablations — each reverts exactly one Table 1 attribute
    parser.add_argument(
        "--prism-no-route",
        action="store_true",
        default=False,
        help="Ablation PRISM-NoRoute: failure clusters are computed but NOT routed "
        "to surface slots; every mutation slot is full-access (prompt+middleware) "
        "over all failures.",
    )
    parser.add_argument(
        "--prism-no-gate",
        action="store_true",
        default=False,
        help="Ablation PRISM-NoGate: the gate (holdout) split is not consulted during "
        "search — in-loop scoring and frontier selection use the repair (train) split "
        "only; gate evaluated once at final acceptance (GEPA/MIPROv2 discipline).",
    )
    parser.add_argument(
        "--prism-no-matrix",
        action="store_true",
        default=False,
        help="Ablation PRISM-NoMatrix: no cross-iteration failure matrix; clustering "
        "operates on the current generation's failures only.",
    )
    parser.add_argument(
        "--prism-no-constraint",
        action="store_true",
        default=False,
        help="Ablation PRISM-NoConstraint: middleware edits are NOT restricted to the "
        "three tool-boundary patterns; the mutator may edit execution logic freely.",
    )
    parser.add_argument(
        "--prism-no-crossover",
        action="store_true",
        default=False,
        help="Ablation PRISM-NoCrossover: skip the crossover step entirely.",
    )
    parser.add_argument(
        "--prism-population-cap",
        type=int,
        default=5,
        help="PRISM Pareto frontier population cap (default 5). Frontier-retention "
        "ablation: 1 = single incumbent (Frontier-1), 10 = large frontier (Frontier-10). "
        "The cap binds under Pareto acceptance (--prism-pass-rate-metric with HoldoutPareto).",
    )
    parser.add_argument(
        "--bh-prompt-only",
        action="store_true",
        default=False,
        help="BH variant: proposer uses prompt_only (never touches middleware). "
        "Equivalent to bh_prompt_only in the EMNLP experiment plan.",
    )

    # BetterHarness
    parser.add_argument("--max-iterations", type=int, default=10)

    # MIPROv2
    parser.add_argument(
        "--miprov2-num-candidates",
        type=int,
        default=10,
        help="Instruction candidates to generate (default 10)",
    )
    parser.add_argument(
        "--miprov2-num-trials",
        type=int,
        default=20,
        help="Bayesian optimization trials (default 20)",
    )
    parser.add_argument(
        "--miprov2-minibatch-size",
        type=int,
        default=25,
        help="Cases per trial evaluation (default 25)",
    )
    parser.add_argument(
        "--miprov2-full-eval-steps",
        type=int,
        default=10,
        help="Run full train eval every N trials (default 10)",
    )
    parser.add_argument("--seed", type=int, default=42)

    # GEPA
    parser.add_argument(
        "--gepa-max-metric-calls",
        type=int,
        default=200,
        help="Total evaluate() calls budget (default 200)",
    )
    parser.add_argument(
        "--gepa-reflection-minibatch-size",
        type=int,
        default=15,
        help="Trainset cases per reflection minibatch (default 15)",
    )
    parser.add_argument(
        "--gepa-middleware",
        action="store_true",
        default=False,
        help="GEPA-MW: expose the pattern-guided tool-boundary middleware surface "
        "as a second GEPA component (same three edit patterns as PRISM).",
    )
    parser.add_argument(
        "--miprov2-middleware",
        action="store_true",
        default=False,
        help="MIPROv2-MW: expose the pattern-guided tool-boundary middleware surface "
        "as a second optimized predictor (same three edit patterns as PRISM).",
    )

    # Shared
    parser.add_argument(
        "--outer-max-turns",
        type=int,
        default=300,
        help="Max turns for the outer proposer agent (default 300)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output directory"
    )
    parser.add_argument(
        "--human-approval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause at each decision point for human override",
    )

    # Cross-validation split seed
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for train/holdout split (BFCL: 0-3 for 4-fold CV; tau: any int)",
    )
    parser.add_argument(
        "--split-pool",
        default="base",
        choices=["base", "full"],
        help="Task pool for tau-telecom splits: 'base' (default, curated 114-task "
        "balanced pool) or 'full' (all tasks). Ignored for BFCL.",
    )

    # tau-bench
    parser.add_argument(
        "--tau-data-dir", default=None, help="Path to tau2-bench data dir (sets TAU2_DATA_DIR)"
    )
    parser.add_argument(
        "--scorecard-trials",
        type=int,
        default=1,
        help="k for pass^k: run final scorecard k times; case passes only if it "
        "passes in ALL k trials (tau benchmarks only, default 1)",
    )

    # Shared baseline (Option A fairness fix)
    parser.add_argument(
        "--shared-baseline-dir",
        type=Path,
        default=None,
        help="Directory holding a pre-scored shared baseline (baseline/train.json "
        "and baseline/holdout.json). If set and the files exist they are "
        "reused; if missing the CLI scores and saves them before running the "
        "optimizer. All four optimizers in a fold should point to the same "
        "directory so they share an identical baseline.",
    )

    # EMNLP experiment identity
    parser.add_argument(
        "--search-seed",
        type=int,
        default=0,
        help="Stochastic seed for the proposer LLM (default 0); vary per repeat",
    )
    parser.add_argument(
        "--repeat-id",
        type=int,
        default=0,
        help="Search repeat index 0,1,2,… (default 0); used for labeling only",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Run label (default: output-dir basename), e.g. 'bfcl-bh-s0-r2'",
    )
    parser.add_argument(
        "--condition-id",
        type=str,
        default=None,
        help="Condition label (default: auto from benchmark+split_seed), e.g. 'bfcl_random_s0'",
    )

    # Acceptance criterion
    parser.add_argument(
        "--acceptance",
        default="holdout_pass_rate",
        choices=["holdout_pass_rate", "holdout_pareto", "combined_pass_rate"],
        help="Acceptance criterion: holdout_pass_rate (default, strict >) | "
        "holdout_pareto (Pareto dominance on holdout pass_rate + reliability) | "
        "combined_pass_rate (simple avg of train + holdout pass_rate)",
    )

    args = parser.parse_args()

    from agent_harness_optimizer.framework.acceptance import (
        CombinedPassRate,
        HoldoutPareto,
        HoldoutPassRate,
    )

    _acceptance_map = {
        "holdout_pass_rate": HoldoutPassRate(),
        "holdout_pareto": HoldoutPareto(),
        "combined_pass_rate": CombinedPassRate(),
    }
    acceptance = _acceptance_map.get(args.acceptance)
    if acceptance is None:
        raise ValueError(
            f"Unknown acceptance criterion: {args.acceptance!r}. "
            f"Choose from: {list(_acceptance_map)}"
        )

    benchmark = _build_benchmark(args)

    case_split = None
    if args.benchmark.startswith("tau-") and args.split_seed is not None:
        case_split = _build_tau_case_split(args, benchmark)

    shared_baseline_dir = _ensure_shared_baseline(args, benchmark, case_split)

    _experiment_id = args.experiment_id or args.output_dir.name
    _condition_id = (
        (args.condition_id or f"{args.benchmark}_random_s{args.split_seed}")
        if args.split_seed is not None
        else (args.condition_id or args.benchmark)
    )

    config = OptimizeConfig(
        output_dir=args.output_dir,
        inner_model=args.inner_model,
        outer_model=args.outer_model,
        outer_max_turns=args.outer_max_turns,
        resume=args.resume,
        human_approval=args.human_approval,
        split_seed=args.split_seed,
        case_split=case_split,
        acceptance=acceptance,
        shared_baseline_dir=shared_baseline_dir,
        experiment_id=_experiment_id,
        condition_id=_condition_id,
        repeat_id=args.repeat_id,
        search_seed=args.search_seed,
        num_scorecard_trials=args.scorecard_trials,
    )
    optimizer = _build_optimizer(args, benchmark, config)
    optimizer.run()


if __name__ == "__main__":
    main()
