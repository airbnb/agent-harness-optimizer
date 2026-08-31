"""PRISMOptimizer — benchmark-agnostic evolutionary prompt optimization.

Algorithm per generation:
  1. Pick best viable — max(frontier, key=(holdout_pass_rate, reliability)), zero rollouts
  2. Reconstruct train SplitScore from stored per_case, zero rollouts
  3. LLM root-cause analysis + mutations:
       p0 always: prompt_middleware_both, all failures (BH behavior)
       p1 optional: prompt_only, if LLM found prompt-only clusters
       p2 optional: middleware_only, if LLM found middleware-only clusters
  4. Full eval — all children scored on train + holdout in parallel
  5. Crossover — conditional on ≥2 children with complementary failures
  6. Pareto    — update frontier via (holdout_pass_rate, reliability) dominance
"""

from __future__ import annotations

import asyncio
import difflib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from agent_harness_optimizer.framework.acceptance import HoldoutPareto
from agent_harness_optimizer.framework.benchmark import Benchmark, CaseScore, SplitScore
from agent_harness_optimizer.framework.optimizer import OptimizeConfig, Optimizer
from agent_harness_optimizer.framework.report import build_report, print_report
from agent_harness_optimizer.optimizers.prism.proposer import (
    analyze_patterns,
    crossover_all_children,
    mutate,
)
from agent_harness_optimizer.optimizers.prism.types import Candidate, pareto_frontier
from agent_harness_optimizer.utils.harness import harness_hash


def _write_diff(out: Path, baseline_prompt: str, final_prompt: str, final_mw: Path | None) -> None:
    """Write final_diff.md: unified diff of prompt + middleware changes."""
    lines = ["# Final Diff — Baseline → Best Accepted\n"]
    prompt_diff = list(
        difflib.unified_diff(
            baseline_prompt.splitlines(keepends=True),
            final_prompt.splitlines(keepends=True),
            fromfile="baseline/system_prompt.txt",
            tofile="best/system_prompt.txt",
        )
    )
    if prompt_diff:
        lines.append("## System Prompt\n\n```diff\n")
        lines.extend(prompt_diff)
        lines.append("```\n")
    else:
        lines.append("## System Prompt\n\n_(no change)_\n")

    if final_mw and final_mw.is_dir():
        lines.append("\n## Middleware\n")
        for f in sorted(final_mw.iterdir()):
            if f.is_file():
                lines.append(f"\n### `{f.name}`\n\n```python\n")
                lines.append(f.read_text())
                lines.append("```\n")
    else:
        lines.append("\n## Middleware\n\n_(none)_\n")

    (out / "final_diff.md").write_text("".join(lines))


@dataclass
class PRISMConfig:
    train_cases: int = 100
    holdout_cases: int = 100
    mutations_per_gen: int = 3
    population_cap: int = 5
    seed: int = 42
    pass_rate_metric: str = "combined"  # "combined"=(train+holdout)/total; "holdout"=holdout-only
    prompt_only: bool = False  # when True, p0 is always prompt_only; middleware is never touched

    # ------------------------------------------------------------------
    # §6.3 component ablation flags — each reverts exactly one Table 1
    # attribute to the baselines' setting, everything else held fixed.
    # ------------------------------------------------------------------
    # NoRoute: the failure-clustering analyst still runs, but clusters are NOT
    # routed to surface-specific mutation slots — every slot is full-access
    # (prompt+middleware) over all failures.
    no_route: bool = False
    # NoGate: the gate (holdout) split is not consulted during search — child
    # scoring and frontier selection use the repair (train) split only; the gate
    # is evaluated once, at final acceptance (GEPA/MIPROv2 discipline).
    no_gate: bool = False
    # NoMatrix: no cross-iteration failure matrix — clustering operates on the
    # current generation's failures only.
    no_matrix: bool = False
    # NoConstraint: middleware edits are NOT restricted to the three
    # tool-boundary patterns; the mutator may edit execution logic freely.
    no_constraint: bool = False
    # NoCrossover: the crossover step is skipped entirely.
    no_crossover: bool = False


class PRISMOptimizer(Optimizer):
    """Genetic-Pareto Evolutionary Prompt Algorithm.

    Usage::

        benchmark = BFCLBenchmark(resource_budget=ResourceBudget(wall_time_s=120))
        config = OptimizeConfig(output_dir=Path("runs/exp1"),
                                inner_model="azure/gpt-5.4-mini",
                                outer_model="bedrock/claude-opus-4-5")
        PRISMOptimizer(benchmark, config, generations=10).run()
    """

    def __init__(
        self,
        benchmark: Benchmark,
        config: OptimizeConfig,
        *,
        generations: int = 10,
        prism_config: PRISMConfig | None = None,
    ) -> None:
        super().__init__(benchmark, config)
        self.generations = generations
        self.gc = prism_config or PRISMConfig()

    def run(self) -> None:
        asyncio.run(self._run_async())

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    async def _score(
        self,
        prompt: str,
        split: str,
        out_dir: Path,
        middleware_dir: Path | None,
        max_cases: int | None,
        label: str = "",
        case_indices: list[int] | None = None,
        num_trials: int = 1,
    ) -> SplitScore:
        out_dir.mkdir(parents=True, exist_ok=True)
        if label:
            print(f"[prism]   scoring {label}…")
        return await self.benchmark.score_async(
            prompt,
            split,
            out_dir,
            middleware_dir=middleware_dir,
            max_cases=max_cases,
            case_indices=case_indices,
            num_trials=num_trials,
        )

    # ------------------------------------------------------------------
    # Failure matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _build_failure_matrix(
        frontier: list[Candidate],
        all_candidates: list[Candidate],
    ) -> dict[str, str]:
        """Build failure labels using both accepted (frontier) and all-attempted history.

        Accepted-history labels (from frontier, generation-ordered):
          FIXED      — passing now on frontier, was failing in a prior accepted generation
          NEW        — failing now on frontier, was passing in a prior accepted generation (regression)
          PERSISTENT — failing across every accepted generation including current
          RECURRING  — failing now on frontier, but was passing in at least one prior accepted generation

        All-attempts enrichment (upgrades PERSISTENT cases using rejected children):
          PERSISTENT  → stays PERSISTENT if every mutation ever tried also failed it
          RECURRING   → upgraded from PERSISTENT if any rejected child passed it
                        (it's solvable — some mutation found a fix, just wasn't accepted)

        This lets the proposer distinguish truly hard cases (every attempt fails) from
        cases where a discarded mutation already solved the problem.
        """
        # Step 1: labels from accepted frontier history
        ordered = sorted(frontier, key=lambda c: c.generation)
        accepted_history: dict[str, list[bool]] = {}
        for c in ordered:
            for row in c.per_case:
                cid = row["case_id"]
                accepted_history.setdefault(cid, []).append(bool(row["passed"]))
        labels: dict[str, str] = {}
        for cid, hist in accepted_history.items():
            currently_passing = hist[-1]
            ever_failed_before = any(not p for p in hist[:-1])
            ever_passed_before = any(p for p in hist[:-1])
            if currently_passing and ever_failed_before:
                labels[cid] = "FIXED"
            elif not currently_passing and not ever_passed_before:
                labels[cid] = "PERSISTENT"
            elif not currently_passing and ever_passed_before:
                labels[cid] = "RECURRING"
            elif not currently_passing and not ever_failed_before:
                labels[cid] = "NEW"

        # Step 2: upgrade PERSISTENT → RECURRING for cases a rejected child solved
        # Any candidate not currently on the frontier is a rejected mutation attempt.
        frontier_set = set(id(c) for c in frontier)
        rejected = [c for c in all_candidates if id(c) not in frontier_set]
        solved_by_rejected: set[str] = set()
        for c in rejected:
            for row in c.per_case:
                if bool(row["passed"]):
                    solved_by_rejected.add(row["case_id"])
        for cid in list(labels):
            if labels[cid] == "PERSISTENT" and cid in solved_by_rejected:
                labels[cid] = "RECURRING"
        return labels

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_gen(
        self,
        gen: int,
        frontier: list[Candidate],
        all_candidates: list[Candidate],
        screen_rollouts: int = 0,
        gen_stats: dict | None = None,
    ) -> None:
        d = self.config.output_dir / f"gen-{gen:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "frontier.json").write_text(json.dumps([c.to_dict() for c in frontier], indent=2))
        (d / "all_candidates.json").write_text(
            json.dumps([c.to_dict() for c in all_candidates], indent=2)
        )
        (d / "screen_rollouts.json").write_text(json.dumps({"screen_rollouts": screen_rollouts}))
        if gen_stats is not None:
            (d / "gen_stats.json").write_text(json.dumps(gen_stats, indent=2))

    def _load_state(self) -> tuple[int, list[Candidate], list[Candidate], int]:
        gen_dirs = sorted(self.config.output_dir.glob("gen-*"))
        # Find the latest gen dir that has a complete frontier.json (guards against
        # interrupted runs that created the dir but didn't finish writing).
        last = next((d for d in reversed(gen_dirs) if (d / "frontier.json").exists()), None)
        if last is None:
            return -1, [], [], 0
        frontier = [
            Candidate.from_dict(d) for d in json.loads((last / "frontier.json").read_text())
        ]
        all_cands = [
            Candidate.from_dict(d) for d in json.loads((last / "all_candidates.json").read_text())
        ]
        sr_path = last / "screen_rollouts.json"
        screen_rollouts = (
            json.loads(sr_path.read_text())["screen_rollouts"] if sr_path.exists() else 0
        )
        return int(last.name.split("-")[1]), frontier, all_cands, screen_rollouts

    def _save_report(
        self,
        frontier: list[Candidate],
        all_candidates: list[Candidate],
        baseline_prompt: str,
        final_scorecard=None,
        baseline_scorecard: SplitScore | None = None,
        final_train: SplitScore | None = None,
        final_holdout: SplitScore | None = None,
        screen_rollouts: int = 0,
        search_outer_calls: int = 0,
        search_outer_tokens_in: int = 0,
        search_outer_tokens_out: int = 0,
    ) -> None:
        best = max(frontier, key=lambda c: c.pass_rate, default=None)
        seed = all_candidates[0] if all_candidates else None

        # holdout series: baseline (gen 0) + best per generation
        by_gen: dict[int, list[Candidate]] = {}
        for c in all_candidates:
            by_gen.setdefault(c.generation, []).append(c)
        holdout_series = []
        for g in sorted(by_gen):
            best_g = max(by_gen[g], key=lambda c: c.pass_rate)
            holdout_series.append(
                {
                    "generation": g,
                    "uid": best_g.uid,
                    "holdout_passed": best_g.holdout_passed,
                    "holdout_total": best_g.holdout_total,
                    "holdout_pass_rate": round(best_g.holdout_passed / best_g.holdout_total, 4)
                    if best_g.holdout_total
                    else 0.0,
                }
            )

        # Reconstruct SplitScore objects from Candidate fields for build_report().
        def _cand_train_score(c: Candidate) -> SplitScore:
            return SplitScore(
                passed=c.train_passed,
                total=c.train_total,
                reliability=c.reliability,
                prompt_tokens_per_case=c.prompt_tokens_per_case,
                completion_tokens_per_case=c.completion_tokens_per_case,
            )

        def _cand_holdout_score(c: Candidate) -> SplitScore:
            return SplitScore(
                passed=c.holdout_passed,
                total=c.holdout_total,
                reliability=1.0,
            )

        base_train = _cand_train_score(seed) if seed else SplitScore(0, 0, 1.0)
        base_holdout = _cand_holdout_score(seed) if seed else SplitScore(0, 0, 1.0)
        # Prefer fresh bookend scores (have stuck_breakdown); fall back to Candidate fields.
        ft = (
            final_train
            if final_train is not None
            else (_cand_train_score(best) if best else base_train)
        )
        fh = (
            final_holdout
            if final_holdout is not None
            else (_cand_holdout_score(best) if best else base_holdout)
        )

        # optimization_rollouts: full evals (winner+crossover per gen) + screen/retrain cases.
        # Seed (all_candidates[0]) is the baseline; subtract it from full-eval count.
        # NoGate: children are scored on train only during search.
        _cases_per_eval = self.gc.train_cases + (0 if self.gc.no_gate else self.gc.holdout_cases)
        full_eval_rollouts = max(0, len(all_candidates) - 1) * _cases_per_eval
        opt_rollouts = full_eval_rollouts + screen_rollouts

        _best_hash = harness_hash(best.prompt, best.middleware_dir) if best else ""
        report = build_report(
            optimizer="prism",
            benchmark_name=self.benchmark.name,
            inner_model=self.config.inner_model,
            outer_model=self.config.outer_model,
            split_seed=self.config.split_seed,
            acceptance_criterion=type(self.acceptance).__name__,
            base_train=base_train,
            base_holdout=base_holdout,
            final_train=ft,
            final_holdout=fh,
            final_scorecard=final_scorecard,
            baseline_scorecard=baseline_scorecard,
            optimization_rollouts=opt_rollouts,
            experiment_id=self.config.experiment_id,
            condition_id=self.config.condition_id,
            repeat_id=self.config.repeat_id,
            search_seed=self.config.search_seed,
            num_scorecard_trials=self.config.num_scorecard_trials,
            search_outer_calls=search_outer_calls,
            search_outer_tokens_in=search_outer_tokens_in,
            search_outer_tokens_out=search_outer_tokens_out,
            optimizer_config={
                "generations": self.generations,
                "mutations_per_gen": self.gc.mutations_per_gen,
                "train_cases": self.gc.train_cases,
                "holdout_cases": self.gc.holdout_cases,
                "seed": self.gc.seed,
                "best_uid": best.uid if best else None,
                "frontier_size": len(frontier),
                "total_evaluated": len(all_candidates),
                "screen_rollouts": screen_rollouts,
                "full_eval_rollouts": full_eval_rollouts,
                "baseline_harness_hash": harness_hash(baseline_prompt, None),
                "selected_harness_hash": _best_hash,
            },
        )
        report["holdout_series"] = holdout_series
        report["per_candidate_history"] = [c.to_dict() for c in all_candidates]
        report["frontier"] = [c.to_dict() for c in sorted(frontier, key=lambda c: -c.pass_rate)]
        (self.config.output_dir / "report.json").write_text(json.dumps(report, indent=2))

        if best:
            _write_diff(self.config.output_dir, baseline_prompt, best.prompt, best.middleware_dir)

        print_report(report)
        print(f"  Report: {self.config.output_dir / 'report.json'}")

    # ------------------------------------------------------------------
    # Candidate metrics
    # ------------------------------------------------------------------

    def _apply_metrics(self, c: Candidate, train: SplitScore, holdout: SplitScore) -> None:
        c.train_passed = train.passed
        c.train_total = train.total
        c.holdout_passed = holdout.passed
        c.holdout_total = holdout.total
        if self.gc.no_gate:
            # NoGate ablation: the gate split is not consulted during search, so
            # the in-loop pass_rate is computed from the repair (train) split only.
            c.pass_rate = train.passed / train.total if train.total else 0.0
        elif self.gc.pass_rate_metric == "combined":
            total = train.total + holdout.total
            c.pass_rate = (train.passed + holdout.passed) / total if total else 0.0
        else:
            c.pass_rate = holdout.passed / holdout.total if holdout.total else 0.0
        c.reliability = train.reliability
        c.prompt_tokens_per_case = train.prompt_tokens_per_case
        c.completion_tokens_per_case = train.completion_tokens_per_case
        # Merge train+holdout per_case so failure_matrix spans both splits.
        # train cases keyed by case_id; holdout cases appended (different index space).
        seen: set[str] = set()
        merged = []
        for cs in train.cases:
            merged.append(cs.to_dict())
            seen.add(cs.case_id)
        for cs in holdout.cases:
            if cs.case_id not in seen:
                merged.append(cs.to_dict())
        c.per_case = merged

    # ------------------------------------------------------------------
    # Main async loop
    # ------------------------------------------------------------------

    async def _run_async(self) -> None:
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        config_path = out / "experiment_config.json"
        if not config_path.exists():
            config_path.write_text(
                json.dumps(
                    {
                        "benchmark": self.benchmark.name,
                        "inner_model": self.config.inner_model,
                        "outer_model": self.config.outer_model,
                        "resource_budget": self.benchmark.resource_budget.to_dict(),
                        "generations": self.generations,
                        **vars(self.gc),
                    },
                    indent=2,
                )
            )

        # Warm up auth on the main thread so threaded outer-agent calls inherit valid credentials
        self.benchmark.build_model(self.config.outer_model)

        if self.config.resume:
            last_gen, frontier, all_candidates, screen_rollouts = self._load_state()
        else:
            last_gen, frontier, all_candidates, screen_rollouts = -1, [], [], 0

        baseline_prompt = self.benchmark.default_system_prompt

        # Case indices for stratified splits — must be set before any scoring call,
        # including the resume path where the seeding block is skipped.
        _cs = self.config.case_split
        _train_idx = _cs.train if _cs else None
        _holdout_idx = _cs.holdout if _cs else None

        _outer_calls_total = 0
        _outer_tok_in_total = 0
        _outer_tok_out_total = 0
        search_trace: list[dict] = []

        if not frontier:
            print("[prism] === Generation 0: seeding ===")
            seed_dir = out / "gen-000"
            seed_dir.mkdir(parents=True, exist_ok=True)
            initial_prompt = baseline_prompt
            _seed_t0 = time.monotonic()
            _sbd = self.config.shared_baseline_dir
            if (
                _sbd
                and (_sbd / "baseline" / "train.json").exists()
                and (_sbd / "baseline" / "holdout.json").exists()
            ):
                train = SplitScore.from_dict(
                    json.loads((_sbd / "baseline" / "train.json").read_text())
                )
                holdout = SplitScore.from_dict(
                    json.loads((_sbd / "baseline" / "holdout.json").read_text())
                )
                print(
                    f"[prism] shared baseline loaded: train={train.passed}/{train.total} "
                    f"holdout={holdout.passed}/{holdout.total}"
                )
            else:
                train = await self._score(
                    initial_prompt,
                    self.config.train_split,
                    seed_dir / "train",
                    None,
                    self.gc.train_cases,
                    "seed train",
                    case_indices=_train_idx,
                )
                holdout = await self._score(
                    initial_prompt,
                    self.config.holdout_split,
                    seed_dir / "holdout",
                    None,
                    self.gc.holdout_cases,
                    "seed holdout",
                    case_indices=_holdout_idx,
                )
            seed = Candidate(
                uid="gen000_seed", generation=0, prompt=initial_prompt, middleware_dir=None
            )
            self._apply_metrics(seed, train, holdout)
            seed.eval_wall_time_s = round(time.monotonic() - _seed_t0, 2)
            (out / "cache" / "gen000_seed_train.json").parent.mkdir(exist_ok=True)
            (out / "cache" / "gen000_seed_train.json").write_text(json.dumps(train.to_dict()))
            frontier = [seed]
            all_candidates = [seed]
            self._save_gen(0, frontier, all_candidates, screen_rollouts)
            last_gen = 0
            print(
                f"[prism] Seed: pass_rate={seed.pass_rate:.3f} reliability={seed.reliability:.3f} "
                f"train={seed.train_passed}/{seed.train_total} holdout={seed.holdout_passed}/{seed.holdout_total} "
                f"eval_wall_time_s={seed.eval_wall_time_s}"
            )

        loop = asyncio.get_event_loop()

        for gen in range(max(1, last_gen + 1), self.generations + 1):
            print(f"\n[prism] === Generation {gen}/{self.generations} ===")
            best_in_frontier = max(frontier, key=lambda c: c.pass_rate)
            print(
                f"[prism] Frontier: {len(frontier)}, best pass_rate={best_in_frontier.pass_rate:.3f}"
            )
            gen_dir = out / f"gen-{gen:03d}"
            gen_dir.mkdir(parents=True, exist_ok=True)

            if self.gc.no_matrix:
                # NoMatrix ablation: no cross-iteration failure tracking — the
                # analyst clusters the current generation's failures only.
                fm_cases: dict[str, str] = {}
            else:
                fm_cases = self._build_failure_matrix(frontier, all_candidates)

            # Step 1: pick best viable from frontier using existing scores (no re-screen needed)
            best_viable = max(frontier, key=lambda c: (c.pass_rate, c.reliability))

            # Step 2: reconstruct train SplitScore from stored per_case — train rows come first,
            # numbered by train_total. Saves 100 rollouts vs re-scoring with identical pinned indices.
            _train_rows = best_viable.per_case[: best_viable.train_total]
            best_train = SplitScore(
                passed=best_viable.train_passed,
                total=best_viable.train_total,
                reliability=best_viable.reliability,
                prompt_tokens_per_case=best_viable.prompt_tokens_per_case,
                completion_tokens_per_case=best_viable.completion_tokens_per_case,
                cases=[CaseScore.from_dict(r) for r in _train_rows],
            )

            # Step 3: LLM root-cause analysis → 3 fixed-variant mutations.
            #
            # Always run exactly 3 mutations, one per surface:
            #   prompt_middleware_both — exact BH behavior: all failures visible, no filtering
            #   prompt_only           — PRISM addition: targets cases best fixed by prompt rules
            #   middleware_only       — PRISM addition: targets cases best fixed by middleware
            #
            # LLM analysis identifies root-cause clusters and assigns a fix_surface to each.
            # prompt_only and middleware_only see only their surface's clusters' case_ids.
            # prompt_middleware_both always sees all failures (target_pattern=None).
            print("[prism] Step 3: analyzing failure root causes (3 variants: both/prompt/mw)…")
            analysis_ws = gen_dir / f"workspace_analyze_{best_viable.uid}"
            # Ask for enough clusters to give good signal for surface routing (2 × mutations)
            n_clusters = max(self.gc.mutations_per_gen, 4)
            llm_patterns, _analyze_tok_in, _analyze_tok_out = analyze_patterns(
                best_train,
                n=n_clusters,
                outer_model=self.config.outer_model,
                workspace_dir=analysis_ws,
                fm_cases=fm_cases,
            )
            _outer_calls_total += 1  # one analyze_patterns call per generation
            _outer_tok_in_total += _analyze_tok_in
            _outer_tok_out_total += _analyze_tok_out

            # If LLM analysis yielded nothing, fall back to heuristic patterns
            if not llm_patterns:
                llm_patterns = self.benchmark.extract_top_patterns(best_train, n=n_clusters)

            # Build surface-specific patterns from clusters tagged for that surface only.
            # prompt_middleware_both clusters are covered by p0 — not duplicated here.
            def _surface_pattern(surface: str) -> dict | None:
                ids: list[str] = []
                reasons: list[str] = []
                for cl in llm_patterns:
                    if cl.get("fix_surface") == surface:
                        for cid in cl.get("case_ids", []):
                            if cid not in ids:
                                ids.append(cid)
                        reasons.append((cl.get("root_cause") or cl.get("key", ""))[:60])
                if not ids:
                    return None
                return {
                    "root_cause": "; ".join(reasons[:3]),
                    "fix_surface": surface,
                    "case_ids": ids,
                }

            prompt_pattern = _surface_pattern("prompt_only")
            mw_pattern = _surface_pattern("middleware_only")

            # p0 always runs (BH-style, all failures).
            # p1/p2 only run if LLM found clusters exclusively for that surface.
            # When prompt_only=True (prism_prompt_only variant), p0 is always prompt_only
            # and middleware variants are never added.
            if self.gc.prompt_only:
                mutation_specs: list[tuple[str, dict | None]] = [("prompt_only", None)]
                # Pad to mutations_per_gen with additional prompt_only attempts
                while len(mutation_specs) < self.gc.mutations_per_gen:
                    mutation_specs.append(("prompt_only", None))
            elif self.gc.no_route:
                # NoRoute ablation: clusters are computed (analyze_patterns above)
                # but NOT routed to surface slots — every mutation slot is
                # full-access (prompt+middleware) over all failures.
                mutation_specs = [("prompt_middleware_both", None)] * self.gc.mutations_per_gen
            else:
                mutation_specs = [("prompt_middleware_both", None)]
                if prompt_pattern:
                    mutation_specs.append(("prompt_only", prompt_pattern))
                elif len(mutation_specs) < self.gc.mutations_per_gen:
                    # Force at least one prompt_only attempt for surface diversity
                    mutation_specs.append(("prompt_only", None))
                if mw_pattern and len(mutation_specs) < self.gc.mutations_per_gen:
                    mutation_specs.append(("middleware_only", mw_pattern))
                # Pad remaining slots to guarantee mutations_per_gen mutations
                while len(mutation_specs) < self.gc.mutations_per_gen:
                    mutation_specs.append(("prompt_middleware_both", None))

            n_failures = len([c for c in best_train.cases if not c.passed])
            print(f"[prism] Step 3: {len(mutation_specs)} parallel mutations on {best_viable.uid}…")
            print(
                f"  p0 {'prompt_only' if self.gc.prompt_only else 'prompt_middleware_both'}: all {n_failures} failures"
            )
            if not self.gc.prompt_only and not self.gc.no_route:
                if prompt_pattern:
                    print(
                        f"  p1 prompt_only: {len(prompt_pattern['case_ids'])} cases — {prompt_pattern['root_cause'][:80]}"
                    )
                if mw_pattern:
                    print(
                        f"  p{len(mutation_specs) - 1} middleware_only: {len(mw_pattern['case_ids'])} cases — {mw_pattern['root_cause'][:80]}"
                    )

            async def _one_mutation(
                idx: int, variant: str, pattern: dict | None
            ) -> tuple[Candidate | None, int, int]:
                ws = gen_dir / f"workspace_mutate_p{idx}_{best_viable.uid}"
                try:
                    _prop_t0 = time.monotonic()
                    (
                        new_prompt,
                        new_mw,
                        proposal,
                        used_variant,
                        _tok_in,
                        _tok_out,
                    ) = await loop.run_in_executor(
                        None,
                        lambda _p=pattern, _ws=ws, _v=variant: mutate(
                            best_viable,
                            best_train,
                            generation=gen,
                            frontier=frontier,
                            all_candidates=all_candidates,
                            fm_cases=fm_cases,
                            benchmark=self.benchmark,
                            outer_model=self.config.outer_model,
                            workspace_dir=_ws,
                            max_turns=self.config.outer_max_turns,
                            target_pattern=_p,
                            variant=_v,
                            unconstrained=self.gc.no_constraint,
                        ),
                    )
                    prop_wall = round(time.monotonic() - _prop_t0, 2)
                    uid = f"gen{gen:03d}_m{idx}_{used_variant[:4]}"
                    child = Candidate(
                        uid=uid,
                        generation=gen,
                        prompt=new_prompt,
                        middleware_dir=new_mw,
                        parent_uids=[best_viable.uid],
                        proposal=f"[{used_variant}] {proposal}",
                    )
                    child.proposal_wall_time_s = prop_wall
                    # Dedup gate: discard if agent made no change
                    if (
                        new_prompt.strip() == best_viable.prompt.strip()
                        and new_mw == best_viable.middleware_dir
                    ):
                        print(f"[prism]   Mutation p{idx} produced no change — skipping eval")
                        return None, _tok_in, _tok_out
                    print(
                        f"[prism]   Mutation child {uid} generated (variant={used_variant}, proposal_wall_time_s={prop_wall})"
                    )
                    return child, _tok_in, _tok_out
                except Exception as exc:
                    print(f"[prism]   Mutation p{idx} failed: {exc}")
                    return None, 0, 0

            mut_results = await asyncio.gather(
                *[_one_mutation(i, v, p) for i, (v, p) in enumerate(mutation_specs)]
            )
            children: list[Candidate] = [c for c, _, _ in mut_results if c is not None]
            _outer_calls_total += len(children)  # one mutate() call per successful child
            _outer_tok_in_total += sum(ti for _, ti, _ in mut_results)
            _outer_tok_out_total += sum(to for _, _, to in mut_results)

            if not children:
                print("[prism] No children — skipping eval")
                self._save_gen(
                    gen,
                    frontier,
                    all_candidates,
                    screen_rollouts,
                    gen_stats={
                        "generation": gen,
                        "crossover": "no_children",
                        "complementary_cases": 0,
                        "crossover_base_uid": None,
                        "crossover_uid": None,
                        "crossover_in_frontier": False,
                    },
                )
                continue

            # Step 4: full eval all children (train + holdout in parallel)
            full_eval_candidates = list(children)
            print(f"[prism] Step 4: full eval of {len(full_eval_candidates)} candidates…")
            _eval_t0 = time.monotonic()
            eval_tasks = []
            for ch in full_eval_candidates:
                eval_tasks.append(
                    self._score(
                        ch.prompt,
                        self.config.train_split,
                        gen_dir / f"train_{ch.uid}",
                        ch.middleware_dir,
                        self.gc.train_cases,
                        f"{ch.uid} train",
                        case_indices=_train_idx,
                    )
                )
                if not self.gc.no_gate:
                    eval_tasks.append(
                        self._score(
                            ch.prompt,
                            self.config.holdout_split,
                            gen_dir / f"holdout_{ch.uid}",
                            ch.middleware_dir,
                            self.gc.holdout_cases,
                            f"{ch.uid} holdout",
                            case_indices=_holdout_idx,
                        )
                    )
            eval_scores = await asyncio.gather(*eval_tasks)
            eval_wall = round(time.monotonic() - _eval_t0, 2)

            _stride = 1 if self.gc.no_gate else 2
            _empty_holdout = SplitScore(passed=0, total=0, reliability=1.0)
            print(f"\n[prism] Generation {gen} child eval results (eval_wall_time_s={eval_wall}):")
            child_train_scores: list = []
            for idx, ch in enumerate(full_eval_candidates):
                tr = eval_scores[idx * _stride]
                ho = _empty_holdout if self.gc.no_gate else eval_scores[idx * _stride + 1]
                self._apply_metrics(ch, tr, ho)
                ch.eval_wall_time_s = eval_wall
                child_train_scores.append(tr)
                print(
                    f"  {ch.uid}: pass_rate={ch.pass_rate:.3f} reliability={ch.reliability:.3f} "
                    f"train={ch.train_passed}/{ch.train_total} holdout={ch.holdout_passed}/{ch.holdout_total}"
                )

            # Step 5: crossover — uses full-eval train scores so agent sees real pass/fail per child.
            # Gate: only run if ≥2 children AND at least one non-base child can solve cases the
            # best child fails (complementarity > 0). Saves 200 rollouts when children are redundant.
            cross: Candidate | None = None
            # Structured crossover telemetry, persisted per generation in
            # gen_stats.json so firing rates are reportable without log scraping.
            _cx_status = "single_child"
            _cx_complement = 0
            _cross_key = (
                (lambda c: c.train_pass_rate)
                if self.gc.no_gate
                else (lambda c: c.holdout_pass_rate)
            )
            crossover_base = max(full_eval_candidates, key=_cross_key)
            if self.gc.no_crossover:
                _cx_status = "disabled"
                print("[prism] Step 5: skipping crossover — NoCrossover ablation")
            elif len(full_eval_candidates) >= 2:
                base_fails = {r["case_id"] for r in crossover_base.per_case if not r["passed"]}
                others_unique = {
                    r["case_id"]
                    for c in full_eval_candidates
                    if c is not crossover_base
                    for r in c.per_case
                    if r["passed"]
                } & base_fails
                if not others_unique:
                    _cx_status = "no_complement"
                    print(
                        f"[prism] Step 5: skipping crossover — no complementary cases (base={crossover_base.uid} dominates all others)"
                    )
                else:
                    _cx_status = "fired"
                    _cx_complement = len(others_unique)
                    print(
                        f"[prism] Step 5: crossover ({len(others_unique)} complementary cases, base={crossover_base.uid})…"
                    )
                    ws_cross = gen_dir / f"workspace_cross_all_gen{gen:03d}"
                    try:
                        (
                            new_prompt,
                            new_mw,
                            proposal,
                            _cx_tok_in,
                            _cx_tok_out,
                        ) = await loop.run_in_executor(
                            None,
                            lambda: crossover_all_children(
                                full_eval_candidates,
                                child_train_scores,
                                crossover_base,
                                generation=gen,
                                frontier=frontier,
                                all_candidates=all_candidates,
                                fm_cases=fm_cases,
                                benchmark=self.benchmark,
                                outer_model=self.config.outer_model,
                                workspace_dir=ws_cross,
                                max_turns=self.config.outer_max_turns,
                            ),
                        )
                        cross = Candidate(
                            uid=f"gen{gen:03d}_cro_all",
                            generation=gen,
                            prompt=new_prompt,
                            middleware_dir=new_mw,
                            parent_uids=[c.uid for c in full_eval_candidates],
                            proposal=f"[crossover-all] {proposal}",
                        )
                        _outer_calls_total += 1  # one crossover call per generation
                        _outer_tok_in_total += _cx_tok_in
                        _outer_tok_out_total += _cx_tok_out
                        print(f"[prism]   Crossover-all generated (base={crossover_base.uid})")
                    except Exception as exc:
                        _cx_status = "failed"
                        print(f"[prism]   Crossover-all failed: {exc}")
            else:
                print("[prism] Step 5: skipping crossover — only 1 mutation child")

            children = full_eval_candidates + ([cross] if cross else [])

            # Full eval crossover candidate
            if cross:
                _cx_t0 = time.monotonic()
                if self.gc.no_gate:
                    cx_train = await self._score(
                        cross.prompt,
                        self.config.train_split,
                        gen_dir / f"train_{cross.uid}",
                        cross.middleware_dir,
                        self.gc.train_cases,
                        f"{cross.uid} train",
                        case_indices=_train_idx,
                    )
                    cx_holdout = _empty_holdout
                else:
                    cx_train, cx_holdout = await asyncio.gather(
                        self._score(
                            cross.prompt,
                            self.config.train_split,
                            gen_dir / f"train_{cross.uid}",
                            cross.middleware_dir,
                            self.gc.train_cases,
                            f"{cross.uid} train",
                            case_indices=_train_idx,
                        ),
                        self._score(
                            cross.prompt,
                            self.config.holdout_split,
                            gen_dir / f"holdout_{cross.uid}",
                            cross.middleware_dir,
                            self.gc.holdout_cases,
                            f"{cross.uid} holdout",
                            case_indices=_holdout_idx,
                        ),
                    )
                self._apply_metrics(cross, cx_train, cx_holdout)
                cross.eval_wall_time_s = round(time.monotonic() - _cx_t0, 2)
                print(
                    f"  {cross.uid}: pass_rate={cross.pass_rate:.3f} reliability={cross.reliability:.3f} "
                    f"train={cross.train_passed}/{cross.train_total} holdout={cross.holdout_passed}/{cross.holdout_total}"
                )

            print(f"\n[prism] Generation {gen} final summary:")
            for ch in children:
                print(
                    f"  {ch.uid}: pass_rate={ch.pass_rate:.3f} reliability={ch.reliability:.3f} "
                    f"train={ch.train_passed}/{ch.train_total} holdout={ch.holdout_passed}/{ch.holdout_total} "
                    f"prompt_tok/case={ch.prompt_tokens_per_case:.0f} completion_tok/case={ch.completion_tokens_per_case:.0f}"
                )

            # Step 6: update frontier using the plugged-in acceptance criterion.
            # HoldoutPareto → full Pareto dominance on (holdout_pass_rate, reliability).
            # HoldoutPassRate (default) → keep only the single best holdout candidate.
            all_candidates.extend(children)
            # NoGate ablation: all in-loop selection signals come from the repair
            # (train) split — the gate split is consulted only at final acceptance.
            _objective = "train" if self.gc.no_gate else "holdout"
            if isinstance(self.acceptance, HoldoutPareto):
                new_frontier = pareto_frontier(frontier + children, objective=_objective)
                if len(new_frontier) > self.gc.population_cap:
                    _rate = (
                        (lambda c: c.train_pass_rate)
                        if self.gc.no_gate
                        else (lambda c: c.holdout_pass_rate)
                    )
                    new_frontier = sorted(new_frontier, key=lambda c: -(_rate(c) + c.reliability))[
                        : self.gc.population_cap
                    ]
            else:
                # Strict pass count on the selection split: keep the single best
                # candidate. Ties broken by train reliability.
                _count = (
                    (lambda c: c.train_passed) if self.gc.no_gate else (lambda c: c.holdout_passed)
                )
                best = max(
                    frontier + children,
                    key=lambda c: (_count(c), c.reliability),
                )
                new_frontier = [best]

            added = [c.uid for c in new_frontier if c not in frontier]
            removed = [c.uid for c in frontier if c not in new_frontier]
            print(f"[prism] Frontier: added={added} removed={removed} size={len(new_frontier)}")
            frontier = new_frontier
            if self.config.human_approval:
                from agent_harness_optimizer.utils.human_approval import ask_prism_frontier

                frontier = ask_prism_frontier(frontier, gen)
            _frontier_uids = {c.uid for c in frontier}
            self._save_gen(
                gen,
                frontier,
                all_candidates,
                screen_rollouts,
                gen_stats={
                    "generation": gen,
                    "crossover": _cx_status,
                    "complementary_cases": _cx_complement,
                    "crossover_base_uid": crossover_base.uid,
                    "crossover_uid": cross.uid if cross else None,
                    "crossover_in_frontier": bool(cross and cross.uid in _frontier_uids),
                },
            )

        # Final eval bookend — fresh train+holdout score for the best candidate.
        # Matches BH/MIPROv2/GEPA pattern; provides clean stuck_breakdown in report.
        best = max(frontier, key=lambda c: (c.pass_rate, c.reliability), default=None)
        final_train_score: SplitScore | None = None
        final_holdout_score: SplitScore | None = None
        if best:
            print("[prism] final train eval…")
            print("[prism] final holdout eval…")
            final_train_score, final_holdout_score = await asyncio.gather(
                self._score(
                    best.prompt,
                    self.config.train_split,
                    self.config.output_dir / "final" / "train",
                    best.middleware_dir,
                    self.gc.train_cases,
                    "final train",
                    case_indices=_train_idx,
                ),
                self._score(
                    best.prompt,
                    self.config.holdout_split,
                    self.config.output_dir / "final" / "holdout",
                    best.middleware_dir,
                    self.gc.holdout_cases,
                    "final holdout",
                    case_indices=_holdout_idx,
                ),
            )
            print(f"[prism] final train:   {final_train_score.passed}/{final_train_score.total}")
            print(
                f"[prism] final holdout: {final_holdout_score.passed}/{final_holdout_score.total}"
            )

        final_scorecard = None
        if self.config.split_seed is not None and best:
            _sc_idx = self.config.scorecard_case_indices
            _sc_n = len(_sc_idx) if _sc_idx else "all"
            _k = self.config.num_scorecard_trials
            print(f"[prism] final scorecard eval ({_sc_n} out-of-sample cases, k={_k})…")
            final_scorecard = await self._score(
                best.prompt,
                "scorecard",
                self.config.output_dir / "final" / "scorecard",
                best.middleware_dir,
                None,
                "final scorecard",
                case_indices=_sc_idx,
                num_trials=_k,
            )
        # Load baseline_scorecard if available
        baseline_scorecard: SplitScore | None = None
        _sbd = self.config.shared_baseline_dir
        if _sbd and (_sbd / "baseline" / "scorecard.json").exists():
            baseline_scorecard = SplitScore.from_dict(
                json.loads((_sbd / "baseline" / "scorecard.json").read_text())
            )

        # Write candidate search trace
        for c in all_candidates[1:]:  # skip gen0 seed (baseline)
            search_trace.append(
                {
                    "optimizer_run_id": out.name,
                    "candidate_id": c.uid,
                    "generation": c.generation,
                    "parent_candidate_id": c.parent_uids[0] if c.parent_uids else "baseline",
                    "mutation_type": "crossover" if "cro" in c.uid else "mutation",
                    "search_surface": "prompt-only" if self.gc.prompt_only else "prompt+middleware",
                    "repair_score": c.train_passed / c.train_total if c.train_total else 0.0,
                    "gate_score": c.holdout_passed / c.holdout_total if c.holdout_total else 0.0,
                    "validity_rate": c.reliability,
                    "harness_hash": harness_hash(c.prompt, c.middleware_dir),
                    "outer_calls_this_step": 1,  # one mutate() or crossover() call per candidate
                    "outer_tokens_in": 0,
                    "outer_tokens_out": 0,
                    "eval_wall_clock_s": c.eval_wall_time_s,
                    "proposal_wall_clock_s": c.proposal_wall_time_s,
                    "accepted": any(fc.uid == c.uid for fc in frontier),
                    "selected_final": best is not None and c.uid == best.uid,
                }
            )
        if search_trace:
            trace_path = out / "candidate_search_trace.jsonl"
            with trace_path.open("w") as _tf:
                for entry in search_trace:
                    _tf.write(json.dumps(entry) + "\n")

        self._save_report(
            frontier,
            all_candidates,
            baseline_prompt,
            final_scorecard,
            baseline_scorecard=baseline_scorecard,
            final_train=final_train_score,
            final_holdout=final_holdout_score,
            screen_rollouts=screen_rollouts,
            search_outer_calls=_outer_calls_total,
            search_outer_tokens_in=_outer_tok_in_total,
            search_outer_tokens_out=_outer_tok_out_total,
        )
