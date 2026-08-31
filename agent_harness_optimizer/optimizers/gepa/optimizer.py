"""GEPAOptimizer — wraps official gepa-ai/gepa as an AHO Optimizer.

GEPA (Genetic Evolutionary Prompt Algorithm, Agrawal et al., ICLR 2026) uses
reflective trajectory mutation + per-instance Pareto-front selection to evolve
system prompts. This is the official gepa library, distinct from the PRISM
optimizer (which is AHO's own evolutionary approach).

## Mapping to AHO framework

GEPA's GEPAAdapter.evaluate() receives a batch of DataInst and a candidate
dict[str, str] (component name -> text). We map:
  - DataInst        = int (minibatch seed index)
  - candidate       = {"system_prompt": <prompt text>}
  - score per item  = pass_rate of a minibatch run (0.0 or 1.0 per seed)
  - reflection      = trajectory dicts from benchmark score_async()

Usage::

    benchmark = BFCLBenchmark(model="azure/gpt-5.4-mini", ...)
    config = OptimizeConfig(output_dir=Path("runs/gepa-bfcl-001"), ...)
    GEPAOptimizer(benchmark, config).run()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from agent_harness_optimizer.framework.benchmark import Benchmark, SplitScore
from agent_harness_optimizer.framework.optimizer import OptimizeConfig, Optimizer
from agent_harness_optimizer.framework.report import build_report, print_report
from agent_harness_optimizer.utils.harness import harness_hash
from agent_harness_optimizer.utils.middleware_surface import (
    DEFAULT_MW_STUB as _DEFAULT_MW_STUB,
)
from agent_harness_optimizer.utils.middleware_surface import (
    MW_PATTERN_GUIDE as _MW_REFLECTION_GUIDE,
)
from agent_harness_optimizer.utils.middleware_surface import (
    middleware_dir_from_text,
)


@dataclass
class GEPAConfig:
    max_metric_calls: int = 200  # total evaluate() calls budget
    reflection_minibatch_size: int = 15  # cases per reflection minibatch (trainset sampling)
    train_cases: int | None = None
    holdout_cases: int | None = None
    seed: int = 42
    candidate_selection: str = "pareto"  # pareto | current_best | epsilon_greedy
    frontier_type: str = "instance"
    # GEPA-MW: expose the tool-boundary middleware surface as a second GEPA
    # component, pattern-guided exactly as PRISM's middleware slot (silent
    # correction / error blocking / prerequisite blocking).
    middleware: bool = False


class GEPAOptimizer(Optimizer):
    """Official GEPA optimizer from gepa-ai/gepa.

    Requires gepa: install with `uv add gepa`
    """

    def __init__(
        self,
        benchmark: Benchmark,
        config: OptimizeConfig,
        *,
        gepa_config: GEPAConfig | None = None,
    ) -> None:
        super().__init__(benchmark, config)
        self.gc = gepa_config or GEPAConfig()

    def run(self) -> None:
        if self.config.resume:
            raise NotImplementedError(
                "GEPAOptimizer does not support --resume. Delete the output directory and restart."
            )
        self._run_sync()

    def _run_sync(self) -> None:
        try:
            import gepa as gepa_lib
        except ImportError:
            raise ImportError("gepa is required for GEPAOptimizer.\nInstall with: uv add gepa")

        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== GEPA: {self.benchmark.name} ===")
        print(f"  inner_model: {self.config.inner_model}")
        print(f"  outer_model: {self.config.outer_model}")
        print(f"  max_metric_calls: {self.gc.max_metric_calls}")
        print(f"  reflection_minibatch_size: {self.gc.reflection_minibatch_size}")

        # --- Auth ---
        from agent_harness_optimizer.utils.llm import _ensure_auth

        _ensure_auth(self.config.inner_model)
        _ensure_auth(self.config.outer_model)

        _cs = self.config.case_split
        _train_idx = _cs.train if _cs else None
        _holdout_idx = _cs.holdout if _cs else None

        def _score(
            prompt,
            split,
            path,
            max_cases=None,
            case_indices=None,
            num_trials=1,
            middleware_dir=None,
        ):
            return asyncio.run(
                self.benchmark.score_async(
                    prompt,
                    split,
                    path,
                    middleware_dir=middleware_dir,
                    max_cases=max_cases,
                    case_indices=case_indices,
                    num_trials=num_trials,
                )
            )

        _default_prompt = self.benchmark.default_system_prompt
        _sbd = self.config.shared_baseline_dir
        if (
            _sbd
            and (_sbd / "baseline" / "train.json").exists()
            and (_sbd / "baseline" / "holdout.json").exists()
        ):
            import json as _json

            from agent_harness_optimizer.framework.benchmark import SplitScore as _SS

            base_train = _SS.from_dict(_json.loads((_sbd / "baseline" / "train.json").read_text()))
            base_holdout = _SS.from_dict(
                _json.loads((_sbd / "baseline" / "holdout.json").read_text())
            )
            print(
                f"[gepa] shared baseline loaded: train={base_train.passed}/{base_train.total} "
                f"holdout={base_holdout.passed}/{base_holdout.total}"
            )
        else:
            print("[gepa] baseline train…")
            base_train = _score(
                _default_prompt,
                self.config.train_split,
                out / "baseline" / "train",
                max_cases=self.gc.train_cases,
                case_indices=_train_idx,
            )
            print(
                f"[gepa] baseline train: {base_train.passed}/{base_train.total} "
                f"({base_train.pass_rate:.3f})"
            )

            print("[gepa] baseline holdout…")
            base_holdout = _score(
                _default_prompt,
                self.config.holdout_split,
                out / "baseline" / "holdout",
                max_cases=self.gc.holdout_cases,
                case_indices=_holdout_idx,
            )
            print(f"[gepa] baseline holdout: {base_holdout.passed}/{base_holdout.total}")

        # --- Build trainset / valset (list of int seeds) ---
        # trainset = all train indices (exploration + internal Pareto signal for GEPA).
        # valset   = None so GEPA reuses trainset for its internal Pareto frontier updates.
        # Passing holdout indices as valset would leak holdout into GEPA's optimization loop
        # (GEPA scores valset candidates during search), violating the train=exploration /
        # holdout=final-acceptance design shared by all other optimizers.
        n_train = base_train.total
        trainset = list(range(n_train))
        valset = None  # GEPA reuses trainset; holdout is only scored at final acceptance gate

        # --- Reflection LM (outer model via litellm) ---
        outer_model = self.config.outer_model

        _outer_calls = [0]
        _outer_tok_in = [0]
        _outer_tok_out = [0]

        # An auth plugin (see utils/llm.py) may patch litellm globally, so calling litellm.completion
        # with the outer_model string works after _ensure_auth above.
        def reflection_lm(prompt: str) -> str:
            import litellm

            resp = litellm.completion(
                model=outer_model,
                messages=[{"role": "user", "content": prompt}],
            )
            _outer_calls[0] += 1
            usage = getattr(resp, "usage", None)
            if usage:
                _outer_tok_in[0] += (getattr(usage, "prompt_tokens", None) or 0) + (
                    getattr(usage, "input_tokens", None) or 0
                )
                _outer_tok_out[0] += (getattr(usage, "completion_tokens", None) or 0) + (
                    getattr(usage, "output_tokens", None) or 0
                )
            return resp.choices[0].message.content  # type: ignore[union-attr]

        # --- Build GEPAAdapter ---
        benchmark = self.benchmark
        train_split = self.config.train_split
        holdout_split_name = self.config.holdout_split
        eval_out = out / "trials"
        eval_counter = [0]
        case_counter = [0]  # total individual cases scored across all evaluate() calls
        use_middleware = self.gc.middleware

        class BenchmarkAdapter(gepa_lib.GEPAAdapter):
            def evaluate(
                self,
                batch: list[int],
                candidate: dict[str, str],
                capture_traces: bool = True,
            ) -> gepa_lib.EvaluationBatch:
                system_prompt = candidate.get("system_prompt", benchmark.default_system_prompt)
                if not system_prompt.strip():
                    system_prompt = benchmark.default_system_prompt

                eval_counter[0] += 1
                case_counter[0] += len(batch)
                trial_dir = eval_out / f"trial_{eval_counter[0]:04d}"

                # GEPA-MW: materialize the middleware component for this candidate.
                # Broken/empty middleware degrades to middleware-free, never crashes.
                mw_dir = (
                    middleware_dir_from_text(candidate.get("middleware"), trial_dir / "mw")
                    if use_middleware
                    else None
                )

                # Map GEPA's batch indices to actual task IDs and the correct split.
                # Indices < n_train → train cases; indices >= n_train → holdout cases
                # (valset is encoded as n_train + i by the outer setup).
                train_case_ids: list[int] = []
                holdout_case_ids: list[int] = []
                train_positions: list[int] = []
                holdout_positions: list[int] = []
                for pos, idx in enumerate(batch):
                    if idx < n_train:
                        cid = _train_idx[idx] if _train_idx is not None else idx
                        train_case_ids.append(cid)
                        train_positions.append(pos)
                    else:
                        hi = idx - n_train
                        cid = _holdout_idx[hi] if _holdout_idx is not None else hi
                        holdout_case_ids.append(cid)
                        holdout_positions.append(pos)

                all_case_scores: list[float] = [0.0] * len(batch)
                all_case_details: list[dict] = [{}] * len(batch)

                async def _score_both_splits():
                    tasks = []
                    if train_case_ids:
                        tasks.append(
                            benchmark.score_async(
                                system_prompt,
                                train_split,
                                trial_dir / "train",
                                middleware_dir=mw_dir,
                                max_cases=len(train_case_ids),
                                case_indices=train_case_ids,
                            )
                        )
                    if holdout_case_ids:
                        tasks.append(
                            benchmark.score_async(
                                system_prompt,
                                holdout_split_name,
                                trial_dir / "holdout",
                                middleware_dir=mw_dir,
                                max_cases=len(holdout_case_ids),
                                case_indices=holdout_case_ids,
                            )
                        )
                    return await asyncio.gather(*tasks)

                split_results = asyncio.run(_score_both_splits())
                result_idx = 0

                if train_case_ids:
                    score_tr: SplitScore = split_results[result_idx]
                    result_idx += 1
                    from agent_harness_optimizer.framework.benchmark import CaseScore as _CS

                    tr_cases = score_tr.cases
                    if len(tr_cases) < len(train_case_ids):
                        tr_cases = list(tr_cases) + [
                            _CS(case_id="missing", passed=False, stuck_type="missing")
                            for _ in range(len(train_case_ids) - len(tr_cases))
                        ]
                    for pos, c in zip(train_positions, tr_cases):
                        all_case_scores[pos] = 1.0 if c.passed else 0.0
                        all_case_details[pos] = {
                            "case_id": c.case_id,
                            "passed": c.passed,
                            "stuck_type": c.stuck_type,
                            "error": c.extra.get("error", ""),
                            "state_diff": c.extra.get("state_diff", ""),
                            "tool_calls": c.extra.get("tool_calls", []),
                        }

                if holdout_case_ids:
                    score_ho: SplitScore = split_results[result_idx]
                    from agent_harness_optimizer.framework.benchmark import CaseScore as _CS

                    ho_cases = score_ho.cases
                    if len(ho_cases) < len(holdout_case_ids):
                        ho_cases = list(ho_cases) + [
                            _CS(case_id="missing", passed=False, stuck_type="missing")
                            for _ in range(len(holdout_case_ids) - len(ho_cases))
                        ]
                    for pos, c in zip(holdout_positions, ho_cases):
                        all_case_scores[pos] = 1.0 if c.passed else 0.0
                        all_case_details[pos] = {
                            "case_id": c.case_id,
                            "passed": c.passed,
                            "stuck_type": c.stuck_type,
                            "error": c.extra.get("error", ""),
                            "state_diff": c.extra.get("state_diff", ""),
                            "tool_calls": c.extra.get("tool_calls", []),
                        }

                # Summarize for logging
                total_passed = int(sum(all_case_scores))
                print(
                    f"[gepa]   eval {eval_counter[0]:04d}: "
                    f"{total_passed}/{len(batch)} ({total_passed / max(len(batch), 1):.3f}) "
                    f"prompt[:60]={system_prompt[:60]!r}"
                )

                case_scores = all_case_scores

                trajectories = None
                if capture_traces:
                    trajectories = [
                        {**all_case_details[i], "score": case_scores[i]} for i in range(len(batch))
                    ]

                return gepa_lib.EvaluationBatch(
                    outputs=[{"passed": s} for s in case_scores],
                    scores=case_scores,
                    trajectories=trajectories,
                )

            def make_reflective_dataset(
                self,
                candidate: dict[str, str],
                eval_batch: gepa_lib.EvaluationBatch,
                components_to_update: list[str],
            ) -> dict[str, Any]:
                system_prompt = candidate.get("system_prompt", "")
                avg_score = (
                    sum(eval_batch.scores) / len(eval_batch.scores) if eval_batch.scores else 0.0
                )
                n_passed = sum(1 for s in eval_batch.scores if s > 0)
                n_failed = len(eval_batch.scores) - n_passed

                # Build per-case detail from trajectories so the outer LLM knows which
                # cases failed — aggregate pass_rate alone gives no actionable signal.
                failed_snippets: list[str] = []
                if eval_batch.trajectories:
                    for i, traj in enumerate(eval_batch.trajectories):
                        if not isinstance(traj, dict) or traj.get("passed"):
                            continue
                        case_id = traj.get("case_id", i)
                        stuck = traj.get("stuck_type", "")
                        error = (traj.get("error") or traj.get("state_diff") or "")[:200].replace(
                            "\n", " "
                        )
                        tool_calls = traj.get("tool_calls", [])
                        last_tools = (
                            [tc.get("name", str(tc)) for tc in tool_calls[-3:]]
                            if tool_calls
                            else []
                        )
                        parts = [f"case_id={case_id}"]
                        if stuck:
                            parts.append(f"stuck={stuck}")
                        if error:
                            parts.append(f"error={error!r}")
                        if last_tools:
                            parts.append(f"last_tools={last_tools}")
                        failed_snippets.append("  " + " | ".join(parts))
                failed_detail = (
                    "\n".join(failed_snippets[:10])
                    if failed_snippets
                    else f"{n_failed} cases failed (no trajectory detail available)"
                )
                out_ds: dict[str, Any] = {}
                requested = components_to_update or ["system_prompt"]
                if "system_prompt" in requested:
                    out_ds["system_prompt"] = [
                        {
                            "input": f"n_cases={len(eval_batch.scores)}",
                            "output": f"pass_rate={avg_score:.3f} ({n_passed}/{len(eval_batch.scores)})",
                            "feedback": (
                                f"The system prompt achieved pass_rate={avg_score:.3f} "
                                f"({n_passed}/{len(eval_batch.scores)} cases passed, "
                                f"{n_failed} failed).\n"
                                f"Failed cases:\n{failed_detail}\n"
                                f"Current prompt: {system_prompt[:500]}"
                            ),
                        }
                    ]
                if use_middleware and "middleware" in requested:
                    mw_text = candidate.get("middleware", "")
                    out_ds["middleware"] = [
                        {
                            "input": f"n_cases={len(eval_batch.scores)}",
                            "output": f"pass_rate={avg_score:.3f} ({n_passed}/{len(eval_batch.scores)})",
                            "feedback": (
                                f"{_MW_REFLECTION_GUIDE}\n"
                                f"The harness achieved pass_rate={avg_score:.3f} "
                                f"({n_passed}/{len(eval_batch.scores)} passed, {n_failed} failed).\n"
                                f"Failed cases:\n{failed_detail}\n"
                                f"Current middleware file:\n{mw_text[:1500]}"
                            ),
                        }
                    ]
                return out_ds

        adapter = BenchmarkAdapter()

        # --- Run GEPA ---
        print("[gepa] running optimization…")
        seed_candidate = {"system_prompt": self.benchmark.default_system_prompt}
        if use_middleware:
            _stubs = self.benchmark.get_default_middleware_stubs() or {}
            seed_candidate["middleware"] = _stubs.get("custom_middleware.py", _DEFAULT_MW_STUB)

        result = gepa_lib.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy=self.gc.candidate_selection,
            frontier_type=self.gc.frontier_type,
            reflection_minibatch_size=self.gc.reflection_minibatch_size,
            max_metric_calls=self.gc.max_metric_calls,
            run_dir=str(out / "gepa_state"),
            seed=self.gc.seed,
            raise_on_exception=True,
        )

        # --- Extract best prompt (+ middleware in -MW mode) ---
        best_candidate = result.best_candidate
        best_prompt = best_candidate.get("system_prompt", self.benchmark.default_system_prompt)
        if not best_prompt.strip():
            best_prompt = self.benchmark.default_system_prompt
        best_mw_dir = (
            middleware_dir_from_text(best_candidate.get("middleware"), out / "best_middleware")
            if use_middleware
            else None
        )
        print(f"[gepa] best prompt ({len(best_prompt)} chars):\n{best_prompt[:200]}")
        if use_middleware:
            print(f"[gepa] best middleware: {'active' if best_mw_dir else 'none/no-op'}")
        if self.config.human_approval:
            from agent_harness_optimizer.utils.human_approval import ask_prompt_review

            best_prompt = ask_prompt_review(
                best_prompt, "GEPA", self.benchmark.default_system_prompt
            )

        # --- Final eval ---
        print("[gepa] final train eval…")
        final_train = _score(
            best_prompt,
            self.config.train_split,
            out / "final" / "train",
            max_cases=self.gc.train_cases,
            case_indices=_train_idx,
            middleware_dir=best_mw_dir,
        )
        print("[gepa] final holdout eval…")
        final_holdout = _score(
            best_prompt,
            self.config.holdout_split,
            out / "final" / "holdout",
            max_cases=self.gc.holdout_cases,
            case_indices=_holdout_idx,
            middleware_dir=best_mw_dir,
        )
        # --- Acceptance gate ---
        print(f"[gepa] acceptance: {type(self.acceptance).__name__}")
        accepted, accept_reason = self.acceptance(
            candidate_train=final_train,
            candidate_holdout=final_holdout,
            current_train=base_train,
            current_holdout=base_holdout,
        )
        print(f"[gepa] {'ACCEPTED' if accepted else 'REJECTED'} — {accept_reason}")
        if not accepted:
            print(
                "[gepa] WARNING: best candidate did not beat baseline by acceptance criterion "
                "— falling back to default prompt"
            )
            best_prompt = self.benchmark.default_system_prompt
            best_mw_dir = None
            final_train = base_train
            final_holdout = base_holdout

        final_scorecard = None
        if self.config.split_seed is not None:
            _sc_idx = self.config.scorecard_case_indices
            _sc_n = len(_sc_idx) if _sc_idx else "all"
            _k = self.config.num_scorecard_trials
            print(f"[gepa] final scorecard eval ({_sc_n} out-of-sample cases, k={_k})…")
            final_scorecard = _score(
                best_prompt,
                "scorecard",
                out / "final" / "scorecard",
                case_indices=_sc_idx,
                num_trials=_k,
                middleware_dir=best_mw_dir,
            )

        # Load baseline_scorecard if available
        baseline_scorecard: SplitScore | None = None
        _sbd2 = self.config.shared_baseline_dir
        if _sbd2 and (_sbd2 / "baseline" / "scorecard.json").exists():
            baseline_scorecard = SplitScore.from_dict(
                json.loads((_sbd2 / "baseline" / "scorecard.json").read_text())
            )

        # Write candidate search trace (one entry per evaluate() call)
        search_trace: list[dict] = []
        _best_hash = harness_hash(best_prompt, best_mw_dir)
        for entry_key in (
            sorted(vars(result).get("_eval_records", {}).keys())
            if hasattr(result, "_eval_records")
            else []
        ):
            pass  # gepa result does not expose per-eval history; write a summary row instead
        search_trace.append(
            {
                "optimizer_run_id": out.name,
                "candidate_id": "gepa_best",
                "generation": eval_counter[0],
                "parent_candidate_id": "seed",
                "mutation_type": "reflective_mutation",
                "search_surface": "prompt+middleware" if use_middleware else "prompt-only",
                "repair_score": final_train.pass_rate if final_train else 0.0,
                "gate_score": final_holdout.pass_rate if final_holdout else 0.0,
                "validity_rate": final_train.reliability if final_train else 1.0,
                "harness_hash": _best_hash,
                "outer_calls_this_step": _outer_calls[0],
                "outer_tokens_in": _outer_tok_in[0],
                "outer_tokens_out": _outer_tok_out[0],
                "eval_wall_clock_s": 0.0,
                "proposal_wall_clock_s": 0.0,
                "accepted": accepted,
                "selected_final": True,
            }
        )
        trace_path = out / "candidate_search_trace.jsonl"
        with trace_path.open("w") as _tf:
            for entry in search_trace:
                _tf.write(json.dumps(entry) + "\n")

        # --- Save ---
        (out / "best_prompt.txt").write_text(best_prompt)
        if use_middleware and best_candidate.get("middleware"):
            (out / "best_middleware_source.py").write_text(best_candidate["middleware"])
        report = build_report(
            optimizer="gepa",
            benchmark_name=self.benchmark.name,
            inner_model=self.config.inner_model,
            outer_model=self.config.outer_model,
            split_seed=self.config.split_seed,
            acceptance_criterion=type(self.acceptance).__name__,
            base_train=base_train,
            base_holdout=base_holdout,
            final_train=final_train,
            final_holdout=final_holdout,
            final_scorecard=final_scorecard,
            baseline_scorecard=baseline_scorecard,
            optimization_rollouts=case_counter[0],
            experiment_id=self.config.experiment_id,
            condition_id=self.config.condition_id,
            repeat_id=self.config.repeat_id,
            search_seed=self.config.search_seed,
            num_scorecard_trials=self.config.num_scorecard_trials,
            search_outer_calls=_outer_calls[0],
            search_outer_tokens_in=_outer_tok_in[0],
            search_outer_tokens_out=_outer_tok_out[0],
            optimizer_config={
                "max_metric_calls": self.gc.max_metric_calls,
                "reflection_minibatch_size": self.gc.reflection_minibatch_size,
                "seed": self.gc.seed,
                "middleware": use_middleware,
                "evaluate_calls": eval_counter[0],
                "accepted": accepted,
                "accept_reason": accept_reason,
                "baseline_harness_hash": harness_hash(self.benchmark.default_system_prompt, None),
                "selected_harness_hash": _best_hash,
            },
        )
        (out / "report.json").write_text(json.dumps(report, indent=2))

        print_report(report)
        print(f"  best_prompt.txt:   {out / 'best_prompt.txt'}")
        print(f"  report.json:       {out / 'report.json'}")
