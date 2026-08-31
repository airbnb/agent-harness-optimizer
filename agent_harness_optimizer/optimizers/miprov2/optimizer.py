"""MIPROv2Optimizer — wraps DSPy MIPROv2 as an AHO Optimizer.

DSPy MIPROv2 uses Bayesian optimization (TPE via Optuna) to jointly search
over instruction candidates + few-shot demos. It is the key baseline that
GEPA (Agrawal et al., ICLR 2026) explicitly benchmarks against (+10% claimed).

## Mapping to AHO framework

MIPROv2 is inherently example-level: it calls module.forward(example) for each
training example and collects per-example metric scores. Our benchmark is
batch-level: score_async() runs N cases and returns aggregate pass_rate.

We bridge this by treating each train case as one DSPy example:
  - trainset  = list of N train case indices (one per case)
  - forward() = run benchmark on ONE case, return 1.0 if passed else 0.0
  - metric()  = identity (forward already returns the score)
  - DSPy minibatch_size controls how many cases are sampled per trial

MIPROv2 optimizes the Signature.instructions field, which we map 1:1 to the
agent system prompt fed into benchmark.score_async().

Usage::

    benchmark = BFCLBenchmark(resource_budget=ResourceBudget(wall_time_s=120))
    config = OptimizeConfig(
        output_dir=Path("runs/bfcl-miprov2-001"),
        inner_model="azure/gpt-4.1",
        outer_model="azure/gpt-4.1",
    )
    MIPROv2Optimizer(benchmark, config, num_candidates=10, num_trials=20).run()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from dataclasses import dataclass
from typing import Any

from agent_harness_optimizer.framework.benchmark import Benchmark, SplitScore
from agent_harness_optimizer.framework.optimizer import OptimizeConfig, Optimizer
from agent_harness_optimizer.framework.report import build_report, print_report
from agent_harness_optimizer.utils.harness import harness_hash
from agent_harness_optimizer.utils.middleware_surface import (
    DEFAULT_MW_STUB,
    MW_PATTERN_GUIDE,
    middleware_dir_from_text,
)


def _run_coro_sync(coro):
    """Run a coroutine synchronously from any context.

    asyncio.run() raises RuntimeError if called from inside a running event loop
    (e.g., DSPy bootstrapping calls forward() from within async infra).
    In that case, spin up a fresh thread which has no event loop.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


@dataclass
class MIPROv2Config:
    num_candidates: int = 10  # instruction candidates to generate
    num_trials: int = 20  # Bayesian optimization trials
    max_bootstrapped_demos: int = 0  # keep 0 — demos don't apply to system-prompt optimization
    max_labeled_demos: int = 0
    train_cases: int | None = None  # None = all
    holdout_cases: int | None = None
    minibatch_size: int = 25  # cases per trial evaluation
    minibatch_full_eval_steps: int = 10
    num_threads: int = 4  # DSPy parallel trial threads (1 = sequential, original default)
    seed: int = 42
    verbose: bool = True
    # MIPROv2-MW: expose the tool-boundary middleware surface as a second
    # predictor whose instructions hold the custom_middleware.py source,
    # pattern-guided exactly as PRISM's middleware slot.
    middleware: bool = False


class MIPROv2Optimizer(Optimizer):
    """DSPy MIPROv2 as an AHO optimizer.

    Requires dspy-ai: install with `uv sync --extra miprov2`
    """

    def __init__(
        self,
        benchmark: Benchmark,
        config: OptimizeConfig,
        *,
        mipro_config: MIPROv2Config | None = None,
        num_candidates: int = 10,
        num_trials: int = 20,
    ) -> None:
        super().__init__(benchmark, config)
        self.mc = mipro_config or MIPROv2Config(
            num_candidates=num_candidates,
            num_trials=num_trials,
        )

    def run(self) -> None:
        if self.config.resume:
            raise NotImplementedError(
                "MIPROv2Optimizer does not support --resume. "
                "Delete the output directory and restart."
            )
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "dspy-ai is required for MIPROv2Optimizer.\nInstall with: uv sync --extra miprov2"
            )

        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== MIPROv2: {self.benchmark.name} ===")
        print(f"  inner_model: {self.config.inner_model}")
        print(f"  outer_model: {self.config.outer_model}")
        print(f"  num_candidates: {self.mc.num_candidates}")
        print(f"  num_trials: {self.mc.num_trials}")
        print(f"  minibatch_size: {self.mc.minibatch_size}")
        print(f"  acceptance: {type(self.acceptance).__name__}")

        # --- Configure DSPy LMs ---
        # Auth (utils/llm.py _ensure_auth) must be initialized before dspy.LM() so that
        # litellm routes through LFH with proper IAT auth.
        from agent_harness_optimizer.utils.llm import _ensure_auth

        _ensure_auth(self.config.inner_model)
        _ensure_auth(self.config.outer_model)
        task_lm = dspy.LM(model=self.config.inner_model)
        prompt_lm = dspy.LM(model=self.config.outer_model)
        dspy.configure(lm=task_lm)

        _default_prompt = self.benchmark.default_system_prompt
        _cs = self.config.case_split
        _train_idx = _cs.train if _cs else None
        _holdout_idx = _cs.holdout if _cs else None
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
                f"[miprov2] shared baseline loaded: train={base_train.passed}/{base_train.total} "
                f"holdout={base_holdout.passed}/{base_holdout.total}"
            )
        else:
            print("[miprov2] baseline train…")
            base_train = await self.benchmark.score_async(
                _default_prompt,
                self.config.train_split,
                out / "baseline" / "train",
                max_cases=self.mc.train_cases,
                case_indices=_train_idx,
            )
            print(
                f"[miprov2] baseline: {base_train.passed}/{base_train.total}  "
                f"reliability={base_train.reliability:.3f}"
            )

            print("[miprov2] baseline holdout…")
            base_holdout = await self.benchmark.score_async(
                _default_prompt,
                self.config.holdout_split,
                out / "baseline" / "holdout",
                max_cases=self.mc.holdout_cases,
                case_indices=_holdout_idx,
            )
            print(f"[miprov2] baseline holdout: {base_holdout.passed}/{base_holdout.total}")

        # --- Build DSPy module ---
        eval_counter = [0]
        benchmark = self.benchmark
        train_split = self.config.train_split
        eval_out = out / "trials"

        # Pool layout: seeds 0..n_train-1 → train cases only.
        # forward() is called with batch_seed from trainset (first 80% of train) and valset
        # (last 20% of train). All seeds map into train_id_pool to avoid touching holdout
        # during DSPy's internal optimization loop.
        n_train = base_train.total

        train_id_pool = list(_train_idx) if _train_idx is not None else list(range(n_train))
        case_id_pool = train_id_pool  # DSPy only ever scores train cases internally

        def _split_for_seed(seed: int) -> str:
            return train_split  # always train; holdout only used at final acceptance gate

        use_middleware = self.mc.middleware

        class AgentOptimizeSignature(dspy.Signature):
            __doc__ = self.benchmark.default_system_prompt
            batch_seed: int = dspy.InputField(desc="case pool index")
            pass_rate: float = dspy.OutputField(desc="1.0 if case passed else 0.0")

        if use_middleware:
            _stubs = self.benchmark.get_default_middleware_stubs() or {}
            _seed_mw = _stubs.get("custom_middleware.py", DEFAULT_MW_STUB)
            # The pattern constraint rides along as Python comments so the seed
            # instructions (and echoes of them) stay compilable middleware.
            _mw_doc = "".join(f"# {line}\n" for line in MW_PATTERN_GUIDE.splitlines()) + _seed_mw

            class MiddlewareSignature(dspy.Signature):
                __doc__ = _mw_doc
                batch_seed: int = dspy.InputField(desc="case pool index")
                pass_rate: float = dspy.OutputField(desc="1.0 if case passed else 0.0")

        class BenchmarkModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict(AgentOptimizeSignature)
                if use_middleware:
                    # Second optimized surface: MIPROv2 proposes new instruction
                    # text for this predictor = new custom_middleware.py content.
                    self.middleware = dspy.Predict(MiddlewareSignature)

            def forward(self, batch_seed: int = 0) -> dspy.Prediction:
                system_prompt = self.predict.signature.instructions
                if not system_prompt or not system_prompt.strip():
                    system_prompt = benchmark.default_system_prompt

                eval_counter[0] += 1
                trial_num = eval_counter[0]
                trial_dir = eval_out / f"trial_{trial_num:04d}"

                mw_dir = (
                    middleware_dir_from_text(
                        self.middleware.signature.instructions, trial_dir / "mw"
                    )
                    if use_middleware
                    else None
                )

                idx = batch_seed % len(case_id_pool)
                case_id = case_id_pool[idx]
                split = _split_for_seed(idx)

                score: SplitScore = _run_coro_sync(
                    benchmark.score_async(
                        system_prompt,
                        split,
                        trial_dir,
                        middleware_dir=mw_dir,
                        max_cases=1,
                        case_indices=[case_id],
                    )
                )
                print(
                    f"[miprov2]   call {trial_num:04d}: split={split} case={case_id} "
                    f"passed={score.passed}/{score.total} "
                    f"prompt[:60]={system_prompt[:60]!r}"
                )
                return dspy.Prediction(pass_rate=score.pass_rate)

        module = BenchmarkModule()

        # trainset = first 80% of train cases (exploration + DSPy minibatch sampling).
        # valset   = last 20% of train cases (DSPy internal full-eval checkpoints + best-trial selection).
        # Using a train subset as valset avoids leaking true holdout into DSPy's internal
        # candidate selection loop. True holdout is only scored at the final acceptance gate,
        # consistent with BH/PRISM/GEPA design.
        split_idx = max(1, int(n_train * 0.8))
        trainset = [dspy.Example(batch_seed=i).with_inputs("batch_seed") for i in range(split_idx)]
        valset = [
            dspy.Example(batch_seed=i).with_inputs("batch_seed") for i in range(split_idx, n_train)
        ]

        def metric(example: Any, pred: Any, trace: Any = None) -> float:
            return float(getattr(pred, "pass_rate", 0.0))

        # --- Run MIPROv2 ---
        teleprompter = dspy.MIPROv2(
            metric=metric,
            prompt_model=prompt_lm,
            task_model=task_lm,
            num_candidates=self.mc.num_candidates,
            auto=None,
            max_bootstrapped_demos=self.mc.max_bootstrapped_demos,
            max_labeled_demos=self.mc.max_labeled_demos,
            init_temperature=1.0,
            num_threads=self.mc.num_threads,
            verbose=self.mc.verbose,
            seed=self.mc.seed,
            max_errors=10,
        )

        print("[miprov2] running optimization…")
        # trainset = train[:80%], valset = train[80%:] — holdout never touched internally.
        compiled = teleprompter.compile(
            module,
            trainset=trainset,
            valset=valset,
            num_trials=self.mc.num_trials,
            minibatch=True,
            minibatch_size=min(self.mc.minibatch_size, len(trainset), len(valset)),
            minibatch_full_eval_steps=self.mc.minibatch_full_eval_steps,
        )

        # --- Extract best prompt ---
        # DSPy selects the trial with highest mean valset score (argmax over train[80%]).
        # True holdout acceptance is applied post-hoc at the final gate.
        best_prompt = compiled.predict.signature.instructions
        if not best_prompt or not best_prompt.strip():
            best_prompt = self.benchmark.default_system_prompt
        best_mw_text = (
            compiled.middleware.signature.instructions
            if use_middleware and hasattr(compiled, "middleware")
            else None
        )
        best_mw_dir = (
            middleware_dir_from_text(best_mw_text, out / "best_middleware")
            if use_middleware
            else None
        )
        print(f"[miprov2] best prompt ({len(best_prompt)} chars):\n{best_prompt[:200]}")
        if use_middleware:
            print(f"[miprov2] best middleware: {'active' if best_mw_dir else 'none/no-op'}")
        print(f"[miprov2] acceptance: {type(self.acceptance).__name__}")
        if self.config.human_approval:
            from agent_harness_optimizer.utils.human_approval import ask_prompt_review

            best_prompt = ask_prompt_review(
                best_prompt, "MIPROv2", self.benchmark.default_system_prompt
            )

        # --- Final eval ---
        print("[miprov2] final train eval…")
        final_train = await self.benchmark.score_async(
            best_prompt,
            self.config.train_split,
            out / "final" / "train",
            middleware_dir=best_mw_dir,
            max_cases=self.mc.train_cases,
            case_indices=_train_idx,
        )
        print("[miprov2] final holdout eval…")
        final_holdout = await self.benchmark.score_async(
            best_prompt,
            self.config.holdout_split,
            out / "final" / "holdout",
            middleware_dir=best_mw_dir,
            max_cases=self.mc.holdout_cases,
            case_indices=_holdout_idx,
        )

        # --- Acceptance gate (post-hoc, for reporting consistency) ---
        accepted, accept_reason = self.acceptance(
            candidate_train=final_train,
            candidate_holdout=final_holdout,
            current_train=base_train,
            current_holdout=base_holdout,
        )
        print(
            f"[miprov2] acceptance ({type(self.acceptance).__name__}): "
            f"{'ACCEPTED' if accepted else 'REJECTED'} — {accept_reason}"
        )
        if not accepted:
            print(
                "[miprov2] WARNING: best compiled prompt did not beat baseline "
                "by acceptance criterion — falling back to default prompt"
            )
            best_prompt = self.benchmark.default_system_prompt
            best_mw_dir = None
            final_train = base_train
            final_holdout = base_holdout

        # Scorecard uses post-gate best_prompt so it reflects what is actually shipped.
        final_scorecard = None
        if self.config.split_seed is not None:
            _sc_idx = self.config.scorecard_case_indices
            _sc_n = len(_sc_idx) if _sc_idx else "all"
            _k = self.config.num_scorecard_trials
            print(f"[miprov2] final scorecard eval ({_sc_n} out-of-sample cases, k={_k})…")
            final_scorecard = await self.benchmark.score_async(
                best_prompt,
                "scorecard",
                out / "final" / "scorecard",
                middleware_dir=best_mw_dir,
                case_indices=_sc_idx,
                num_trials=_k,
            )

        # --- Outer token counts via DSPy LM history ---
        # history entries are dicts: {"usage": {"prompt_tokens": N, "completion_tokens": N, ...}, ...}
        _history = prompt_lm.history if hasattr(prompt_lm, "history") else []
        _outer_calls = len(_history) or self.mc.num_candidates
        _outer_tok_in = sum(
            (h.get("usage", {}) if isinstance(h, dict) else {}).get("prompt_tokens", 0)
            for h in _history
        )
        _outer_tok_out = sum(
            (h.get("usage", {}) if isinstance(h, dict) else {}).get("completion_tokens", 0)
            for h in _history
        )

        # Load baseline_scorecard if available
        baseline_scorecard: SplitScore | None = None
        _sbd = self.config.shared_baseline_dir
        if _sbd and (_sbd / "baseline" / "scorecard.json").exists():
            baseline_scorecard = SplitScore.from_dict(
                json.loads((_sbd / "baseline" / "scorecard.json").read_text())
            )

        # Write candidate search trace (one row per trial from compiled state)
        search_trace: list[dict] = []
        _best_hash = harness_hash(best_prompt, best_mw_dir)
        try:
            history = getattr(prompt_lm, "history", [])
            for i, h in enumerate(history):
                search_trace.append(
                    {
                        "optimizer_run_id": out.name,
                        "candidate_id": f"trial_{i:04d}",
                        "generation": i,
                        "parent_candidate_id": "bootstrap",
                        "mutation_type": "bayesian",
                        "search_surface": "prompt+middleware" if use_middleware else "prompt-only",
                        "repair_score": 0.0,
                        "gate_score": 0.0,
                        "validity_rate": 1.0,
                        "harness_hash": "",
                        "outer_calls_this_step": 1,
                        "outer_tokens_in": (getattr(h, "usage", None) or {}).get(
                            "prompt_tokens", 0
                        ),
                        "outer_tokens_out": (getattr(h, "usage", None) or {}).get(
                            "completion_tokens", 0
                        ),
                        "eval_wall_clock_s": 0.0,
                        "proposal_wall_clock_s": 0.0,
                        "accepted": False,
                        "selected_final": False,
                    }
                )
        except Exception:
            pass
        if search_trace:
            search_trace[-1]["selected_final"] = True
            trace_path = out / "candidate_search_trace.jsonl"
            with trace_path.open("w") as _tf:
                for entry in search_trace:
                    _tf.write(json.dumps(entry) + "\n")

        # --- Save ---
        (out / "best_prompt.txt").write_text(best_prompt)
        if use_middleware and best_mw_text:
            (out / "best_middleware_source.py").write_text(best_mw_text)
        opt_rollouts = eval_counter[0]
        report = build_report(
            optimizer="miprov2",
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
            optimization_rollouts=opt_rollouts,
            experiment_id=self.config.experiment_id,
            condition_id=self.config.condition_id,
            repeat_id=self.config.repeat_id,
            search_seed=self.config.search_seed,
            num_scorecard_trials=self.config.num_scorecard_trials,
            search_outer_calls=_outer_calls,
            search_outer_tokens_in=_outer_tok_in,
            search_outer_tokens_out=_outer_tok_out,
            optimizer_config={
                "num_candidates": self.mc.num_candidates,
                "num_trials": self.mc.num_trials,
                "minibatch_size": self.mc.minibatch_size,
                "minibatch_full_eval_steps": self.mc.minibatch_full_eval_steps,
                "seed": self.mc.seed,
                "middleware": use_middleware,
                "accepted": accepted,
                "accept_reason": accept_reason,
                "baseline_harness_hash": harness_hash(self.benchmark.default_system_prompt, None),
                "selected_harness_hash": _best_hash,
                "outer_calls_estimated": not hasattr(prompt_lm, "history"),
            },
        )
        (out / "report.json").write_text(json.dumps(report, indent=2))

        print_report(report)
        print(f"  best_prompt.txt:   {out / 'best_prompt.txt'}")
        print(f"  report.json:       {out / 'report.json'}")
