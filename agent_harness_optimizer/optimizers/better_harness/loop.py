"""BetterHarnessOptimizer — benchmark-agnostic linear iterate→propose→accept loop.

Acceptance gate: pluggable AcceptanceCriterion (default: HoldoutPassRate — strict
holdout improvement; optional: HoldoutPareto — Pareto dominance on pass_rate + reliability).

Per iteration:
  1. Build workspace (task.md, asi.md, train_cases/, history/, current/, surface_manifest.json)
  2. Build failure matrix from cross-iteration history
  3. Propose one change via outer LLM (prompt_middleware_both variant), retry up to 3×
  4. Full-score candidate on train + holdout in parallel
  5. Apply acceptance gate; accept: advance current prompt; reject: keep current
"""

from __future__ import annotations

import asyncio
import difflib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent_harness_optimizer.framework.benchmark import Benchmark, SplitScore
from agent_harness_optimizer.framework.optimizer import OptimizeConfig, Optimizer
from agent_harness_optimizer.framework.report import build_report, print_report
from agent_harness_optimizer.utils.harness import harness_hash

_PROPOSER_HUMAN = (
    "Read task.md, asi.md, train_cases/failures/, train_cases/passing/, "
    "history/history.md, and history/failure_matrix.md. "
    "Propose one safe improvement. Write verdict to proposal.md. Apply to current/ only if safe."
)


def _write_diff(out: Path, baseline_prompt: str, final_prompt: str, final_mw: Path | None) -> None:
    """Write final_diff.md: unified diff of prompt + middleware changes."""
    lines = ["# Final Diff — Baseline → Best Accepted\n"]
    prompt_diff = list(
        difflib.unified_diff(
            baseline_prompt.splitlines(keepends=True),
            final_prompt.splitlines(keepends=True),
            fromfile="baseline/system_prompt.txt",
            tofile="current/system_prompt.txt",
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


def _build_failure_matrix(
    history: list[dict],
    current_score: SplitScore,
) -> dict[str, str]:
    """Compute PERSISTENT/RECURRING/NEW/FIXED labels per case_id.

    A case is:
      FIXED      — was failing in a prior accepted iteration, now passing
      NEW        — passing before but now failing (regression)
      PERSISTENT — failing in all accepted iterations including current
      RECURRING  — failing now, was passing in at least one prior accepted iteration
    """
    # Collect per-case pass history from accepted iterations
    prior_states: dict[str, list[bool]] = {}
    for h in history:
        if h.get("decision") != "accepted":
            continue
        for case in h.get("per_case", []):
            cid = case["case_id"]
            prior_states.setdefault(cid, []).append(bool(case["passed"]))

    result: dict[str, str] = {}
    for c in current_score.cases:
        cid = c.case_id
        prior = prior_states.get(cid, [])
        if not prior:
            # No prior accepted iteration to compare — label as NEW if failing
            if not c.passed:
                result[cid] = "NEW"
            continue
        ever_passed_before = any(prior)
        ever_failed_before = any(not p for p in prior)
        if c.passed and ever_failed_before:
            result[cid] = "FIXED"
        elif not c.passed and not ever_passed_before:
            result[cid] = "PERSISTENT"
        elif not c.passed and ever_passed_before:
            result[cid] = "RECURRING"
        # always-passing cases: no label needed (not a failure, not a fix)

    return result


@dataclass
class BHConfig:
    max_iterations: int = 10
    train_cases: int | None = None  # None = all cases in split
    holdout_cases: int | None = None
    seed: int = 42
    prompt_only: bool = (
        False  # when True, proposer uses prompt_only variant; middleware never touched
    )


class BetterHarnessOptimizer(Optimizer):
    """Linear iterate→propose→accept/reject loop.

    Usage::

        BetterHarnessOptimizer(benchmark, config, max_iterations=10).run()
    """

    def __init__(
        self,
        benchmark: Benchmark,
        config: OptimizeConfig,
        *,
        max_iterations: int = 10,
        bh_config: BHConfig | None = None,
    ) -> None:
        super().__init__(benchmark, config)
        self.bh = bh_config or BHConfig(max_iterations=max_iterations)

    def run(self) -> None:
        if self.config.resume:
            raise NotImplementedError(
                "BetterHarnessOptimizer does not support --resume. "
                "Delete the output directory and restart."
            )
        asyncio.run(self._run_async())

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
            print(f"[bh]   scoring {label}…")
        return await self.benchmark.score_async(
            prompt,
            split,
            out_dir,
            middleware_dir=middleware_dir,
            max_cases=max_cases,
            case_indices=case_indices,
            num_trials=num_trials,
        )

    def _build_workspace(
        self,
        ws: Path,
        prompt: str,
        middleware_dir: Path | None,
        score: SplitScore,
        history: list[dict],
        iteration: int,
        failure_matrix: dict[str, str],
        prompt_only: bool = False,
    ) -> None:
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)

        # task.md — benchmark-specific
        (ws / "task.md").write_text(self.benchmark.build_task_md(score, iteration))

        # asi.md with failure matrix labels
        (ws / "asi.md").write_text(self.benchmark.build_asi(score, failure_matrix or None))

        # per-case files
        self.benchmark.write_case_files(ws, score)

        # current/ — system prompt + middleware
        current = ws / "current"
        current.mkdir()
        (current / "system_prompt.txt").write_text(prompt)

        mw_stubs = self.benchmark.get_default_middleware_stubs()
        if mw_stubs is not None and not prompt_only:
            mw = current / "middleware"
            mw.mkdir()
            if middleware_dir and middleware_dir.is_dir():
                for f in middleware_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, mw / f.name)
            else:
                for fname, content in mw_stubs.items():
                    (mw / fname).write_text(content)

        # history/
        hist_dir = ws / "history"
        hist_dir.mkdir()

        history_lines = ["# Iteration History", ""]
        for h in history[-8:]:
            history_lines.append(
                f"- iter {h['iteration']}: decision={h['decision']} "
                f"train={h.get('train_passed', '?')}/{h.get('train_total', '?')} "
                f"holdout={h.get('holdout_passed', '?')}/{h.get('holdout_total', '?')} "
                f"reason={h.get('reason', '')}"
            )
            if h.get("proposal"):
                history_lines.append(f"  proposal: {h['proposal'][:120]}")
        (hist_dir / "history.md").write_text("\n".join(history_lines))

        # failure_matrix.md
        fm_lines = [
            "# Failure Matrix",
            "",
            "Labels: PERSISTENT=failing every iteration, RECURRING=sometimes passes,",
            "        NEW=regression this iteration, FIXED=now passing",
            "",
        ]
        for cid, label in sorted(failure_matrix.items()):
            fm_lines.append(f"- {cid}: {label}")
        (hist_dir / "failure_matrix.md").write_text("\n".join(fm_lines))

        # surface_manifest.json — tells proposer what surfaces exist
        surfaces: dict[str, str] = {"system_prompt": "current/system_prompt.txt"}
        if mw_stubs is not None and not prompt_only:
            surfaces["middleware"] = "current/middleware/"
        (ws / "surface_manifest.json").write_text(json.dumps(surfaces, indent=2))

        # proposal.md placeholder
        (ws / "proposal.md").write_text("# Proposal\n\n- Pattern:\n- Fix:\n- Verdict:\n")

    def _read_surfaces(self, ws: Path) -> tuple[str, Path | None]:
        import re

        prompt = (ws / "current" / "system_prompt.txt").read_text().strip()
        mw = ws / "current" / "middleware"
        middleware_dir = None
        if mw.is_dir():
            setup = mw / "agent_setup.py"
            if setup.exists():
                if any(
                    re.search(r"MIDDLEWARE\s*=\s*\[.+\]", line)
                    for line in setup.read_text().splitlines()
                    if "MIDDLEWARE" in line and not line.strip().startswith("#")
                ):
                    middleware_dir = mw
        return prompt, middleware_dir

    def _run_variant(
        self,
        ws_variant: Path,
        variant_name: str,
        variant_system: str,
        outer_model: str,
        max_turns: int,
    ) -> tuple[str, Path | None, str, int, int, int]:
        """Run one proposer variant in a dedicated workspace copy.

        Returns (prompt, mw_dir, proposal, outer_calls, outer_tokens_in, outer_tokens_out).
        """
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend

        from agent_harness_optimizer.utils.llm import build_model

        model = build_model(outer_model)
        backend = FilesystemBackend(root_dir=str(ws_variant), virtual_mode=True)
        agent = create_deep_agent(model=model, system_prompt=variant_system, backend=backend)
        result = agent.invoke(
            {"messages": [HumanMessage(content=_PROPOSER_HUMAN)]},
            config={"recursion_limit": max(max_turns * 3, 900)},
        )
        prompt, mw = self._read_surfaces(ws_variant)
        proposal = (
            (ws_variant / "proposal.md").read_text().strip()
            if (ws_variant / "proposal.md").exists()
            else ""
        )

        # Count outer LLM token usage from response metadata
        outer_calls = 0
        outer_tok_in = 0
        outer_tok_out = 0
        for msg in result.get("messages", []):
            usage = getattr(msg, "usage_metadata", None) or getattr(
                msg, "response_metadata", {}
            ).get("usage", {})
            if usage:
                outer_calls += 1
                outer_tok_in += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                outer_tok_out += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

        return prompt, mw, proposal, outer_calls, outer_tok_in, outer_tok_out

    async def _run_async(self) -> None:
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)

        initial_prompt = self.benchmark.default_system_prompt
        print(f"\n=== BetterHarness: {self.benchmark.name} ===")
        print(f"  inner_model: {self.config.inner_model}")
        print(f"  outer_model: {self.config.outer_model}")
        print(f"  resource_budget: {self.benchmark.resource_budget.to_dict()}")
        print(f"  max_iterations: {self.bh.max_iterations}")

        # Warm up auth on the main thread so threaded calls inherit valid credentials
        self.benchmark.build_model(self.config.outer_model)

        _cs = self.config.case_split
        _train_idx = _cs.train if _cs else None
        _holdout_idx = _cs.holdout if _cs else None
        _t0 = time.monotonic()
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
                f"[bh] shared baseline loaded: train={base_train.passed}/{base_train.total} "
                f"holdout={base_holdout.passed}/{base_holdout.total}"
            )
        else:
            print("[bh] baseline train…")
            base_train, base_holdout = await asyncio.gather(
                self._score(
                    initial_prompt,
                    self.config.train_split,
                    out / "baseline" / "train",
                    None,
                    self.bh.train_cases,
                    "baseline train",
                    case_indices=_train_idx,
                ),
                self._score(
                    initial_prompt,
                    self.config.holdout_split,
                    out / "baseline" / "holdout",
                    None,
                    self.bh.holdout_cases,
                    "baseline holdout",
                    case_indices=_holdout_idx,
                ),
            )
            print(f"[bh] baseline train: {base_train.passed}/{base_train.total}")
            print(f"[bh] baseline holdout: {base_holdout.passed}/{base_holdout.total}")
        _baseline_eval_time = round(time.monotonic() - _t0, 1)

        (out / "baseline").mkdir(parents=True, exist_ok=True)
        (out / "baseline" / "summary.json").write_text(
            json.dumps(
                {
                    "train": {
                        "passed": base_train.passed,
                        "total": base_train.total,
                        "reliability": base_train.reliability,
                        "prompt_tokens_per_case": base_train.prompt_tokens_per_case,
                        "completion_tokens_per_case": base_train.completion_tokens_per_case,
                    },
                    "holdout": {"passed": base_holdout.passed, "total": base_holdout.total},
                    "resource_budget": self.benchmark.resource_budget.to_dict(),
                    "eval_wall_time_s": _baseline_eval_time,
                },
                indent=2,
            )
        )

        current_prompt = initial_prompt
        current_mw: Path | None = None
        current_train = base_train
        current_holdout = base_holdout
        history: list[dict] = []
        search_trace: list[dict] = []
        _outer_calls_total = 0
        _outer_tok_in_total = 0
        _outer_tok_out_total = 0

        for iteration in range(1, self.bh.max_iterations + 1):
            if (
                current_train.passed == current_train.total
                and current_holdout.passed == current_holdout.total
            ):
                print(f"[bh] iter-{iteration:03d}: all cases pass — stopping early")
                break

            print(f"\n[bh] === Iteration {iteration}/{self.bh.max_iterations} ===")
            iter_dir = out / f"iter-{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Build failure matrix from prior accepted iterations
            failure_matrix = _build_failure_matrix(history, current_train)

            ws_base = iter_dir / "workspace"
            self._build_workspace(
                ws_base,
                current_prompt,
                current_mw,
                current_train,
                history,
                iteration,
                failure_matrix,
                prompt_only=self.bh.prompt_only,
            )

            # Single proposal per iteration — full-access variant (prompt + middleware)
            print(f"[bh] iter-{iteration:03d}: running proposer…")
            ws_propose = iter_dir / "ws_propose"
            if ws_propose.exists():
                shutil.rmtree(ws_propose)
            shutil.copytree(ws_base, ws_propose)

            _proposal_t0 = time.monotonic()
            new_prompt, new_mw, proposal = current_prompt, current_mw, ""
            _iter_outer_calls = 0
            _iter_outer_tok_in = 0
            _iter_outer_tok_out = 0
            last_err = None
            proposer_failed = False
            for attempt in range(3):
                if attempt:
                    wait = 2**attempt * 10
                    print(f"[bh]   proposer retry {attempt}/2 in {wait}s ({last_err})")
                    await asyncio.sleep(wait)
                try:
                    variants = self.benchmark.get_proposer_variants()
                    if self.bh.prompt_only and "prompt_only" in variants:
                        variant_key = "prompt_only"
                    elif "prompt_middleware_both" in variants:
                        variant_key = "prompt_middleware_both"
                    else:
                        variant_key = next(iter(variants))
                    (
                        new_prompt,
                        new_mw,
                        proposal,
                        _iter_outer_calls,
                        _iter_outer_tok_in,
                        _iter_outer_tok_out,
                    ) = await asyncio.to_thread(
                        self._run_variant,
                        ws_propose,
                        variant_key,
                        variants[variant_key],
                        self.config.outer_model,
                        self.config.outer_max_turns,
                    )
                    break
                except Exception as e:
                    last_err = type(e).__name__
                    if attempt == 2:
                        print(f"[bh]   Proposer failed after 3 attempts: {e}")
                        proposer_failed = True
            _proposal_wall_time = round(time.monotonic() - _proposal_t0, 1)
            _outer_calls_total += _iter_outer_calls
            _outer_tok_in_total += _iter_outer_tok_in
            _outer_tok_out_total += _iter_outer_tok_out

            if proposer_failed:
                history.append(
                    {
                        "iteration": iteration,
                        "decision": "proposer_failed",
                        "auto_decision": "proposer_failed",
                        "auto_reason": str(last_err),
                        "proposal": "",
                        "reason": "",
                    }
                )
                continue

            if new_prompt.strip() == current_prompt.strip() and new_mw == current_mw:
                print(f"[bh] iter-{iteration:03d}: no changes proposed")
                history.append(
                    {
                        "iteration": iteration,
                        "decision": "no_change",
                        "auto_decision": "no_change",
                        "auto_reason": "",
                        "proposal": proposal[:80],
                        "reason": "",
                    }
                )
                continue

            # Full score best variant (train + holdout in parallel)
            _eval_t0 = time.monotonic()
            cand_train, cand_holdout = await asyncio.gather(
                self._score(
                    new_prompt,
                    self.config.train_split,
                    iter_dir / "train",
                    new_mw,
                    self.bh.train_cases,
                    f"iter-{iteration:03d} train",
                    case_indices=_train_idx,
                ),
                self._score(
                    new_prompt,
                    self.config.holdout_split,
                    iter_dir / "holdout",
                    new_mw,
                    self.bh.holdout_cases,
                    f"iter-{iteration:03d} holdout",
                    case_indices=_holdout_idx,
                ),
            )
            _eval_wall_time = round(time.monotonic() - _eval_t0, 1)
            print(f"[bh] iter-{iteration:03d} train: {cand_train.passed}/{cand_train.total}")
            print(f"[bh] iter-{iteration:03d} holdout: {cand_holdout.passed}/{cand_holdout.total}")

            accepted, reason = self.acceptance(
                candidate_train=cand_train,
                candidate_holdout=cand_holdout,
                current_train=current_train,
                current_holdout=current_holdout,
            )
            auto_accepted, auto_reason = accepted, reason
            if self.config.human_approval:
                from agent_harness_optimizer.utils.human_approval import ask_bh_decision

                accepted, reason = ask_bh_decision(
                    auto_accepted=auto_accepted,
                    auto_reason=auto_reason,
                    proposal=proposal,
                    cand_train_passed=cand_train.passed,
                    cand_train_total=cand_train.total,
                    cand_holdout_passed=cand_holdout.passed,
                    cand_holdout_total=cand_holdout.total,
                    cand_reliability=cand_train.reliability,
                    iteration=iteration,
                )
            decision = "accepted" if accepted else "rejected"
            print(f"[bh] iter-{iteration:03d}: {decision} — {reason}")

            _cand_hash = harness_hash(new_prompt, new_mw)
            rec = {
                "iteration": iteration,
                "decision": decision,
                "reason": reason,
                "auto_decision": "accepted" if auto_accepted else "rejected",
                "auto_reason": auto_reason,
                "variant": variant_key,
                "proposal": proposal,
                "harness_hash": _cand_hash,
                "train_passed": cand_train.passed,
                "train_total": cand_train.total,
                "holdout_passed": cand_holdout.passed,
                "holdout_total": cand_holdout.total,
                "reliability": cand_train.reliability,
                "tokens_per_case": cand_train.tokens_per_case,
                "prompt_tokens_per_case": cand_train.prompt_tokens_per_case,
                "completion_tokens_per_case": cand_train.completion_tokens_per_case,
                "eval_wall_time_s": _eval_wall_time,
                "proposal_wall_time_s": _proposal_wall_time,
                "outer_calls_this_step": _iter_outer_calls,
                "outer_tokens_in": _iter_outer_tok_in,
                "outer_tokens_out": _iter_outer_tok_out,
                "stuck_breakdown": cand_train.stuck_breakdown,
                "per_case": [{"case_id": c.case_id, "passed": c.passed} for c in cand_train.cases],
            }
            history.append(rec)
            (iter_dir / "decision.json").write_text(json.dumps(rec, indent=2))
            search_trace.append(
                {
                    "optimizer_run_id": out.name,
                    "candidate_id": f"iter{iteration:03d}",
                    "generation": iteration,
                    "parent_candidate_id": f"iter{iteration - 1:03d}"
                    if iteration > 1
                    else "baseline",
                    "mutation_type": "prompt-only" if self.bh.prompt_only else "prompt+middleware",
                    "search_surface": "prompt-only" if self.bh.prompt_only else "prompt+middleware",
                    "repair_score": cand_train.pass_rate,
                    "gate_score": cand_holdout.pass_rate,
                    "validity_rate": cand_train.reliability,
                    "harness_hash": _cand_hash,
                    "outer_calls_this_step": _iter_outer_calls,
                    "outer_tokens_in": _iter_outer_tok_in,
                    "outer_tokens_out": _iter_outer_tok_out,
                    "eval_wall_clock_s": _eval_wall_time,
                    "proposal_wall_clock_s": _proposal_wall_time,
                    "accepted": accepted,
                    "selected_final": False,  # updated below
                }
            )

            if accepted:
                current_prompt = new_prompt
                current_mw = new_mw
                current_train = cand_train
                current_holdout = cand_holdout
                surfaces = out / "current"
                surfaces.mkdir(exist_ok=True)
                (surfaces / "system_prompt.txt").write_text(current_prompt)
                if current_mw and current_mw.is_dir():
                    mw_dest = surfaces / "middleware"
                    if mw_dest.exists():
                        shutil.rmtree(mw_dest)
                    shutil.copytree(current_mw, mw_dest)

        # holdout pass rate series: baseline + one entry per iteration
        holdout_series = [
            {
                "iteration": 0,
                "holdout_passed": base_holdout.passed,
                "holdout_total": base_holdout.total,
                "holdout_pass_rate": round(base_holdout.pass_rate, 4),
            }
        ]
        for h in history:
            if "holdout_passed" in h:
                holdout_series.append(
                    {
                        "iteration": h["iteration"],
                        "holdout_passed": h["holdout_passed"],
                        "holdout_total": h["holdout_total"],
                        "holdout_pass_rate": round(h["holdout_passed"] / h["holdout_total"], 4)
                        if h["holdout_total"]
                        else 0.0,
                        "decision": h.get("decision"),
                    }
                )

        # Human-readable diff: baseline prompt vs best accepted prompt
        _write_diff(out, initial_prompt, current_prompt, current_mw)

        # Final eval bookend — score best accepted prompt on train+holdout as a clean
        # post-loop measurement, matching the bookend pattern of PRISM/MIPROv2/GEPA.
        print("[bh] final train eval…")
        final_train, final_holdout = await asyncio.gather(
            self._score(
                current_prompt,
                self.config.train_split,
                out / "final" / "train",
                current_mw,
                self.bh.train_cases,
                "final train",
                case_indices=_train_idx,
            ),
            self._score(
                current_prompt,
                self.config.holdout_split,
                out / "final" / "holdout",
                current_mw,
                self.bh.holdout_cases,
                "final holdout",
                case_indices=_holdout_idx,
            ),
        )
        print(f"[bh] final train:   {final_train.passed}/{final_train.total}")
        print(f"[bh] final holdout: {final_holdout.passed}/{final_holdout.total}")

        final_scorecard = None
        if self.config.split_seed is not None:
            _sc_idx = self.config.scorecard_case_indices
            _sc_n = len(_sc_idx) if _sc_idx else "all"
            _k = self.config.num_scorecard_trials
            print(f"[bh] final scorecard eval ({_sc_n} out-of-sample cases, k={_k})…")
            final_scorecard = await self._score(
                current_prompt,
                "scorecard",
                out / "final" / "scorecard",
                current_mw,
                None,
                "final scorecard",
                case_indices=_sc_idx,
                num_trials=_k,
            )

        # optimization_rollouts = scored iterations × (train+holdout).
        # Only "accepted" and "rejected" iterations actually score cases;
        # "no_change" and "proposer_failed" both skip scoring.
        scored_iters = sum(1 for h in history if h.get("decision") in ("accepted", "rejected"))
        opt_rollouts = scored_iters * (base_train.total + base_holdout.total)

        # Load baseline_scorecard if available
        baseline_scorecard: SplitScore | None = None
        _sbd = self.config.shared_baseline_dir
        if _sbd and (_sbd / "baseline" / "scorecard.json").exists():
            baseline_scorecard = SplitScore.from_dict(
                json.loads((_sbd / "baseline" / "scorecard.json").read_text())
            )

        # Mark final accepted candidate in search trace
        _final_hash = harness_hash(current_prompt, current_mw)
        for entry in search_trace:
            if entry["harness_hash"] == _final_hash and entry["accepted"]:
                entry["selected_final"] = True
                break

        # Write candidate search trace
        if search_trace:
            trace_path = out / "candidate_search_trace.jsonl"
            with trace_path.open("w") as _tf:
                for entry in search_trace:
                    _tf.write(json.dumps(entry) + "\n")

        report = build_report(
            optimizer="bh",
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
            search_outer_calls=_outer_calls_total,
            search_outer_tokens_in=_outer_tok_in_total,
            search_outer_tokens_out=_outer_tok_out_total,
            optimizer_config={
                "max_iterations": self.bh.max_iterations,
                "train_cases": self.bh.train_cases,
                "holdout_cases": self.bh.holdout_cases,
                "seed": self.bh.seed,
                "baseline_harness_hash": harness_hash(initial_prompt, None),
                "selected_harness_hash": _final_hash,
            },
        )
        report["holdout_series"] = holdout_series
        report["iterations"] = history
        (out / "report.json").write_text(json.dumps(report, indent=2))

        print_report(report)
        print(f"  Report: {out / 'report.json'}")
