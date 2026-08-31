"""Re-score final scorecard for completed tau-telecom runs with 3 additional trials.

The original scorecard is trial 0 (already done). This script runs trials 1, 2, 3
then combines all 4 trials into pass^4 (AND rule) and updates report.json.

Usage:
    .venv/bin/python examples/rescore_telecom_passk.py --batch <N> --total-batches 6
    # batch 0-5, 30 runs each

    # or run all at once (not recommended):
    .venv/bin/python examples/rescore_telecom_passk.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TAU2_DATA_DIR", str(Path.home() / "projects" / "tau2-bench" / "data"))

TELECOM_DIRS = [
    REPO_ROOT / "experiment_runs",
    REPO_ROOT / "data/tau-telecom/emnlp-tau-telecom-raw-data-r0",
    REPO_ROOT / "data/tau-telecom/emnlp-tau-telecom-raw-data-r1r2",
]

NUM_NEW_TRIALS = 3  # trials 1, 2, 3 (trial 0 = original scorecard)


def harness_hash(prompt: str, middleware_dir=None) -> str:
    key = prompt + (str(middleware_dir) if middleware_dir else "")
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_final_prompt(run_dir: Path) -> str | None:
    """Extract the final optimized prompt from a completed run."""
    # GEPA / MIPROv2: best_prompt.txt at root
    bp = run_dir / "best_prompt.txt"
    if bp.exists():
        return bp.read_text().strip()

    # BH / PRISM: use candidate_search_trace.jsonl to find selected_final hash
    trace_file = run_dir / "candidate_search_trace.jsonl"
    if trace_file.exists():
        target_hash = None
        for line in trace_file.read_text().strip().split("\n"):
            e = json.loads(line)
            if e.get("selected_final"):
                target_hash = e.get("harness_hash")
                break

        if target_hash:
            # Search all workspace current dirs for matching hash
            for sp_file in sorted(run_dir.rglob("system_prompt.txt")):
                if "ws_propose" in str(sp_file):
                    continue
                prompt = sp_file.read_text().strip()
                if harness_hash(prompt) == target_hash:
                    return prompt

        # Fallback: last iter-N/workspace/current/system_prompt.txt
        iter_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter-")],
            key=lambda d: d.name,
            reverse=True,
        )
        for iter_dir in iter_dirs:
            sp = iter_dir / "workspace" / "current" / "system_prompt.txt"
            if sp.exists():
                return sp.read_text().strip()

    # Last resort: gen-N dirs for PRISM (find selected_final in all_candidates.json)
    for gen_dir in sorted(run_dir.iterdir(), key=lambda d: d.name, reverse=True):
        if not gen_dir.is_dir() or not gen_dir.name.startswith("gen-"):
            continue
        ac = gen_dir / "all_candidates.json"
        if not ac.exists():
            continue
        cands = json.loads(ac.read_text())
        if not isinstance(cands, list):
            continue
        for c in cands:
            prompt = c.get("prompt", "")
            if prompt:
                return prompt  # use most recent gen's first candidate as fallback

    return None


def collect_telecom_runs() -> list[Path]:
    """Collect all completed tau-telecom runs that need rescoring (k < 4)."""
    runs = []
    seen = set()
    for base in TELECOM_DIRS:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or "-baseline-" in d.name or "ablation" in d.name:
                continue
            rp = d / "report.json"
            if not rp.exists():
                continue
            r = json.loads(rp.read_text())
            bm = r.get("benchmark", "")
            if "telecom" not in bm.lower():
                continue
            if d.name in seen:
                continue
            seen.add(d.name)
            k = r.get("num_scorecard_trials", 1)
            if k >= 4:
                print(f"  SKIP (already k={k}): {d.name}")
                continue
            runs.append(d)
    return runs


async def rescore_run(run_dir: Path) -> bool:
    """Run 3 additional scorecard trials for a run and update report.json."""
    from agent_harness_optimizer.benchmarks.tau_bench import TauBenchmark
    from agent_harness_optimizer.framework.benchmark import ResourceBudget, SplitScore

    rp = run_dir / "report.json"
    report = json.loads(rp.read_text())

    prompt = get_final_prompt(run_dir)
    if prompt is None:
        print(f"  [ERROR] could not find final prompt for {run_dir.name}")
        return False

    inner_model = report["inner_model"]
    split_seed = report.get("split_seed")

    # Load scorecard case indices from the shared baseline split
    case_indices = None
    if split_seed is not None:
        # Try to load from a sibling baseline dir
        sbd_name = f"emnlp-baseline-tau-s{split_seed}-2026-05-12"
        for base in TELECOM_DIRS + [REPO_ROOT / "experiment_runs"]:
            sbd = base / sbd_name
            if not sbd.exists():
                sbd = REPO_ROOT / "experiment_runs" / sbd_name
            if sbd.exists():
                # Reconstruct split
                from tau2.run import get_tasks

                from agent_harness_optimizer.benchmarks.tau_bench import build_tau_strata
                from agent_harness_optimizer.utils.splits import make_split

                all_tasks = get_tasks("telecom", task_split_name=None)
                try:
                    base_tasks = get_tasks("telecom", task_split_name="base")
                    id_to_idx = {t.id: i for i, t in enumerate(all_tasks)}
                    pool = sorted(id_to_idx[t.id] for t in base_tasks if t.id in id_to_idx)
                    idx_to_task = {id_to_idx[t.id]: t for t in base_tasks if t.id in id_to_idx}
                    strata = build_tau_strata("telecom", pool, idx_to_task)
                except Exception:
                    pool, strata = None, None

                cs = make_split(
                    benchmark_name="tau-telecom",
                    total_cases=len(all_tasks),
                    train_cases=20,
                    holdout_cases=20,
                    seed=split_seed,
                    pool=pool,
                    strata=strata,
                )
                case_indices = cs.scorecard
                break

    budget = ResourceBudget(wall_time_s=900.0, max_steps=200, max_tokens=500_000)
    benchmark = TauBenchmark(domain="telecom", model=inner_model, budget=budget)

    sc_dir = run_dir / "final" / "scorecard"

    print(
        f"  Scoring {run_dir.name} — 3 new trials ({len(case_indices) if case_indices else 'all'} cases each)"
    )

    # Run trials 1, 2, 3 sequentially to avoid rate limit saturation
    async def _run_trial(t: int) -> SplitScore:
        trial_dir = run_dir / "final" / f"scorecard_trial{t}"
        if (trial_dir / ".done").exists():
            # Already finished — load from per-case results
            from agent_harness_optimizer.framework.benchmark import CaseScore

            cases = []
            for case_dir in trial_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                rf = case_dir / "result.json"
                if rf.exists():
                    r = json.loads(rf.read_text())
                    cases.append(
                        CaseScore(
                            case_id=case_dir.name,
                            passed=bool(r.get("passed", False)),
                            category="telecom",
                        )
                    )
            passed = sum(1 for c in cases if c.passed)
            return SplitScore(passed=passed, total=len(cases), cases=cases)
        s = await benchmark.score_async(
            prompt,
            "scorecard",
            trial_dir,
            case_indices=case_indices,
            num_trials=1,
        )
        (trial_dir / ".done").write_text("done")
        return s

    trial_scores = []
    for t in range(1, 4):
        trial_scores.append(await _run_trial(t))

    # Load original trial 0 (existing scorecard)
    t0_cases: dict[str, bool] = {}
    for case_dir in sc_dir.iterdir():
        if not case_dir.is_dir():
            continue
        rf = case_dir / "result.json"
        if rf.exists():
            r2 = json.loads(rf.read_text())
            t0_cases[case_dir.name] = bool(r2.get("passed", False))

    # AND across all 4 trials
    all_case_ids = set(t0_cases.keys())
    for s in trial_scores:
        all_case_ids &= {c.case_id for c in s.cases}

    trial_maps = [t0_cases] + [{c.case_id: c.passed for c in s.cases} for s in trial_scores]
    pass4_results: dict[str, bool] = {}
    for cid in all_case_ids:
        pass4_results[cid] = all(tm.get(cid, False) for tm in trial_maps)

    # Write pass^4 combined result.json per case into scorecard_k4/
    sc_k4_dir = run_dir / "final" / "scorecard_k4"
    sc_k4_dir.mkdir(exist_ok=True)
    for cid, passed in pass4_results.items():
        case_dir = sc_k4_dir / cid
        case_dir.mkdir(exist_ok=True)
        trial_passes = [tm.get(cid, False) for tm in trial_maps]
        (case_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_id": cid,
                    "passed": passed,
                    "num_trials": 4,
                    "trial_passes": trial_passes,
                },
                indent=2,
            )
        )

    total = len(pass4_results)
    passed4 = sum(1 for v in pass4_results.values() if v)
    pass4_rate = round(passed4 / total, 4) if total else 0.0

    # Also compute pass^1/2/3 from the 4 trials
    def pass_k_rate(k: int) -> float:
        count = sum(
            1 for cid in all_case_ids if all(trial_maps[t].get(cid, False) for t in range(k))
        )
        return round(count / len(all_case_ids), 4) if all_case_ids else 0.0

    # Update report.json
    report["num_scorecard_trials"] = 4
    if "final_scorecard" in report and report["final_scorecard"]:
        sc_block = report["final_scorecard"]
        sc_block["pass_rate_k1"] = sc_block.get("pass_rate", pass_k_rate(1))
        sc_block["pass_rate_k2"] = pass_k_rate(2)
        sc_block["pass_rate_k3"] = pass_k_rate(3)
        sc_block["pass_rate_k4"] = pass4_rate
        sc_block["passed_k4"] = passed4
        sc_block["total_k4"] = total
        # Update primary pass_rate to pass^4
        sc_block["pass_rate"] = pass4_rate
        sc_block["passed"] = passed4
        sc_block["total"] = total
        # Recompute delta vs baseline
        bl_pr = sc_block.get("baseline_pass_rate", 0.0)
        if bl_pr:
            delta = pass4_rate - bl_pr
            sc_block["delta_scorecard"] = round(delta, 4)
            sc_block["delta_scorecard_pp"] = round(delta * 100, 2)
            sc_block["improving_run"] = delta > 0.0
            sc_block["rep_success_1pp"] = delta >= 0.01
            sc_block["rep_success_2pp"] = delta >= 0.02

    rp.write_text(json.dumps(report, indent=2))
    print(
        f"  Done: {run_dir.name}  pass^4={passed4}/{total} ({pass4_rate * 100:.1f}%)  "
        f"k1={pass_k_rate(1) * 100:.1f}% k2={pass_k_rate(2) * 100:.1f}% k3={pass_k_rate(3) * 100:.1f}%"
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=None, help="Batch index (0-5)")
    parser.add_argument("--total-batches", type=int, default=6)
    parser.add_argument("--all", action="store_true", help="Run all batches sequentially")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent runs")
    args = parser.parse_args()

    runs = collect_telecom_runs()
    print(f"Total runs to rescore: {len(runs)}")

    if args.dry_run:
        for r in runs:
            print(f"  {r.name}")
        return

    if args.all:
        batches = [runs]
    elif args.batch is not None:
        batch_size = (len(runs) + args.total_batches - 1) // args.total_batches
        start = args.batch * batch_size
        batches = [runs[start : start + batch_size]]
        print(
            f"Batch {args.batch}/{args.total_batches}: runs {start}–{start + len(batches[0]) - 1} ({len(batches[0])} runs)"
        )
    else:
        parser.print_help()
        return

    for batch in batches:

        async def _run_batch():
            sem = asyncio.Semaphore(args.concurrency)

            async def _limited(r):
                async with sem:
                    return await rescore_run(r)

            results = await asyncio.gather(*[_limited(r) for r in batch], return_exceptions=True)
            ok = sum(1 for r in results if r is True)
            err = sum(1 for r in results if r is not True)
            print(f"\nBatch done: {ok} ok, {err} errors")

        asyncio.run(_run_batch())


if __name__ == "__main__":
    main()
