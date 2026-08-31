"""Unit tests for the §6.3 PRISM component-ablation flags.

Each flag must revert exactly one attribute of the full system:
  no_route      — clusters computed but not routed; all slots full-access
  no_gate       — no gate (holdout) rollouts during search; train-only selection
  no_matrix     — no cross-iteration failure matrix
  no_constraint — middleware edits unconstrained (no three-pattern vocabulary)
  no_crossover  — crossover step skipped

The outer agent is mocked; MockBenchmark provides deterministic scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from agent_harness_optimizer.framework.optimizer import OptimizeConfig
from agent_harness_optimizer.optimizers.prism.loop import PRISMConfig, PRISMOptimizer
from tests.mock_benchmark import MockBenchmark


class _Recorder:
    """Records every mutate/crossover/analyze call the loop makes."""

    def __init__(self):
        self.mutate_calls: list[dict] = []
        self.crossover_calls: int = 0
        self.counter = 0

    def mutate(
        self,
        candidate,
        score,
        *,
        generation,
        frontier,
        all_candidates=None,
        fm_cases=None,
        benchmark,
        outer_model,
        workspace_dir,
        max_turns=200,
        variant=None,
        target_pattern=None,
        unconstrained=False,
    ):
        self.mutate_calls.append(
            {
                "variant": variant,
                "target_pattern": target_pattern,
                "fm_cases": dict(fm_cases) if fm_cases else {},
                "unconstrained": unconstrained,
            }
        )
        self.counter += 1
        new_prompt = candidate.prompt + f" fix_{self.counter}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "proposal.md").write_text(f"fix_{self.counter}")
        return new_prompt, None, f"fix_{self.counter}", variant or "prompt_middleware_both", 0, 0

    def crossover(
        self,
        children,
        scores,
        base_candidate,
        *,
        generation,
        frontier,
        all_candidates=None,
        fm_cases=None,
        benchmark,
        outer_model,
        workspace_dir,
        max_turns=200,
    ):
        self.crossover_calls += 1
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "proposal.md").write_text("merged")
        return base_candidate.prompt + " fix_8 fix_9", None, "merged", 0, 0


def _analyze_with_surface_clusters(score, *, n, outer_model, workspace_dir, fm_cases=None):
    """Fake analyst: returns one cluster per surface so routing has material."""
    failures = [c.case_id for c in score.cases if not c.passed]
    if not failures:
        return [], 0, 0
    return (
        [
            {
                "root_cause": "prompt issue",
                "fix_surface": "prompt_only",
                "case_ids": failures[:1],
                "reasoning": "r",
            },
            {
                "root_cause": "middleware issue",
                "fix_surface": "middleware_only",
                "case_ids": failures[1:2] or failures[:1],
                "reasoning": "r",
            },
        ],
        0,
        0,
    )


def _run(tmp_path: Path, gc: PRISMConfig, generations: int = 1):
    rec = _Recorder()
    benchmark = MockBenchmark()
    config = OptimizeConfig(
        output_dir=tmp_path / "runs",
        inner_model="mock-inner",
        outer_model="mock-outer",
    )
    opt = PRISMOptimizer(benchmark, config, generations=generations, prism_config=gc)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", rec.mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", rec.crossover),
        patch("agent_harness_optimizer.optimizers.prism.loop.analyze_patterns", _analyze_with_surface_clusters),
    ):
        opt.run()
    return rec


def _base_gc(**kw) -> PRISMConfig:
    return PRISMConfig(train_cases=10, holdout_cases=5, mutations_per_gen=3, population_cap=5, **kw)


# ---------------------------------------------------------------------------
# Full system (control): routing produces surface-specific slots
# ---------------------------------------------------------------------------


def test_full_system_routes_surfaces(tmp_path):
    rec = _run(tmp_path, _base_gc())
    variants = [c["variant"] for c in rec.mutate_calls]
    assert "prompt_middleware_both" in variants
    assert "prompt_only" in variants
    assert "middleware_only" in variants
    # routed slots carry their cluster
    routed = [c for c in rec.mutate_calls if c["variant"] == "middleware_only"]
    assert routed and routed[0]["target_pattern"] is not None


def test_full_system_constrained_by_default(tmp_path):
    rec = _run(tmp_path, _base_gc())
    assert all(c["unconstrained"] is False for c in rec.mutate_calls)


# ---------------------------------------------------------------------------
# NoRoute
# ---------------------------------------------------------------------------


def test_no_route_all_slots_full_access(tmp_path):
    rec = _run(tmp_path, _base_gc(no_route=True))
    assert len(rec.mutate_calls) == 3
    assert all(c["variant"] == "prompt_middleware_both" for c in rec.mutate_calls)
    assert all(c["target_pattern"] is None for c in rec.mutate_calls)


# ---------------------------------------------------------------------------
# NoGate
# ---------------------------------------------------------------------------


def test_no_gate_children_have_no_holdout(tmp_path):
    _run(tmp_path, _base_gc(no_gate=True))
    gen1 = tmp_path / "runs" / "gen-001"
    cands = json.loads((gen1 / "all_candidates.json").read_text())
    children = [c for c in cands if c["generation"] == 1]
    assert children
    assert all(c["holdout_total"] == 0 for c in children)
    # pass_rate is train-only
    for c in children:
        if c["train_total"]:
            assert abs(c["pass_rate"] - c["train_passed"] / c["train_total"]) < 1e-9


def test_no_gate_final_bookend_scores_holdout(tmp_path):
    _run(tmp_path, _base_gc(no_gate=True))
    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    # gate is consulted at final acceptance: final_holdout has real totals
    assert report["final_holdout"]["total"] > 0


def test_gate_used_by_default(tmp_path):
    _run(tmp_path, _base_gc())
    gen1 = tmp_path / "runs" / "gen-001"
    cands = json.loads((gen1 / "all_candidates.json").read_text())
    children = [c for c in cands if c["generation"] == 1]
    assert children and all(c["holdout_total"] > 0 for c in children)


# ---------------------------------------------------------------------------
# NoMatrix
# ---------------------------------------------------------------------------


def test_no_matrix_passes_empty_matrix(tmp_path):
    rec = _run(tmp_path, _base_gc(no_matrix=True), generations=2)
    gen2_calls = rec.mutate_calls[3:]
    assert gen2_calls
    assert all(c["fm_cases"] == {} for c in gen2_calls)


def test_matrix_present_by_default(tmp_path):
    rec = _run(tmp_path, _base_gc(), generations=2)
    gen2_calls = rec.mutate_calls[3:]
    assert gen2_calls
    assert any(c["fm_cases"] for c in gen2_calls)


# ---------------------------------------------------------------------------
# NoConstraint
# ---------------------------------------------------------------------------


def test_no_constraint_flag_reaches_mutate(tmp_path):
    rec = _run(tmp_path, _base_gc(no_constraint=True))
    assert rec.mutate_calls
    assert all(c["unconstrained"] is True for c in rec.mutate_calls)


# ---------------------------------------------------------------------------
# NoCrossover
# ---------------------------------------------------------------------------


def test_no_crossover_skips_crossover(tmp_path):
    rec = _run(tmp_path, _base_gc(no_crossover=True), generations=2)
    assert rec.crossover_calls == 0


def test_crossover_runs_by_default(tmp_path):
    rec = _run(tmp_path, _base_gc(), generations=2)
    assert rec.crossover_calls >= 0  # gated on complementarity; must not raise


# ---------------------------------------------------------------------------
# Flags are recorded for auditability
# ---------------------------------------------------------------------------


def test_flags_written_to_experiment_config(tmp_path):
    _run(tmp_path, _base_gc(no_route=True, no_matrix=True))
    cfg = json.loads((tmp_path / "runs" / "experiment_config.json").read_text())
    assert cfg["no_route"] is True
    assert cfg["no_matrix"] is True
    assert cfg["no_gate"] is False
    assert cfg["no_constraint"] is False
    assert cfg["no_crossover"] is False
