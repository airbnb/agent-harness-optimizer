"""Integration test for BetterHarnessOptimizer using MockBenchmark.

The outer agent is replaced with a mock _run_variant that deterministically
adds "fix_N" keywords to the prompt each iteration, so we can test the
full loop (iterations, accept/reject gate, failure matrix, workspace layout)
without any real LLM calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from agent_harness_optimizer.framework.optimizer import OptimizeConfig
from agent_harness_optimizer.optimizers.better_harness.loop import BetterHarnessOptimizer
from tests.mock_benchmark import MockBenchmark


def _make_optimizer(tmp_path: Path, max_iterations: int = 5) -> BetterHarnessOptimizer:
    benchmark = MockBenchmark()
    config = OptimizeConfig(
        output_dir=tmp_path / "runs",
        inner_model="mock-inner",
        outer_model="mock-outer",
    )
    return BetterHarnessOptimizer(benchmark, config, max_iterations=max_iterations)


def _mock_run_variant(iteration_counter: list[int]):
    """Each call adds 'fix_<N>' for the next unfixed case to the prompt."""

    def _run_variant(self, ws_variant, variant_name, variant_system, outer_model, max_turns):
        n = iteration_counter[0]
        iteration_counter[0] += 1
        current = (ws_variant / "current" / "system_prompt.txt").read_text()
        new_prompt = current + f" fix_{n}"
        (ws_variant / "current" / "system_prompt.txt").write_text(new_prompt)
        (ws_variant / "proposal.md").write_text(f"# Proposal\n\nfix case_{n}")
        return new_prompt, None, f"fix case_{n}", 1, 100, 50

    return _run_variant


def test_loop_runs_and_produces_report(tmp_path):
    opt = _make_optimizer(tmp_path, max_iterations=4)
    counter = [0]
    with patch.object(type(opt), "_run_variant", _mock_run_variant(counter)):
        opt.run()

    report_path = tmp_path / "runs" / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["benchmark"] == "mock"
    assert "baseline_train" in report
    assert "final_train" in report
    assert "iterations" in report
    assert len(report["iterations"]) > 0


def test_accepted_iterations_improve_score(tmp_path):
    opt = _make_optimizer(tmp_path, max_iterations=5)
    counter = [0]
    with patch.object(type(opt), "_run_variant", _mock_run_variant(counter)):
        opt.run()

    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    baseline = report["baseline_train"]["passed"]
    final = report["final_train"]["passed"]
    assert final >= baseline


def test_current_surfaces_written_on_accept(tmp_path):
    opt = _make_optimizer(tmp_path, max_iterations=3)
    counter = [0]
    with patch.object(type(opt), "_run_variant", _mock_run_variant(counter)):
        opt.run()

    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    accepted = [r for r in report["iterations"] if r["decision"] == "accepted"]
    if accepted:
        assert (tmp_path / "runs" / "current" / "system_prompt.txt").exists()


def test_early_stop_when_all_pass(tmp_path):
    """Loop should stop before max_iterations if all cases pass."""
    opt = _make_optimizer(tmp_path, max_iterations=20)

    def _fix_all(self, ws_variant, variant_name, variant_system, outer_model, max_turns):
        (ws_variant / "current" / "system_prompt.txt").write_text("fix_all")
        (ws_variant / "proposal.md").write_text("# Proposal\nfix everything")
        return "fix_all", None, "fix everything", 1, 100, 50

    with patch.object(type(opt), "_run_variant", _fix_all):
        opt.run()

    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    assert len(report["iterations"]) < 20
    assert report["final_train"]["passed"] == report["final_train"]["total"]


def test_no_change_proposal_recorded(tmp_path):
    """If proposer makes no changes, loop records 'no_change' and continues."""
    opt = _make_optimizer(tmp_path, max_iterations=3)

    def _no_change(self, ws_variant, variant_name, variant_system, outer_model, max_turns):
        prompt = (ws_variant / "current" / "system_prompt.txt").read_text()
        (ws_variant / "proposal.md").write_text("No safe fix found")
        return prompt, None, "No safe fix found", 0, 0, 0

    with patch.object(type(opt), "_run_variant", _no_change):
        opt.run()

    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    decisions = [r["decision"] for r in report["iterations"]]
    assert all(d == "no_change" for d in decisions)


def test_workspace_has_task_and_failure_matrix(tmp_path):
    """Workspace must include task.md, asi.md, surface_manifest.json, history/failure_matrix.md."""
    opt = _make_optimizer(tmp_path, max_iterations=2)
    counter = [0]
    with patch.object(type(opt), "_run_variant", _mock_run_variant(counter)):
        opt.run()

    iter_dir = tmp_path / "runs" / "iter-001"
    ws = iter_dir / "workspace"
    assert (ws / "task.md").exists()
    assert (ws / "asi.md").exists()
    assert (ws / "surface_manifest.json").exists()
    assert (ws / "history" / "history.md").exists()
    assert (ws / "history" / "failure_matrix.md").exists()
    assert (ws / "current" / "system_prompt.txt").exists()


def test_decision_records_variant_name(tmp_path):
    """Accepted/rejected decisions must record which variant was chosen."""
    opt = _make_optimizer(tmp_path, max_iterations=3)
    counter = [0]
    with patch.object(type(opt), "_run_variant", _mock_run_variant(counter)):
        opt.run()

    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    changed = [r for r in report["iterations"] if r["decision"] != "no_change"]
    for r in changed:
        assert "variant" in r


def test_failure_matrix_in_history_dir(tmp_path):
    """failure_matrix.md must exist in history/ after at least one iteration."""
    opt = _make_optimizer(tmp_path, max_iterations=2)
    counter = [0]
    with patch.object(type(opt), "_run_variant", _mock_run_variant(counter)):
        opt.run()

    fm_path = tmp_path / "runs" / "iter-001" / "workspace" / "history" / "failure_matrix.md"
    assert fm_path.exists()
    content = fm_path.read_text()
    assert (
        "PERSISTENT" in content or "RECURRING" in content or "NEW" in content or "FIXED" in content
    )
