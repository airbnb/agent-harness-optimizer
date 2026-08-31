"""Tests using MockBenchmark — exercises scoring logic without real agents."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from tests.mock_benchmark import MockBenchmark


@pytest.fixture
def benchmark(tmp_path):
    return MockBenchmark()


def test_baseline_all_fail(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("You are a test agent.", "train", tmp_path / "s"))
    assert score.passed == 0
    assert score.total == 10
    assert score.reliability < 1.0  # case_2 and case_7 are stuck:loop


def test_fix_one_case(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("fix_3", "train", tmp_path / "s"))
    assert score.passed == 1
    passing = [c.case_id for c in score.cases if c.passed]
    assert "case_3" in passing


def test_fix_all(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("fix_all", "train", tmp_path / "s"))
    assert score.passed == score.total


def test_reliability_reflects_stuck(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("You are a test agent.", "train", tmp_path / "s"))
    stuck = [c for c in score.cases if c.stuck_type]
    # case_2 and case_7 should be stuck:loop when not passing
    stuck_ids = {c.case_id for c in stuck}
    assert "case_2" in stuck_ids
    assert "case_7" in stuck_ids
    assert score.stuck_breakdown.get("loop", 0) == 2


def test_holdout_split(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("fix_all", "holdout", tmp_path / "s"))
    assert score.passed == 5
    assert score.total == 5


def test_max_cases_subsample(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("fix_all", "train", tmp_path / "s", max_cases=3))
    assert score.total == 3


def test_build_asi_shows_failures(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("You are a test agent.", "train", tmp_path / "s"))
    asi = benchmark.build_asi(score, None)
    assert "Failures: 10/10" in asi
    assert "case_2" in asi


def test_extract_top_patterns(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("You are a test agent.", "train", tmp_path / "s"))
    patterns = benchmark.extract_top_patterns(score, n=3)
    assert len(patterns) >= 1
    assert all("key" in p and "count" in p and "case_ids" in p for p in patterns)


def test_write_case_files(benchmark, tmp_path):
    score = asyncio.run(benchmark.score_async("fix_3 fix_5", "train", tmp_path / "s"))
    benchmark.write_case_files(tmp_path / "ws", score)
    failures_dir = tmp_path / "ws" / "train_cases" / "failures"
    passing_dir = tmp_path / "ws" / "train_cases" / "passing"
    assert failures_dir.exists()
    assert passing_dir.exists()
    passing_ids = {f.stem for f in passing_dir.iterdir()}
    assert "case_3" in passing_ids
    assert "case_5" in passing_ids


def test_resource_budget_on_mock(tmp_path):
    from agent_harness_optimizer.framework.benchmark import ResourceBudget

    b = MockBenchmark(budget=ResourceBudget(wall_time_s=0.5, max_steps=3, max_tokens=500))
    assert b.resource_budget.wall_time_s == 0.5
    assert b.resource_budget.max_steps == 3
