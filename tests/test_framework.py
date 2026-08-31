"""Unit tests for the framework layer — no external calls, no real benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from agent_harness_optimizer.framework.benchmark import CaseScore, ResourceBudget, SplitScore

# ---------------------------------------------------------------------------
# ResourceBudget
# ---------------------------------------------------------------------------


def test_resource_budget_defaults():
    b = ResourceBudget()
    assert b.wall_time_s == 300.0
    assert b.max_steps == 100
    assert b.max_tokens == 500_000


def test_resource_budget_custom():
    b = ResourceBudget(wall_time_s=60.0, max_steps=20, max_tokens=None)
    assert b.wall_time_s == 60.0
    assert b.max_steps == 20
    assert b.max_tokens is None


def test_resource_budget_to_dict():
    b = ResourceBudget(wall_time_s=120.0, max_steps=50, max_tokens=100_000)
    d = b.to_dict()
    assert d == {"wall_time_s": 120.0, "max_steps": 50, "max_tokens": 100_000}


# ---------------------------------------------------------------------------
# CaseScore
# ---------------------------------------------------------------------------


def test_case_score_total_tokens():
    c = CaseScore(case_id="case_1", passed=True, prompt_tokens=100, completion_tokens=50)
    assert c.total_tokens == 150


def test_case_score_roundtrip():
    original = CaseScore(
        case_id="case_42",
        passed=False,
        stuck_type="timeout",
        within_time_budget=False,
        prompt_tokens=1000,
        completion_tokens=500,
        wall_time_s=305.0,
        category="api_call",
        extra={"tool_calls": [{"name": "get_user"}], "error": "timed out"},
    )
    d = original.to_dict()
    restored = CaseScore.from_dict(d)
    assert restored.case_id == "case_42"
    assert restored.passed is False
    assert restored.stuck_type == "timeout"
    assert restored.within_time_budget is False
    assert restored.total_tokens == 1500
    assert restored.extra["error"] == "timed out"


# ---------------------------------------------------------------------------
# SplitScore
# ---------------------------------------------------------------------------


def test_split_score_pass_rate():
    s = SplitScore(passed=7, total=10, reliability=0.9)
    assert s.pass_rate == pytest.approx(0.7)
    assert s.stuck_rate == pytest.approx(0.1)


def test_split_score_stuck_breakdown():
    cases = [
        CaseScore(case_id="a", passed=False, stuck_type="timeout"),
        CaseScore(case_id="b", passed=False, stuck_type="timeout"),
        CaseScore(case_id="c", passed=False, stuck_type="crash"),
        CaseScore(case_id="d", passed=True, stuck_type=""),
    ]
    s = SplitScore(passed=1, total=4, reliability=0.75, cases=cases)
    bd = s.stuck_breakdown
    assert bd["timeout"] == 2
    assert bd["crash"] == 1
    assert "loop" not in bd


def test_split_score_roundtrip():
    cases = [
        CaseScore(
            case_id="x1", passed=True, category="retail", prompt_tokens=200, completion_tokens=100
        ),
        CaseScore(case_id="x2", passed=False, stuck_type="loop", extra={"tool_calls": []}),
    ]
    original = SplitScore(passed=1, total=2, reliability=0.5, tokens_per_case=150.0, cases=cases)
    d = original.to_dict()
    restored = SplitScore.from_dict(d)
    assert restored.passed == 1
    assert restored.total == 2
    assert restored.reliability == pytest.approx(0.5)
    assert restored.tokens_per_case == pytest.approx(150.0)
    assert len(restored.cases) == 2
    assert restored.cases[0].case_id == "x1"
    assert restored.cases[1].stuck_type == "loop"
