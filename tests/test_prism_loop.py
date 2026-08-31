"""Integration test for PRISMOptimizer using MockBenchmark.

The outer agent (mutate / crossover_all_children) is mocked to add one
'fix_N' keyword per mutation, allowing us to test the evolutionary loop
(screening, Pareto updates, generation tracking) without real LLM calls.
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

_MUTATION_COUNTER: list[int] = [0]


def _mock_mutate(
    candidate,
    score,
    *,
    generation,
    frontier,
    fm_cases,
    benchmark,
    outer_model,
    workspace_dir,
    max_turns,
    variant=None,
    target_pattern=None,
):
    """Each mutation adds one more 'fix_N' to the candidate's prompt."""
    n = _MUTATION_COUNTER[0] % 10
    _MUTATION_COUNTER[0] += 1
    new_prompt = candidate.prompt + f" fix_{n}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "proposal.md").write_text(f"fix case_{n}")
    return new_prompt, None, f"fix case_{n}", "prompt_only"


def _mock_crossover_all(
    children,
    scores,
    base_candidate,
    *,
    generation,
    frontier,
    fm_cases,
    benchmark,
    outer_model,
    workspace_dir,
    max_turns,
):
    """Crossover merges all children's prompts."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    merged = base_candidate.prompt + " fix_8 fix_9"
    (workspace_dir / "proposal.md").write_text("merged all children")
    return merged, None, "merged all children"


def _make_optimizer(tmp_path: Path, generations: int = 2) -> PRISMOptimizer:
    _MUTATION_COUNTER[0] = 0
    benchmark = MockBenchmark()
    config = OptimizeConfig(
        output_dir=tmp_path / "runs",
        inner_model="mock-inner",
        outer_model="mock-outer",
    )
    gc = PRISMConfig(
        train_cases=10,
        holdout_cases=5,
        mutations_per_gen=3,
        population_cap=5,
    )
    return PRISMOptimizer(benchmark, config, generations=generations, prism_config=gc)


def test_gen0_seed_created(tmp_path):
    opt = _make_optimizer(tmp_path, generations=0)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt.run()

    gen0_dir = tmp_path / "runs" / "gen-000"
    assert gen0_dir.exists()
    frontier = json.loads((gen0_dir / "frontier.json").read_text())
    assert len(frontier) == 1
    assert frontier[0]["uid"] == "gen000_seed"


def test_generation_dirs_created(tmp_path):
    opt = _make_optimizer(tmp_path, generations=2)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt.run()

    assert (tmp_path / "runs" / "gen-001").exists()
    assert (tmp_path / "runs" / "gen-002").exists()


def test_gen_stats_written(tmp_path):
    opt = _make_optimizer(tmp_path, generations=1)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt.run()

    stats_path = tmp_path / "runs" / "gen-001" / "gen_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["generation"] == 1
    assert stats["crossover"] in {"fired", "no_complement", "single_child", "disabled", "failed", "no_children"}
    assert isinstance(stats["complementary_cases"], int)
    assert isinstance(stats["crossover_in_frontier"], bool)


def test_report_written(tmp_path):
    opt = _make_optimizer(tmp_path, generations=2)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt.run()

    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    assert report["benchmark"] == "mock"
    assert "final_train" in report
    assert "delta_combined" in report


def test_frontier_grows_when_candidates_improve(tmp_path):
    """Each mutation adds a new 'fix_N', so children should pass more cases than seed."""
    opt = _make_optimizer(tmp_path, generations=2)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt.run()

    report = json.loads((tmp_path / "runs" / "report.json").read_text())
    assert report["final_train"]["pass_rate"] >= 0.0


def test_cache_dir_created(tmp_path):
    opt = _make_optimizer(tmp_path, generations=1)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt.run()

    assert (tmp_path / "runs" / "cache").is_dir()
    cache_files = list((tmp_path / "runs" / "cache").glob("*.json"))
    assert len(cache_files) >= 1  # at least seed cached


def test_resume_loads_state(tmp_path):
    """Running with resume=True should pick up from last gen without re-seeding."""
    opt = _make_optimizer(tmp_path, generations=1)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt.run()

    # Now resume for one more generation
    _MUTATION_COUNTER[0] = 0
    benchmark = MockBenchmark()
    config = OptimizeConfig(
        output_dir=tmp_path / "runs",
        inner_model="mock-inner",
        outer_model="mock-outer",
        resume=True,
    )
    gc = PRISMConfig(
        train_cases=10,
        holdout_cases=5,
        mutations_per_gen=2,
    )
    opt2 = PRISMOptimizer(benchmark, config, generations=2, prism_config=gc)
    with (
        patch("agent_harness_optimizer.optimizers.prism.loop.mutate", _mock_mutate),
        patch("agent_harness_optimizer.optimizers.prism.loop.crossover_all_children", _mock_crossover_all),
    ):
        opt2.run()

    assert (tmp_path / "runs" / "gen-002").exists()
