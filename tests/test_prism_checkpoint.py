"""Offline regression tests for interrupted PRISM checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_harness_optimizer.framework.optimizer import OptimizeConfig
from agent_harness_optimizer.optimizers.prism.loop import PRISMOptimizer
from agent_harness_optimizer.optimizers.prism.types import Candidate
from tests.mock_benchmark import MockBenchmark


@pytest.fixture
def optimizer(tmp_path):
    return PRISMOptimizer(
        MockBenchmark(),
        OptimizeConfig(
            output_dir=tmp_path / "runs",
            inner_model="mock-inner",
            outer_model="mock-outer",
        ),
        generations=0,
    )


def _candidate(gen):
    return Candidate(
        uid=f"gen{gen:03d}_seed",
        generation=gen,
        prompt=f"fix_{gen}",
        middleware_dir=None,
    )


def _write_checkpoint(optimizer, gen, **overrides):
    """Write the legacy on-disk format, including deliberately broken files."""
    directory = optimizer.config.output_dir / f"gen-{gen:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    candidate = _candidate(gen)
    files = {
        "frontier.json": json.dumps([candidate.to_dict()]),
        "all_candidates.json": json.dumps([candidate.to_dict()]),
        "screen_rollouts.json": json.dumps({"screen_rollouts": 17}),
    }
    files.update(overrides)
    for name, contents in files.items():
        if contents is not None:
            (directory / name).write_bytes(
                contents.encode() if isinstance(contents, str) else contents
            )
    return directory


def test_empty_run_has_no_checkpoint(optimizer):
    assert optimizer._load_state() == (-1, [], [], 0)
    (optimizer.config.output_dir / "gen-001").mkdir(parents=True)
    assert optimizer._load_state() == (-1, [], [], 0)


def test_load_latest_complete_checkpoint(optimizer):
    _write_checkpoint(optimizer, 0)
    _write_checkpoint(optimizer, 1)
    # An interrupted evaluation may leave a generation directory without a checkpoint.
    (optimizer.config.output_dir / "gen-002").mkdir()

    gen, frontier, candidates, rollouts = optimizer._load_state()

    assert gen == 1
    assert frontier == candidates == [_candidate(1)]
    assert rollouts == 17


@pytest.mark.parametrize(
    "overrides",
    [
        {"all_candidates.json": None},
        {"frontier.json": "[{"},
        {"all_candidates.json": "[{"},
        {"screen_rollouts.json": "{"},
        {"screen_rollouts.json": "{}"},
        {"frontier.json": "[]"},
        {"all_candidates.json": "[]"},
        {"frontier.json": "{}"},
        {"all_candidates.json": "[{}]"},
        {"frontier.json": b"\xff"},
    ],
    ids=[
        "missing-history",
        "truncated-frontier",
        "truncated-history",
        "truncated-rollouts",
        "missing-rollout-count",
        "empty-frontier",
        "empty-history",
        "invalid-frontier-shape",
        "invalid-candidate",
        "invalid-encoding",
    ],
)
def test_resume_skips_broken_checkpoint(optimizer, capsys, overrides):
    _write_checkpoint(optimizer, 0)
    broken_dir = _write_checkpoint(optimizer, 1, **overrides)

    gen, frontier, candidates, rollouts = optimizer._load_state()

    assert gen == 0
    assert frontier == candidates == [_candidate(0)]
    assert rollouts == 17
    output = capsys.readouterr().out
    assert str(broken_dir) in output
    assert "[prism] Skipping incomplete checkpoint" in output


def test_legacy_checkpoint_without_rollout_count(optimizer):
    _write_checkpoint(optimizer, 0, **{"screen_rollouts.json": None})
    assert optimizer._load_state()[3] == 0


def test_no_valid_checkpoint_does_not_silently_reseed(optimizer):
    _write_checkpoint(optimizer, 0, **{"all_candidates.json": None})
    with pytest.raises(ValueError, match="No complete PRISM checkpoint"):
        optimizer._load_state()


def test_permission_errors_are_not_treated_as_corruption(optimizer, monkeypatch):
    _write_checkpoint(optimizer, 0)

    def denied(*args, **kwargs):
        raise PermissionError("checkpoint is not readable")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError, match="not readable"):
        optimizer._load_state()


@pytest.mark.parametrize("existing_checkpoint", [False, True], ids=["new", "retry"])
@pytest.mark.parametrize(
    "failed_file",
    [".frontier.json.tmp", "all_candidates.json", "screen_rollouts.json", "gen_stats.json"],
)
def test_interrupted_save_is_not_published(
    optimizer, monkeypatch, existing_checkpoint, failed_file
):
    _write_checkpoint(optimizer, 0)
    if existing_checkpoint:
        _write_checkpoint(optimizer, 1)
    original_write = Path.write_text

    def interrupted_write(path, contents, *args, **kwargs):
        if path.name == failed_file:
            original_write(path, "{")  # Simulate interruption partway through a write.
            raise OSError("interrupted checkpoint write")
        return original_write(path, contents, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "write_text", interrupted_write)
        with pytest.raises(OSError, match="interrupted checkpoint write"):
            optimizer._save_gen(1, [_candidate(1)], [_candidate(0), _candidate(1)], 23, {})

    directory = optimizer.config.output_dir / "gen-001"
    assert not (directory / "frontier.json").exists()
    assert optimizer._load_state()[0] == 0

    # Retrying the same generation must tolerate leftover staging files.
    optimizer._save_gen(1, [_candidate(1)], [_candidate(0), _candidate(1)], 23, {})
    gen, frontier, candidates, rollouts = optimizer._load_state()
    assert gen == 1
    assert frontier == [_candidate(1)]
    assert candidates == [_candidate(0), _candidate(1)]
    assert rollouts == 23
    assert not (directory / ".frontier.json.tmp").exists()


def test_frontier_is_published_after_other_files(optimizer, monkeypatch):
    original_replace = Path.replace
    publications = []

    def checked_replace(path, target):
        if target.name == "frontier.json":
            directory = target.parent
            assert json.loads((directory / "all_candidates.json").read_text()) == [
                _candidate(0).to_dict()
            ]
            assert json.loads((directory / "screen_rollouts.json").read_text()) == {
                "screen_rollouts": 23
            }
            assert json.loads((directory / "gen_stats.json").read_text()) == {"generation": 0}
            publications.append(target)
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", checked_replace)
    optimizer._save_gen(0, [_candidate(0)], [_candidate(0)], 23, {"generation": 0})
    assert len(publications) == 1
    assert optimizer._load_state()[0] == 0


@pytest.mark.parametrize("existing_checkpoint", [False, True], ids=["new", "retry"])
def test_failed_publication_keeps_previous_generation(optimizer, monkeypatch, existing_checkpoint):
    _write_checkpoint(optimizer, 0)
    if existing_checkpoint:
        _write_checkpoint(optimizer, 1)
    original_replace = Path.replace

    def interrupted_replace(path, target):
        if target.name == "frontier.json":
            raise OSError("interrupted checkpoint publication")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted checkpoint publication"):
        optimizer._save_gen(1, [_candidate(1)], [_candidate(0), _candidate(1)], 23)
    assert optimizer._load_state()[0] == 0


def test_resume_replays_interrupted_generation_without_reseeding(optimizer):
    optimizer.run()  # A real seed checkpoint, scored only by MockBenchmark.
    seed_path = optimizer.config.output_dir / "gen-000" / "frontier.json"
    seed_bytes = seed_path.read_bytes()
    _write_checkpoint(optimizer, 1, **{"all_candidates.json": None})
    optimizer.config.resume = True
    optimizer.generations = 1
    optimizer.gc.prompt_only = True
    optimizer.gc.mutations_per_gen = 1
    optimizer.gc.no_crossover = True

    with (
        patch(
            "agent_harness_optimizer.optimizers.prism.loop.analyze_patterns",
            autospec=True,
            return_value=([], 0, 0),
        ),
        patch(
            "agent_harness_optimizer.optimizers.prism.loop.mutate",
            autospec=True,
            return_value=("fix_all", None, "fix all mock cases", "prompt_only", 0, 0),
        ) as mutate,
        patch.object(
            optimizer.benchmark, "score_async", wraps=optimizer.benchmark.score_async
        ) as score,
    ):
        optimizer.run()

    mutate.assert_called_once()
    assert mutate.call_args.kwargs["generation"] == 1
    assert mutate.call_args.args[0].uid == "gen000_seed"
    assert seed_path.read_bytes() == seed_bytes
    assert score.await_count == 4  # Child train/holdout and final train/holdout, no seed.
    assert all(call.args[0] == "fix_all" for call in score.call_args_list)
    gen, frontier, candidates, _ = optimizer._load_state()
    assert gen == 1
    assert len(candidates) == 2
    assert frontier[0].prompt == "fix_all"
    report = json.loads((optimizer.config.output_dir / "report.json").read_text())
    assert report["final_train"]["pass_rate"] == 1.0
