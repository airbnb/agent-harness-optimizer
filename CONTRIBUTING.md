# Contributing

Thank you for your interest in improving Agent Harness Optimizer (AHO).
Bug reports, fixes, new benchmarks, and new optimizers are all welcome.

## Getting started

```bash
git clone https://github.com/airbnb/agent-harness-optimizer.git
cd agent-harness-optimizer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest/ruff
pip install -e ".[tau]"          # optional: τ²-bench support
pip install -e ".[miprov2]"      # optional: MIPROv2 optimizer (DSPy)
pip install -e ".[gepa]"         # optional: GEPA optimizer
```

BFCL case data is not redistributed with this repository; see
`data/bfcl/README.md` for how to download it.

## Before you open a pull request

All three of these must pass — CI runs the same commands:

```bash
ruff check agent_harness_optimizer/ scripts/
ruff format --check agent_harness_optimizer/ scripts/
python -m pytest tests/ --cov=agent_harness_optimizer --cov-fail-under=20
```

The test suite is self-contained: it uses a mock benchmark and makes no
external API calls, so it needs no credentials and runs in seconds.

## Project layout

The framework separates **what** is optimized from **how**:

- `agent_harness_optimizer/framework/` — the `Benchmark` and `Optimizer`
  abstract interfaces plus the shared result types (`CaseScore`,
  `SplitScore`, `ResourceBudget`).
- `agent_harness_optimizer/benchmarks/` — benchmark adapters (BFCL,
  τ²-bench). A new benchmark subclasses `Benchmark` and implements its
  abstract methods; no optimizer code needs to change.
- `agent_harness_optimizer/optimizers/` — search procedures (PRISM,
  BetterHarness, GEPA and MIPROv2 wrappers). A new optimizer subclasses
  `Optimizer`; no benchmark code needs to change.
- `scripts/` — metric aggregation (`compute_metrics.py`, `rellift.py`) and
  paper-reproduction helpers.
- `configs/paper/` — the exact run configurations behind the paper's tables.

## Guidelines

- **Keep the two axes independent.** Benchmark-specific data belongs in
  `CaseScore.extra`; optimizers must only read the shared `SplitScore`
  fields.
- **Middleware edits go through the request API.** In BFCL middleware,
  mutate tool calls with `request.override(tool_call=...)`, never by
  mutating `call["args"]` in place.
- **Add tests.** New framework or optimizer behavior should come with a
  test against the mock benchmark (`tests/mock_benchmark.py`), which is the
  pattern used by the existing suite.
- **Don't change paper configs.** Files under `configs/paper/` document the
  published experiments; fixes to them are only warranted when they diverge
  from the paper itself.
- **No secrets or internal hostnames** in code, configs, or test fixtures.

## Reporting issues

Please include: the exact CLI command or config, the benchmark and optimizer,
the Python version, and the relevant part of the run's `report.json` or
console output. For reproducibility questions, start from
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
