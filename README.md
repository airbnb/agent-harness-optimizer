# Agent Harness Optimizer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

**Agent Harness Optimizer (AHO)** is a benchmark-agnostic framework for automatically optimizing LLM agent harnesses. Given a benchmark, it evolves an agent's **system prompt** and **tool-call middleware** using eval feedback — improving pass rate, reliability, and cost without manual tuning.

---

To replicate the paper's experiments, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
and the run configurations in [`configs/paper/`](configs/paper/).

## How It Works

Each run uses two model roles:

- **`--inner-model`** — the agent being optimized; runs benchmark cases and is scored
- **`--outer-model`** — the proposer LLM; reads failures and writes targeted fixes

Two surfaces are edited per run, written to a `current/` workspace directory:

```
current/
  system_prompt.txt          # behavioral instructions for the inner agent
  middleware/
    custom_middleware.py     # tool-call interception — silent corrections, error blocks
    agent_setup.py           # wires MIDDLEWARE = [...] into the agent
```

The optimizer loop proposes edits, scores them, accepts improvements, and checkpoints every accepted change. The final result is written to `report.json` and `final_diff.md`.

---

## Optimizers

| Optimizer | Strategy | Best for |
|-----------|----------|----------|
| **PRISM** | Genetic-Pareto evolutionary search | Multi-surface optimization, highest lift |
| **BetterHarness** | Linear propose → accept loop | Fast iteration, interpretable diffs |
| **MIPROv2** | Bayesian instruction search (DSPy) | Prompt-only, structured instruction tuning |
| **GEPA** | Reflective trajectory mutation (ICLR 2026) | Per-instance Pareto selection |

---

## Install

```bash
# from source
git clone https://github.com/airbnb/agent-harness-optimizer.git
cd agent-harness-optimizer
uv pip install -e .
uv pip install -e ".[tau]"   # adds tau-bench support
uv pip install -e ".[dev]"   # adds pytest/ruff
```

**BFCL data**: the benchmark case files are not redistributed with this
repository. Download the BFCL v4 multi-turn datasets (and matching
`possible_answer` files) from the
[Gorilla / BFCL repository](https://github.com/ShishirPatil/gorilla) and place
them under `data/bfcl/` (see `data/bfcl/README.md` for the expected layout).

---

## Quick Start

```bash
# PRISM on BFCL
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer prism \
    --inner-model openai/gpt-5.4-mini \
    --outer-model bedrock/global.anthropic.claude-sonnet-4-6 \
    --output-dir runs/bfcl-prism-001 \
    --train-cases 100 --holdout-cases 100 \
    --generations 4 --mutations-per-gen 3 \
    --wall-time-s 300 --max-steps 100

# BetterHarness on tau-bench retail
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model bedrock/global.anthropic.claude-sonnet-4-6 \
    --output-dir runs/tau-retail-bh-001 \
    --train-cases 20 --holdout-cases 20 \
    --wall-time-s 600 --max-steps 200

# Resume a crashed run
python -m agent_harness_optimizer.cli ... --resume
```

---

## Benchmarks

| Benchmark flag | Description |
|----------------|-------------|
| `bfcl` | BFCL v4 multi-turn tool calling |
| `tau-airline` | tau-bench airline domain |
| `tau-retail` | tau-bench retail domain |
| `tau-telecom` | tau-bench telecom domain |
| `tau-mock` | tau-bench mock domain |

**tau-bench budgets**: use `--wall-time-s 600 --max-steps 200`. tau2 counts every message exchange (including tool responses) as one step.

---

## Key CLI Flags

### Shared

| Flag | Default | Description |
|------|---------|-------------|
| `--train-cases N` | 100 | Cases used during optimization |
| `--holdout-cases N` | 100 | Cases held out for per-iteration validation |
| `--wall-time-s S` | 300 | Per-case wall-clock timeout |
| `--max-steps N` | 100 | Per-case step/turn limit |
| `--max-tokens N` | 500000 | Per-case token budget |
| `--split-seed N` | — | Enables deterministic train/holdout/scorecard split + final scorecard eval |
| `--scorecard-trials k` | 1 | Run final scorecard k times; case passes only if it passes all k (pass^k, tau only) |
| `--resume` | false | Resume from last checkpoint |
| `--human-approval` | false | Pause for human review before accepting each change |

### PRISM

| Flag | Default | Description |
|------|---------|-------------|
| `--generations N` | 10 | Number of evolutionary generations |
| `--mutations-per-gen N` | 3 | Parallel mutations per generation |
| `--prism-pass-rate-metric` | `combined` | Pareto metric: `combined`, `pass_rate`, or `reliability` |
| `--prism-prompt-only` | false | Disable middleware mutations |

### BetterHarness

| Flag | Default | Description |
|------|---------|-------------|
| `--max-iterations N` | 10 | Max propose → accept iterations |
| `--bh-prompt-only` | false | Disable middleware mutations |

### MIPROv2

| Flag | Default | Description |
|------|---------|-------------|
| `--miprov2-num-candidates N` | 10 | Instruction candidates to evaluate |
| `--miprov2-num-trials N` | 20 | Bayesian optimization trials |
| `--miprov2-minibatch-size N` | 25 | Cases per minibatch score |

---

## Output Structure

```
runs/<name>/
  experiment_config.json        # full run configuration
  report.json                   # baseline → final scores, learning curve, scorecard
  final_diff.md                 # human-readable baseline → best accepted diff
  baseline/
    train/   holdout/   summary.json
  iter-001/
    decision.json               # BetterHarness: per-iteration decision + scores
    train/   holdout/
  gen-001/
    frontier.json               # PRISM: Pareto frontier after each generation
    all_candidates.json
```

**Key fields in `report.json`:**

| Field | Description |
|-------|-------------|
| `baseline_holdout.pass_rate` | Seed prompt pass rate on holdout split |
| `final_holdout.pass_rate` | Best accepted candidate pass rate on holdout |
| `delta_holdout` | `final - baseline` holdout pass rate |
| `final_scorecard.pass_rate` | Out-of-sample generalization score |
| `final_scorecard.delta_scorecard_pp` | Scorecard lift in percentage points |
| `final_scorecard.improving_run` | `true` if the run produced a statistically meaningful lift |
| `final_train.reliability` | `1 - stuck_rate` — fraction of cases completing without timeout/crash |
| `holdout_series` | Holdout pass rate at each iteration/generation |

---

## Algorithms

### PRISM

```
for each generation:
    analyze failures → group by root cause, assign fix surface (LLM)
    run N parallel mutations: prompt_only · middleware_only · both
    deduplicate no-op mutations
    full eval all children on train + holdout
    crossover if children address complementary failure sets
    update Pareto frontier (pass_rate × reliability)
```

### BetterHarness

```
for each iteration:
    outer LLM proposes one fix (prompt / middleware / both)
    screen on train subset
    accept if TCR improves > 1%; else try error-rate; else try cost
    full eval accepted candidate on train + holdout → checkpoint
```

### MIPROv2

Wraps [DSPy MIPROv2](https://github.com/stanfordnlp/dspy). Bayesian (TPE/Optuna) search over instruction candidates, scored on random train minibatches. Requires: `uv pip install -e ".[miprov2]"`.

### GEPA

Wraps [gepa-ai/gepa](https://github.com/gepa-ai/gepa) (Agrawal et al., ICLR 2026). Reflective trajectory mutation with per-instance Pareto-front selection.

---

## Custom Benchmark

Subclass `Benchmark` and implement the required methods:

```python
from agent_harness_optimizer.framework.benchmark import Benchmark, SplitScore, CaseScore, ResourceBudget

class MyBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "my-bench"

    @property
    def default_model(self) -> str:
        return "openai/gpt-4o-mini"

    @property
    def resource_budget(self) -> ResourceBudget:
        return ResourceBudget(wall_time_s=300, max_steps=100)

    async def score_async(
        self, prompt: str, split: str, output_dir: Path,
        *, middleware_dir=None, max_cases=None, case_indices=None, num_trials=1,
    ) -> SplitScore:
        # run your eval, return SplitScore(passed=N, total=M, cases=[...])
        ...

    def build_asi(self, score, failure_matrix_cases): ...
    def extract_top_patterns(self, score, n=3): ...
    def write_case_files(self, workspace, score, failure_matrix_cases): ...
    def build_model(self, model_name): ...
```

---

## Auth

Model strings use [litellm format](https://docs.litellm.ai/docs/providers). Set standard env vars for your provider:

| Provider | Model prefix | Env vars |
|----------|-------------|----------|
| OpenAI | `openai/` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| AWS Bedrock | `bedrock/` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` |
| Azure OpenAI | `azure/` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| Google Vertex | `vertex_ai/` | `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` |

---

## Development

```bash
uv pip install -e ".[dev]"
python -m pytest tests/
ruff check agent_harness_optimizer/ && ruff format agent_harness_optimizer/
```

---

## Contributing

Bug reports and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhao2026beyondprompts,
  title     = {Beyond Prompts: Measuring and Optimizing {LLM} Tool-Agent Harnesses},
  author    = {Zhao, Cen and Ruan, Haibo and Chen, Wenjie and Tu, Pei-fen and Abbasi, Usman and Hesch, Joel},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```
