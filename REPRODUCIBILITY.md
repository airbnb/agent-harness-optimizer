# Reproducibility Guide

This document provides exact commands to replicate all experiments in:

> **"Beyond Prompts: Measuring and Optimizing LLM Tool-Agent Harnesses"**

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Auth setup](#auth-setup)
3. [Paper experiment parameters](#paper-experiment-parameters)
4. [BFCL experiments](#bfcl-experiments)
5. [τ-Retail experiments](#τ-retail-experiments)
6. [τ-Telecom experiments](#τ-telecom-experiments)
7. [Baseline-only scoring](#baseline-only-scoring)
8. [Ablation and middleware-variant arms](#ablation-and-middleware-variant-arms)
9. [Automated reproduction script](#automated-reproduction-script)
10. [Extracting metrics](#extracting-metrics)
11. [Expected outputs](#expected-outputs)

---

## Prerequisites

Python 3.12 is required. We strongly recommend a clean virtual environment:
mixing agent-harness-optimizer into a system Python managed by Homebrew or the OS
package manager often hits package-ownership conflicts (e.g. cffi cannot be
upgraded by pip because it was installed by brew).

```bash
# 1. Create and activate a clean venv (required on macOS + Homebrew Python)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. Install from source with all extras used in the paper
git clone https://github.com/airbnb/agent-harness-optimizer.git
cd agent-harness-optimizer
pip install -e ".[tau,miprov2,gepa,dev]"
```

The `[tau]` extra installs the correct τ²-bench package directly from the
Sierra Research repository at the paper-pinned commit (`337326e`, version
`0.2.1.dev0`). Do **not** run `pip install tau2` from PyPI — that name belongs
to an unrelated chemistry library.

If you only need a subset:

```bash
pip install -e .                    # core (BFCL only)
pip install -e ".[tau]"             # add τ²-bench
pip install -e ".[miprov2]"         # add MIPROv2 optimizer (DSPy)
pip install -e ".[gepa]"            # add GEPA optimizer
pip install -e ".[dev]"             # add pytest + ruff
```

---

## Auth setup

Set environment variables for the LLM providers used in the paper:

```bash
# For Azure OpenAI (inner model: gpt-4o-mini or gpt-5.4-mini)
export AZURE_API_KEY=<your-key>
export AZURE_API_BASE=https://<your-resource>.openai.azure.com/
export AZURE_API_VERSION=2024-02-15-preview

# For Anthropic direct API (outer model: claude-opus-4-7 or claude-sonnet-4-6)
export ANTHROPIC_API_KEY=<your-key>

# For OpenAI (alternative inner model)
export OPENAI_API_KEY=<your-key>

# For AWS Bedrock (alternative provider)
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_REGION_NAME=us-west-2
```

Model strings use [litellm format](https://docs.litellm.ai/docs/providers):
- `openai/gpt-5.4-mini` — OpenAI GPT-4o mini
- `anthropic/claude-opus-4-7` — Anthropic Claude Opus 4.7 (direct API)
- `anthropic/claude-sonnet-4-6` — Anthropic Claude Sonnet 4.6 (direct API)
- `bedrock/us.anthropic.claude-opus-4-7` — same model via AWS Bedrock
- `azure/gpt-4o-mini` — Azure-hosted GPT-4o mini

---

## Paper experiment parameters

All paper runs use these per-case resource budgets:

| Benchmark | --wall-time-s | --max-steps | --max-tokens | --train-cases | --holdout-cases |
|-----------|---------------|-------------|--------------|---------------|-----------------|
| BFCL      | 300           | 100         | 500000       | 100           | 100             |
| τ-Retail  | 600           | 200         | 100000       | 20            | 20              |
| τ-Telecom | 600           | 200         | 100000       | 20            | 20              |

Each τ² domain's 114-task base pool splits into 20 repair (`--train-cases`),
20 gate (`--holdout-cases`), and 74 held-out scorecard cases per split seed.
The outer proposer is capped at `--outer-max-turns 300` (the CLI default).

Optimizer-specific flags:

| Optimizer      | Key flags                                                        |
|----------------|------------------------------------------------------------------|
| PRISM          | `--generations 10 --mutations-per-gen 3`                        |
| BetterHarness  | `--max-iterations 10`                                            |
| MIPROv2 (BFCL) | `--miprov2-num-candidates 10 --miprov2-num-trials 34 --miprov2-minibatch-size 25` |
| MIPROv2 (τ²)   | `--miprov2-num-candidates 10 --miprov2-num-trials 5 --miprov2-minibatch-size 4` |
| GEPA (BFCL)    | `--gepa-max-metric-calls 800 --gepa-reflection-minibatch-size 50` |
| GEPA (τ²)      | `--gepa-max-metric-calls 160 --gepa-reflection-minibatch-size 20` |

---

## BFCL experiments

**Inner model:** `openai/gpt-5.4-mini`
**Outer model:** `anthropic/claude-opus-4-7`

### PRISM on BFCL

```bash
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer prism \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-prism \
    --generations 10 \
    --mutations-per-gen 3 \
    --train-cases 100 \
    --holdout-cases 100 \
    --wall-time-s 300 \
    --max-steps 100 \
    --no-human-approval
```

### BetterHarness on BFCL

```bash
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-bh \
    --max-iterations 10 \
    --train-cases 100 \
    --holdout-cases 100 \
    --wall-time-s 300 \
    --max-steps 100 \
    --no-human-approval
```

### PRISM-PO on BFCL (prompt-only ablation)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer prism \
    --prism-prompt-only \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-prism-po \
    --generations 10 \
    --mutations-per-gen 3 \
    --train-cases 100 \
    --holdout-cases 100 \
    --wall-time-s 300 \
    --max-steps 100 \
    --no-human-approval
```

### BH-PO on BFCL (prompt-only ablation)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer better-harness \
    --bh-prompt-only \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-bh-po \
    --max-iterations 10 \
    --train-cases 100 \
    --holdout-cases 100 \
    --wall-time-s 300 \
    --max-steps 100 \
    --no-human-approval
```

### MIPROv2 on BFCL

```bash
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer miprov2 \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-miprov2 \
    --miprov2-num-candidates 10 \
    --miprov2-num-trials 34 \
    --miprov2-minibatch-size 25 \
    --train-cases 100 \
    --holdout-cases 100 \
    --wall-time-s 300 \
    --max-steps 100 \
    --no-human-approval
```

### GEPA on BFCL

```bash
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer gepa \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-gepa \
    --gepa-max-metric-calls 800 \
    --gepa-reflection-minibatch-size 50 \
    --train-cases 100 \
    --holdout-cases 100 \
    --wall-time-s 300 \
    --max-steps 100 \
    --no-human-approval
```

### BFCL baseline (no optimization)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-baseline \
    --max-iterations 0 \
    --train-cases 100 \
    --holdout-cases 100 \
    --wall-time-s 300 \
    --max-steps 100 \
    --no-human-approval
```

### BFCL cross-validation (4 split seeds × 4 repeats)

The paper reports results across 4 split seeds (0–3) with 4 repeats each.
Use `scripts/reproduce_paper.py` to launch individual seed/repeat combinations:

```bash
# Run all 4 split seeds for BFCL/PRISM (one at a time)
for seed in 0 1 2 3; do
    python scripts/reproduce_paper.py \
        --benchmark bfcl \
        --optimizer prism \
        --inner-model openai/gpt-5.4-mini \
        --outer-model anthropic/claude-opus-4-7 \
        --output-dir runs/paper-bfcl-prism-s${seed} \
        --split-seed ${seed}
done
```

Estimated wall time per seed: ~5–7 h. Run multiple seeds in parallel by
launching them in separate terminals or with a job scheduler.

---

## τ-Retail experiments

**Inner model:** `openai/gpt-5.4-mini`
**Outer model:** `anthropic/claude-opus-4-7`

tau-bench data directory must be set (see Prerequisites):

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

### PRISM on τ-Retail

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer prism \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-prism \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --generations 10 \
    --mutations-per-gen 3 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### BetterHarness on τ-Retail

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-bh \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 10 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### PRISM-PO on τ-Retail (prompt-only ablation)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer prism \
    --prism-prompt-only \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-prism-po \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --generations 10 \
    --mutations-per-gen 3 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### BH-PO on τ-Retail (prompt-only ablation)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer better-harness \
    --bh-prompt-only \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-bh-po \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 10 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### MIPROv2 on τ-Retail

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer miprov2 \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-miprov2 \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --miprov2-num-candidates 10 \
    --miprov2-num-trials 5 \
    --miprov2-minibatch-size 4 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### GEPA on τ-Retail

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer gepa \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-gepa \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --gepa-max-metric-calls 160 \
    --gepa-reflection-minibatch-size 20 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### τ-Retail baseline (no optimization)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-baseline \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 0 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

---

## τ-Telecom experiments

Note on pass^4 scoring: the paper's τ-Telecom PRISM-PO / PRISM-MW rows
(marked § in the paper's tables) were scored from four genuine simulator
trials per case — the original scorecard trial plus three additional trials
run afterwards with `scripts/rescore_telecom_passk.py` — AND-combined into
pass^4 outside the benchmark-native pass^4 harness path used by the other
Telecom arms. To reproduce that scoring path, run the script over a completed
run directory; to use the native path instead, pass `--scorecard-trials 4`.


**Inner model:** `openai/gpt-5.4-mini`
**Outer model:** `anthropic/claude-opus-4-7`

Note: telecom tasks require stronger reasoning; switch to `anthropic/claude-sonnet-4-6`
as the inner model if gpt-5.4-mini success rates are below 30%.

### PRISM on τ-Telecom

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer prism \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-prism \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --generations 10 \
    --mutations-per-gen 3 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### BetterHarness on τ-Telecom

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-bh \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 10 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### PRISM-PO on τ-Telecom (prompt-only ablation)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer prism \
    --prism-prompt-only \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-prism-po \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --generations 10 \
    --mutations-per-gen 3 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### BH-PO on τ-Telecom (prompt-only ablation)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer better-harness \
    --bh-prompt-only \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-bh-po \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 10 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### MIPROv2 on τ-Telecom

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer miprov2 \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-miprov2 \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --miprov2-num-candidates 10 \
    --miprov2-num-trials 5 \
    --miprov2-minibatch-size 4 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### GEPA on τ-Telecom

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer gepa \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-gepa \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --gepa-max-metric-calls 160 \
    --gepa-reflection-minibatch-size 20 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

### τ-Telecom baseline (no optimization)

```bash
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-baseline \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 0 \
    --train-cases 20 \
    --holdout-cases 20 \
    --wall-time-s 600 \
    --max-tokens 100000 \
    --max-steps 200 \
    --scorecard-trials 4 \
    --no-human-approval
```

---

## Baseline-only scoring

To reproduce the baseline numbers (no optimizer, just score the default prompt):

```bash
# BFCL baseline
python -m agent_harness_optimizer.cli \
    --benchmark bfcl \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-bfcl-baseline \
    --max-iterations 0 \
    --train-cases 100 --holdout-cases 100 \
    --wall-time-s 300 --max-steps 100 \
    --no-human-approval

# tau-retail baseline
python -m agent_harness_optimizer.cli \
    --benchmark tau-retail \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-retail-baseline \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 0 \
    --train-cases 20 --holdout-cases 20 \
    --wall-time-s 600 --max-steps 200 --max-tokens 100000 \
    --no-human-approval

# tau-telecom baseline
python -m agent_harness_optimizer.cli \
    --benchmark tau-telecom \
    --optimizer better-harness \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-dir runs/paper-tau-telecom-baseline \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --max-iterations 0 \
    --train-cases 20 --holdout-cases 20 \
    --wall-time-s 600 --max-steps 200 --max-tokens 100000 \
    --no-human-approval
```

---

## Ablation and middleware-variant arms

The §6.3 ablation arms and the Table 2 -MW variants. Every arm
uses the identical protocol and budget as the corresponding main experiment —
only the listed flag differs. Ready-to-run configs with full CLI invocations
are in `configs/paper/`:

| Flag | Paper arm | Config |
|------|-----------|--------|
| `--prism-no-route`      | PRISM-NoRoute      | `{bfcl,tau_retail,tau_telecom}_prism_ablations.yaml` |
| `--prism-no-gate`       | PRISM-NoGate       | `{bfcl,tau_retail,tau_telecom}_prism_ablations.yaml` |
| `--prism-no-matrix`     | PRISM-NoMatrix     | `{bfcl,tau_retail,tau_telecom}_prism_ablations.yaml` |
| `--prism-no-constraint` | PRISM-NoConstraint | `{bfcl,tau_retail,tau_telecom}_prism_ablations.yaml` |
| `--prism-no-crossover`  | PRISM-NoCrossover  | `{bfcl,tau_retail,tau_telecom}_prism_ablations.yaml` |
| `--prism-population-cap 1` (or `10`) | PRISM frontier-retention ablation | `{bfcl,tau_retail,tau_telecom}_prism_ablations.yaml` |
| `--gepa-middleware`     | GEPA-MW            | `tau_{retail,telecom}_gepa_mw.yaml`      |
| `--miprov2-middleware`  | MIPROv2-MW         | `tau_{retail,telecom}_miprov2_mw.yaml`   |
| `--bh-prompt-only`      | BH-PO              | `{bfcl,tau_retail,tau_telecom}_bh_main.yaml` |

Run the ablation grid once per ablation flag. The crossover and
frontier-retention arms complete the component grid: crossover fires only when ≥2 children are complementary, so its
ablation isolates the merge step, and every generation records its crossover
status (fired / no complementary cases / disabled) in `gen_stats.json`.
Frontier retention is ablated via the population cap; note the cap only binds
under `--acceptance holdout_pareto` (the default `holdout_pass_rate`
acceptance keeps a single incumbent regardless), so the frontier arms must run
under Pareto acceptance.

For the revised metrics, `scripts/rellift.py` reports the budgeted-selection
RelLift95(B) estimator with its percentile-bootstrap CI and the subsampling
stability analysis (see "Extracting metrics" above for the exact command);
`scripts/compute_metrics.py` reports MeanLift, WorstLift, and repeat rates.

---

## Automated reproduction script

`scripts/reproduce_paper.py` provides a single entry point to run all or a
subset of paper experiments:

```bash
# Run all experiments (runs sequentially; use --parallel to run concurrently)
python scripts/reproduce_paper.py \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-root runs/paper \
    --tau-data-dir "${TAU2_DATA_DIR}"

# Run only BFCL experiments
python scripts/reproduce_paper.py \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-root runs/paper \
    --benchmarks bfcl

# Run only one optimizer across all benchmarks
python scripts/reproduce_paper.py \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-root runs/paper \
    --optimizers prism

# Dry run — print commands without executing
python scripts/reproduce_paper.py \
    --inner-model openai/gpt-5.4-mini \
    --outer-model anthropic/claude-opus-4-7 \
    --output-root runs/paper \
    --tau-data-dir "${TAU2_DATA_DIR}" \
    --dry-run
```

See `python scripts/reproduce_paper.py --help` for all options.

---

## Extracting metrics

The paper's aggregate tables are produced by two scripts in this archive:

```bash
# Table metrics (MeanLift, WorstLift, RR_0, cost) over a grid of runs,
# with the paper's exclusion criterion (final_train.reliability >= 0.5):
python scripts/compute_metrics.py --runs-dir runs --pattern "<arm>-*"

# RelLift_95 with its percentile-bootstrap CI and the subsampling
# sensitivity analysis (paper App C.6), one command per arm:
python scripts/rellift.py --runs-dir runs --pattern "<arm>-*" \
    --budget 4 --draws 5000 --ci-draws 1000 --subsample-sizes 16,12,8 \
    --seed 42 --json-out <arm>.json
```

For a quick per-run look at the raw reports:

After runs complete, extract the key paper metrics from `report.json`
(BetterHarness / MIPROv2 / GEPA) or `prism_report.json` (PRISM):

```bash
python3 - <<'EOF'
import json, sys
from pathlib import Path

run_dirs = {
    "bfcl/prism":          "runs/paper-bfcl-prism",
    "bfcl/bh":             "runs/paper-bfcl-bh",
    "bfcl/miprov2":        "runs/paper-bfcl-miprov2",
    "bfcl/gepa":           "runs/paper-bfcl-gepa",
    "tau-retail/prism":    "runs/paper-tau-retail-prism",
    "tau-retail/bh":       "runs/paper-tau-retail-bh",
    "tau-retail/miprov2":  "runs/paper-tau-retail-miprov2",
    "tau-retail/gepa":     "runs/paper-tau-retail-gepa",
    "tau-telecom/prism":   "runs/paper-tau-telecom-prism",
    "tau-telecom/bh":      "runs/paper-tau-telecom-bh",
    "tau-telecom/miprov2": "runs/paper-tau-telecom-miprov2",
    "tau-telecom/gepa":    "runs/paper-tau-telecom-gepa",
}

print(f"{'Experiment':<25} {'baseline':>10} {'final':>10} {'uplift':>10} {'holdout':>10} {'reliability':>12}")
print("-" * 80)
for label, d in run_dirs.items():
    d = Path(d)
    r = d / "prism_report.json" if (d / "prism_report.json").exists() else d / "report.json"
    if not r.exists():
        print(f"{label:<25} {'(missing)':>10}")
        continue
    data = json.loads(r.read_text())
    bt = data.get("baseline_train", {})
    bh = data.get("baseline_holdout", {})
    ft = data.get("final_train", {})
    fh = data.get("final_holdout", {})
    base = (bt.get("passed", 0) + bh.get("passed", 0)) / max(bt.get("total", 1) + bh.get("total", 1), 1)
    final = (ft.get("passed", 0) + fh.get("passed", 0)) / max(ft.get("total", 1) + fh.get("total", 1), 1)
    uplift = data.get("delta_combined", final - base)
    holdout_uplift = data.get("delta_holdout", 0)
    rel = ft.get("reliability", "?")
    print(f"{label:<25} {base:>10.3f} {final:>10.3f} {uplift:>+10.3f} {holdout_uplift:>+10.3f} {str(rel):>12}")
EOF
```

---

## Expected outputs

Each completed run writes:

```
runs/<name>/
  experiment_config.json        # CLI args used
  baseline/
    train/                      # per-case baseline scores
    holdout/
    summary.json
  iter-NNN/ (BetterHarness)     # or gen-NNN/ (PRISM)
    decision.json
  report.json                   # BetterHarness / MIPROv2 / GEPA final report
  prism_report.json             # PRISM final report (has holdout_series + per_candidate_history)
  final_diff.md                 # human-readable baseline → best diff
```

Key fields in `report.json` and `prism_report.json`:

| Field                 | Description                                          |
|-----------------------|------------------------------------------------------|
| `baseline_train`      | Pass rate before optimization (train split)          |
| `baseline_holdout`    | Pass rate before optimization (holdout split)        |
| `final_train`         | Pass rate after optimization (train split)           |
| `final_holdout`       | Pass rate after optimization (holdout split)         |
| `delta_combined`      | Net uplift = final_combined − baseline_combined      |
| `delta_holdout`       | Net uplift on holdout only (overfit guard)           |
| `holdout_series`      | Per-iteration/generation holdout pass rate (learning curve) |
| `per_candidate_history` | (PRISM only) All candidates evaluated            |

See `README.md` for complete field definitions.
