# The Yes-Bias in LLM Reasoning

Code and scripts for experiments in *The Yes-Bias in LLM Reasoning* (ICML 2026 submission).

This repository is **not packaged for reproduction yet**: datasets are not included, requirements are incomplete, and several scripts expect private API access and local data. The goal is to document what each part does and how it maps to the paper’s ideas.

## Core ideas (short)

- **Sycophancy in reasoning**: LLMs shift judgments to match a user’s framing even when it conflicts with correctness.
- **Three settings**:
  - **Solution grading**: compare grades under neutral vs. biased prompts.
  - **Code grading**: compare verdicts under positive vs. negative framing.
  - **Synthetic fake-tasks (iGSM)**: test whether models solve contradictions they can later identify.
- **Mitigation and causes**:
  - **Steering vectors**: inference-time reduction of sycophancy on fake-tasks.
  - **Preference optimization (DPO/SimPO)**: controlled post-training that can amplify sycophancy.

## Repository layout (what each part does)

### `solution_grading/`
Math solution grading on hard problems (HLE).

- `bench_gen.py` — builds the grading benchmark:
  - pulls HLE Math (text-only) via `datasets`
  - assigns generator models to produce solution attempts
  - writes the problem–solution pairs for later grading
- `benchmarking.py` — runs the grading experiment:
  - applies **neutral** vs **negatively biased** grading prompts
  - measures grade shifts as a sycophancy signal
  - produces summary tables/plots
- `benchmarking_confidence_intervals.py` — stability / interval estimation for grading metrics.
- `prompts.json` — prompt templates used by the grading scripts.
- `api.json` — model list and API endpoints/tokens used by the grading scripts.

### `code_bench/`
Codeforces-based code grading with bias.

- `run.py` — dataset loading + solution generation:
  - pulls tasks from Codeforces
  - generates “weird” solutions (hard to judge)
  - stores raw model outputs
- `evaluate.py` — evaluates generated solutions:
  - runs **positive** vs **negative** grading prompts
  - extracts verdicts (CORRECT/INCORRECT)
  - computes verdict-flip sycophancy rates
- `get_res.py` — aggregates final numbers for the paper tables.
- `find_sycophancy_sample.py` — finds representative sycophancy cases.
- `manual_evaluate.py` — helper for spot-checking judge reliability.
- `compare_manual_to_basic.py` — compares manual judgments to automated judge outputs.
- `extract_code.py` — extracts code blocks from model outputs.
- `prompt.txt` — solution-generation prompt template.
- `config_run.json` / `config_evaluate.json` / `config_test.json` — run configs and model lists.
- `utils/` — API calls and helpers.

### `igsm_bench/`
Synthetic contradictory math tasks built from iGSM.

- `generate_data.py` — creates consistent + contradictory task pairs by graph perturbation.
- `validate.py` — runs solve + contradiction-detection queries and labels outcomes.
- `prompts/` — prompt sets for iGSM tasks and evaluation.
- `api.json` — model list and API endpoints/tokens.

### `steering/`
Inference-time steering vector experiments for the fake-task setting.

- `compute_steering_vector.py` — computes a steering vector from contrastive generations.
- `gen_contrastive.py` — builds contrastive prompt pairs used to learn the steering direction.
- `train_test_split.py` — prepares splits used for steering evaluation.
- `logs/` — raw logs, task dumps, and intermediate outputs for steering runs.

### `rlhf/`
Post-training experiments for testing whether preference optimization increases sycophancy.

- `torchtune_config_dpo.yaml` / `torchtune_config_simpo.yaml` — training configs for DPO and SimPO.
- `benchmark_sycophancy.py` — evaluates sycophancy after post-training (paired with the grading benchmark).

## Data and private dependencies

This repo expects private API keys and external datasets (HLE, Codeforces, iGSM variants). Those are **not included**. If you want to run anything, you’ll need to provide data and update configs accordingly.
