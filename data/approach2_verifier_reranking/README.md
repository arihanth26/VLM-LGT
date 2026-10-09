# Approach 2: Verifier Reranking

Start here if you are working on this approach. This file says where to put code,
data, and results, and what already exists for you to build on.

## What exists right now

This folder is a scaffold, not a working pipeline yet:

- `dataset_config.py` lists the training/grounding datasets (GQA, Visual Genome,
  RefCOCO) and the harder evaluation sets (ChartQA, AI2D, MathVista, V*Bench,
  HR-Bench, VSR). Ids are verified against the Hugging Face Hub, see
  [DATASETS.md](../../DATASETS.md) at the repo root.
- `download.py` and `preprocess.py` pull those datasets and normalize them into
  shared jsonl records (`image`, `question`, `answer`, `bbox`).
- `build_failure_pool.py` is where the verifier's training pool gets built, a pool
  that deliberately includes wrong candidates, not just correct ones. Its candidate
  sampling function (`sample_candidates`) is a placeholder right now, it returns
  empty predictions. Wiring this up to actually generate and score candidates from
  the policy model is the first real piece of work here.

Nothing for the policy's candidate generation, the verifier model itself, or
reranking/evaluation exists yet. That is what this team is adding.

## Where to put your code

Put it in this folder (`data/approach2_verifier_reranking/`), as flat scripts next
to the existing ones. This matches how Approach 1 is organized: its training,
sweep, and report scripts all live directly in `data/approach1_latent_self_correction/`,
not in a separate `models/approach1/` folder. Do the same here rather than inventing
a new top-level layout, so both approaches stay easy to find.

Some concrete examples of what will probably end up here:

- A script that samples K candidate traces from the policy model for a batch of
  examples (this is what `sample_candidates` in `build_failure_pool.py` needs).
- The verifier model definition and its training script.
- A reranking/inference script that runs the policy and verifier together and
  produces a final answer.
- An evaluation script that compares against the baselines in the project doc:
  always-crop, random-crop, single-pass without a verifier, plus CropVLM and CARES
  as the external comparison points.

If a script needs a new local working directory (checkpoints, cached candidates,
and so on), add it to `.gitignore` the same way `q_models/` and `q_features/` are
handled for Approach 1, so large local files do not get committed by accident.

## Accessing the model

Do not write your own model loading code, use what is already in `models/`
(shared by both approaches):

- `models/load_model.py` has `load_qwen3_vl(size)`, where `size` is `"2b"` or
  `"4b"`. Returns a `(model, processor)` pair. The 2B checkpoint is faster for
  iterating, the 4B one is closer to what full experiments will report on.
- `models/model_config.py` is the registry those sizes come from, if a new
  checkpoint ever needs adding it goes here, not hardcoded in a script.
- `models/run_inference.py` has `build_messages()` for the chat template, plus
  `run_inference()` (returns just the answer text) and `run_inference_detailed()`
  (returns an `InferenceResult` with the answer and token-level confidence stats).
  Both are reusable, the policy's candidate generation can call these directly, or
  `generate_with_confidence()` if you want the same confidence scoring Approach 1
  uses as a reference signal.
- `models/uncertainty.py` and `models/inference_result.py` hold the confidence
  math and the result structure Approach 1 built. Not required for Approach 2, but
  available if the verifier wants token confidence as one of its input features.

## Accessing and adding data

1. Run what is already here:

   ```
   python -m data.approach2_verifier_reranking.download
   python -m data.approach2_verifier_reranking.preprocess
   ```

   This populates `raw/` and `processed/` (both gitignored, everyone regenerates
   their own copy locally).

2. To add a new dataset, add a `DatasetEntry` to the right group in
   `dataset_config.py` (fields: `hf_repo_id`, `hf_config`, `source_url`, `notes`),
   then add a row to [DATASETS.md](../../DATASETS.md). Full instructions are in
   [CONTRIBUTING.md](../../CONTRIBUTING.md).

3. To build the verifier's failure-inclusive pool once candidate sampling is
   wired up:

   ```
   python -m data.approach2_verifier_reranking.build_failure_pool --dataset gqa --size 2b --k 4
   ```

## Where results go

Two places, same split Approach 1 uses:

- **Working results**, while you are iterating: put them under a `results/`
  subfolder here (`data/approach2_verifier_reranking/results/`), same as Approach 1
  does. It is already covered by `.gitignore` (tracked only via a `.gitkeep`), so
  local runs do not get committed.
- **Published results**, once something is ready for the rest of the team to see:
  copy it into `results/approach2/` at the repo root, as numbered files in the
  order the work happened (`01_...`, `02_...`), the same convention Approach 1
  uses under `results/approach1/`. Update `results/approach2/README.md` to point
  to each file and say what it shows. Look at `data/approach1_latent_self_correction/publish_results.py`
  for the pattern, a similar script here would copy from the working `results/`
  folder into `results/approach2/` automatically.

## Before opening a PR

```
python -m tests.test_config
```

This already checks every `DatasetEntry` in this folder's `dataset_config.py`,
so a bad id or missing source URL gets caught before a long download run. Once
there is real logic here (verifier training, reranking), add a `tests/test_*.py`
for it, no network calls, same as `tests/test_scoring_and_splits.py` does for
Approach 1, and list the new test in [CONTRIBUTING.md](../../CONTRIBUTING.md).
