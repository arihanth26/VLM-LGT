# Contributing

## Setup

1. Create a virtual environment and install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. If you hit Hugging Face rate limits, or need a model or dataset that
   later becomes gated, log in once with `huggingface-cli login` (needs a
   free Hugging Face account and access token).

## Adding a model size or checkpoint

Add it to `MODEL_VARIANTS` in [models/model_config.py](models/model_config.py)
rather than hardcoding a repo id somewhere else. Everything else under
`models/` reads from that registry, so a new entry there is picked up
everywhere automatically.

## Adding a dataset

1. Add a `DatasetEntry` for it to the right group in the approach's
   `dataset_config.py`. The fields you can set are `hf_repo_id`,
   `hf_config`, `source_url`, and `notes` (see
   [data/common/dataset_entry.py](data/common/dataset_entry.py)).
2. Run `download.py` for that group and confirm it actually pulls data.
3. If the raw schema does not match what `normalize_grounding_example` or
   `normalize_vqa_example` in `preprocess.py` expects, update the
   normalize function rather than special casing the dataset elsewhere.
4. Add a row for it to [DATASETS.md](DATASETS.md) so the next person does
   not have to re-verify the id from scratch.

For a dataset used by the correctness-head experiments, also add a metric to
[scoring.py](data/approach1_latent_self_correction/scoring.py) if it needs one,
a builder to
[build_qhead_splits.py](data/approach1_latent_self_correction/build_qhead_splits.py)
that writes train, val, and test splits in the shared schema, and an entry to
`QHEAD_DATASETS`. Then `bash slurm/run_qhead_pipeline.sh <name>` runs it end to end.
Group the held-out split by image or document so nothing is shared across splits.

## Before opening a PR

Run the smoke tests. They do not touch the network or download anything,
and take a couple of seconds:

```
python -m tests.test_config
python -m tests.test_scoring_and_splits
python -m tests.test_unified_q_head
```

## Code style

- Every file starts with a short header comment explaining what it is for.
- Every function gets a short comment or docstring explaining what it
  does, not how it does it.
- No em dashes in comments, docstrings, or documentation, use a period or
  comma instead.
- Keep dataset and model specifics in the config files, not scattered
  through download or inference logic.
- Prefer editing an existing script over adding a new one for a small
  variation, unless it is genuinely a different pipeline.
