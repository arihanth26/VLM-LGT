# VLM-LGT: Look, Ground, Then Think

This repo holds the code for our project on teaching vision-language models to notice when
their visual evidence is weak and do something about it before answering, instead of just
guessing.

We are testing two ways to do this:

1. **Latent Self-Correction.** The model scores its own uncertainty and, when it is not
   confident, updates its own internal representation and re-answers, without necessarily
   fetching new pixels.
2. **Verifier Reranking.** The model produces several candidate answers and a separately
   trained verifier picks the best one.

Both approaches build on Qwen3-VL and share the same broad evaluation goals, but the code for
each one lives in its own folder so they can be developed independently.

## Repo layout

```
VLM-LGT/
├── models/                                    Scripts for loading and running Qwen3-VL (2B and 4B)
│   ├── model_config.py                        Registry of model sizes and their Hugging Face repo ids
│   ├── load_model.py                          Loads a model and processor for a given size
│   └── run_inference.py                       CLI script to run the model on an image and a question
│
├── data/
│   ├── common/
│   │   ├── utils.py                           Shared helpers used by both approaches (jsonl read/write, etc)
│   │   └── dataset_entry.py                   Shared DatasetEntry structure (repo id, config, source, notes)
│   │
│   ├── approach1_latent_self_correction/       Datasets for the self-correction approach
│   │   ├── dataset_config.py                  List of datasets and where they come from
│   │   ├── download.py                        Downloads the raw datasets
│   │   ├── preprocess.py                      Converts raw data into a shared training format
│   │   ├── raw/                               Downloaded datasets land here (not tracked in git)
│   │   └── processed/                         Preprocessed jsonl output lands here (not tracked in git)
│   │
│   └── approach2_verifier_reranking/          Datasets for the verifier reranking approach
│       ├── dataset_config.py
│       ├── download.py
│       ├── preprocess.py
│       ├── build_failure_pool.py              Builds the failure-inclusive pool the verifier trains on
│       ├── raw/
│       ├── processed/
│       └── failure_pool/
│
├── tests/
│   └── test_config.py                         Smoke tests for the model and dataset configs, no network needed
│
├── requirements.txt
├── DATASETS.md                                Reference table of every dataset id, with source links and known quirks
├── CONTRIBUTING.md                            How to add a model, add a dataset, and code style expectations
└── README.md
```

## Getting started

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Try the model:

   ```
   python -m models.run_inference --size 2b --image path/to/image.jpg --question "What is in this image?"
   ```

   Swap `--size 2b` for `--size 4b` to compare the larger checkpoint on the same input. Both
   sizes share the same loading code in `models/load_model.py`, so anyone on the project can
   switch between them without changing anything else.

### Working with both model sizes

The project standardizes on two Qwen3-VL checkpoints, registered in
`models/model_config.py`:

- `2b` (`Qwen/Qwen3-VL-2B-Instruct`), good for fast local iteration when you are testing an
  idea or debugging a script.
- `4b` (`Qwen/Qwen3-VL-4B-Instruct`), closer to what full experiments will actually run on.

Everyone should default to `2b` while developing, then confirm results hold on `4b` before
reporting them. Both sizes go through the same `load_qwen3_vl(size)` function, so any script
that takes a `--size` flag works with either one without code changes:

```
python -m models.run_inference --size 2b --image example.jpg --question "What is happening here?"
python -m models.run_inference --size 4b --image example.jpg --question "What is happening here?"
```

Run both on the same input when you want to check whether a behavior is specific to model
scale before you build on top of it.

3. Pull and prepare data for an approach:

   ```
   python -m data.approach1_latent_self_correction.download
   python -m data.approach1_latent_self_correction.preprocess
   ```

   The same pattern applies under `data/approach2_verifier_reranking/`.

4. Check that the configs are still sane before a long download run:

   ```
   python -m tests.test_config
   ```

## Notes

- Model loading defaults to bfloat16 and picks a GPU automatically if one is available,
  falling back to CPU otherwise.
- Every dataset id in `dataset_config.py` was checked against the live Hugging Face Hub.
  See [DATASETS.md](DATASETS.md) for the full table, the verification date, and known
  quirks (split naming, required configs, file formats). If `download.py` errors on a
  dataset, check that file before assuming the code is wrong, dataset owners do rename
  and restructure repos over time.
- Ref-Adv-S, used in the Approach 1 testbed, is not on the Hugging Face Hub and needs to
  be downloaded manually from its GitHub repo. `download.py` will tell you if a local
  copy is missing. Ref-L4 is on the Hub and downloads automatically.
- If you hit Hugging Face rate limits, log in once with `huggingface-cli login`.
- `build_failure_pool.py` under Approach 2 is a starting scaffold. The exact candidate
  sampling format is still an open decision, so the sampling function is left as a
  placeholder until that is settled.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a model size, add a dataset, and
the code style this repo follows. Short version: if you are experimenting with a
different model size or dataset, add it to the relevant config file rather than
hardcoding paths in a script, so everyone else picks it up automatically.
