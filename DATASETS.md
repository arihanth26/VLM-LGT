# Dataset reference

This is the source of truth for every dataset id used under `data/`. Each
`DatasetEntry` in a `dataset_config.py` file carries its own `source_url`
pointing back to the row below it, and this file is the place to update
when a dataset gets renamed, moved, or re-licensed on the Hub.

Last verified against the Hugging Face Hub: 2026-09-28. An earlier version
of the dataset configs had two wrong entries: the bare `visual_genome` repo
id (deprecated) and Ref-L4 marked as requiring a manual download (it does
not). Both are corrected below.

## Shared training and grounding datasets

| Name | Hugging Face id | Config | Notes |
|---|---|---|---|
| GQA | `lmms-lab/GQA` | - | Formatted for lmms-eval, splits are named like `train_balanced_instructions`, not `train`. |
| Visual Genome | `ranjaykrishna/visual_genome` | `region_descriptions_v1.2.0` | The bare `visual_genome` id is deprecated and 404s. Use `region_descriptions_v1.2.0` to get bounding boxes; other configs (`objects`, `question_answers`) exist but do not carry them. |
| RefCOCO | `lmms-lab/RefCOCO` | - | |
| RefCOCOg | `lmms-lab/RefCOCOg` | - | Only has `val`/`test` splits, no `train`. |

## Approach 1 testbed (Self-Correction Mirage split)

| Name | Hugging Face id | Notes |
|---|---|---|
| RefCOCOg | `lmms-lab/RefCOCOg` | Same caveat as above, no train split on the Hub. |
| Ref-Adv-S | not on the Hub | Manual download from [dddraxxx/Ref-Adv](https://github.com/dddraxxx/Ref-Adv) (ICLR 2026). Use the publicly released `Ref-Adv-s` subset, 1,142 cases with evaluation code and model predictions. |
| Ref-L4 | `JierunChen/Ref-L4` | Full release has 45,341 referring expressions across 9,735 images. The Mirage study's exact 2400/510/505 split is a subsample of this, RefCOCOg, and Ref-Adv-S, not any of the raw datasets used whole. |

## Harder and out-of-distribution evaluation sets

| Name | Hugging Face id | Notes |
|---|---|---|
| ChartQA | `lmms-lab/ChartQA` | |
| AI2D | `lmms-lab/ai2d` | |
| MathVista | `AI4Math/MathVista` | Not gated. License permits commercial use as a test set only, do not train on it (fits its role here as evaluation-only). |
| V*Bench | `craigwu/vstar_bench` | Small, 191 examples in a single `test` split (115 attribute recognition, 76 spatial relationship). |
| HR-Bench | `DreamMr/HR-Bench` | Ships as TSV files (`hr_bench_4k.tsv`, `hr_bench_8k.tsv`) rather than the usual parquet layout. A plain `load_dataset` call may need an explicit config, check the dataset card if `download.py` errors on this one. |
| VSR | `cambridgeltl/vsr_random` | Splits are `train`/`dev`/`test`, not `train`/`validation`/`test`. |

## Models

| Name | Hugging Face id |
|---|---|
| Qwen3-VL 2B | `Qwen/Qwen3-VL-2B-Instruct` |
| Qwen3-VL 4B | `Qwen/Qwen3-VL-4B-Instruct` |

## Keeping this current

If `download.py` reports an error for a dataset, or `datasets.load_dataset`
starts raising a "config required" error, check that dataset's page on the
Hub before changing any code. Dataset owners rename and restructure repos
over time, the id that worked when this file was last verified is not
guaranteed to stay that way. When you fix an entry, update both the
`DatasetEntry` in the relevant `dataset_config.py` and the matching row
here, and bump the "last verified" date at the top of this file, so the
two do not drift apart again.
