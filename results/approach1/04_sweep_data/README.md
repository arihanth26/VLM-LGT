# Published data behind the reports

Machine-readable copies of every result the reports use, so each number can be checked without rerunning the models.
`.md` files are readable tables of the `.json` next to them; `*_jobs.txt` files record the Slurm job ids of each run.
`<ds>` is a dataset (`chartqa_full`, `textvqa`, `docvqa`, `scienceqa_img`). Every choice in these files was made on
validation; test was read once.

## Detection: correctness heads (reports 01–09)

| Files | Content |
|---|---|
| `chartqa_original_validation_runs.json` | The 72 validation runs of the original ChartQA study (report 01) |
| `<ds>_separate_heads_sweep.json`, `<ds>_unified_head_sweep.json`, `<ds>_analysis.json` | Per-dataset head sweeps and analyses (reports 02–03) |
| `mixture_*_heads_sweep.json`, `mixture_*_sequential.json`, `mixture_analysis.json` | Heads trained on the dataset mixture; sequential training (report 05) |
| `corruption_*` | The visual-grounding loss on corrupted images (report 06) |
| `pre_head_sweep.json` | Pre-generation head sweep (report 10) |

## Correction (reports 07–12)

| Files | Content |
|---|---|
| `correction_mixture_test_2b_*_limit100.*`, `correction_full_mixture_test_2b_*_full.*` | Prompted retry, best-of-five, deeper answers: first 100 questions and full test split (reports 07–08) |
| `candidate_search.json` | Pools, temperatures, best-of-k (report 10) |
| `candidate_ranker.json`, `ranker_analysis.json` | Ranker v1 and its analysis (report 11) |
| `candidate_ranker_v2.json` | Ranker v2 (report 11) |
| `candidate_ranker_v3.*`, `_ablation.*`, `_retrained.*`, `_multi.*` | Ranker v3, without the dataset input, retrained, with sampled reasoning (report 12) |
| `gating_v3.*`, `gating_v3_multi.*` | Gating of those rankers (report 12) |

## Larger models and further evidence (report 13)

| Files | Content | Report 13 section |
|---|---|---|
| `escalation.*` | Routing to 4B, 8B and enlarged images (no ranker) | 5 |
| `cascade.*` | 2B → 4B → 8B against 2B → 8B | 8 |
| `baselines.*` | Training-free baselines on the same candidates | 9 |
| `candidate_ranker_v3_multi_esc*.json` | Rankers with the 8B answers (`_esc`), with 8B reasoning (`_cot8b`), without the judge (`_drop_judge`) | 3, 6 |
| `gating_v3_multi_esc*.*`, `gating_v3_multi_drop_judge.*` | Their gating; **the accuracy–cost frontier is `gating_v3_multi_esc_cot8b_drop_judge.md`** | 4, 6 |
| `gating_v3_*_seconds.*` | The same gating with measured seconds as a second cost | 4 |
| `best_ranker_analysis.*` | Paired comparisons with routing and 8B, answer origins, oracle gap, abstention, examples | 7 |
| `candidate_ranker_v3_multi_drop_<inputs>.json`, `..._esc_cot8b_drop_<inputs>_judge.json`, `_states`, `_epochs20` | Input ablations and variants | 10 |
| `*_seeds<n>.json`, `*_holdout_<ds>.json`, `*_train<n>.json` | Seeds, leave-one-dataset-out, training-set size | 11 |
| `escalation.*` (fixes `internvl3_8b`, `idefics3_8b`), `*_families_drop_judge.*` | Helpers from other model families: routing, ranker and gating (with seconds) | 5, 6 |
| `unseen.*` | The frozen pipeline on AI2D and InfographicVQA | 12 |
| `base_4b/`, `base_8b/` | Qwen3-VL-4B and 8B as the base model (rankers; the 4B best-ranker analysis) | 13 |
| `latency.json`, `latency_H100_families.json`, `latency_A100-80GB.json`, `latency_L40S.json` | Seconds per component on H100 (the two other-family helpers in a separate run), A100 and L40S | 14 |
