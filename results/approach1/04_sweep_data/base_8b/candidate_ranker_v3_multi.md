# Candidate ranker v3: reasoning and crops in one ranker, with source and dataset features

Chosen on validation: `T1_k0_everything` with `ranker_v3_scalars_ds` (validation gain +0.42 points). Questions: {'train': 6000, 'val': 6420, 'test': 9517}. Baseline to beat: v2 `T1_k0_cot` scalars_judge, +1.84 on test.

| Dataset | Greedy | Chosen | Difference (95% interval) | Fixed / broken | p | Oracle |
|---|---:|---:|---|---|---:|---:|
| pooled | 87.87% | 88.28% | +0.41 (+0.19 to +0.62) | +72 / -33 | 0.0002 | 94.90% |
| chartqa_full | 83.04% | 83.40% | +0.36 (+0.04 to +0.68) | +13 / -4 | 0.0490 | 92.16% |
| textvqa | 83.84% | 84.52% | +0.68 (+0.20 to +1.20) | +29 / -12 | 0.0115 | 93.12% |
| docvqa | 91.40% | 91.92% | +0.52 (+0.20 to +0.88) | +17 / -4 | 0.0072 | 96.16% |
| scienceqa_img | 94.50% | 94.50% | +0.00 (-0.50 to +0.50) | +13 / -13 | 1.0000 | 98.96% |

## Per-dataset router (chosen on validation, untrained)

| Dataset | Configuration | Selector | Val accuracy | Test difference (95% interval) |
|---|---|---|---:|---|
| chartqa_full | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 79.22% | +0.52 (+0.20 to +0.84) |
| textvqa | `T1_k0_first_round` | `ranker_v3_scalars_ds` | 82.33% | +0.56 (+0.20 to +0.92) |
| docvqa | `T1_k5_ground_cot` | `ranker_v3_scalars_ds` | 92.73% | +0.56 (+0.24 to +0.92) |
| scienceqa_img | `T1_k0_cot` | `vote_then_head` | 95.80% | +1.04 (+0.10 to +1.98) |
| pooled | | | | +0.65 (+0.39 to +0.90), accuracy 88.53% |

## Accuracy against cost (each row chosen on validation within the budget, read on test)

Cost is model generate calls per question (greedy, samples, correction sources; the grounded crops share one box call). Candidates is the mean number of distinct candidates, each needing one encoding pass and one judge pass (no generation).

| Budget (calls) | Configuration | Selector | Calls | Candidates | Val accuracy | Test accuracy | Test difference (95% interval) |
|---:|---|---|---:|---:|---:|---:|---|
| 2 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.29 | 86.34% | 88.09% | +0.22 (+0.07 to +0.36) |
| 3 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.29 | 86.34% | 88.09% | +0.22 (+0.07 to +0.36) |
| 4 | `T1_k0_ground` | `ranker_v3_scalars_nods` | 4 | 1.52 | 86.34% | 88.19% | +0.32 (+0.16 to +0.46) |
| 5 | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 5 | 1.70 | 86.43% | 88.24% | +0.37 (+0.17 to +0.56) |
| 6 | `T1_k0_zoom_cot` | `ranker_v3_scalars_ds` | 6 | 3.29 | 86.48% | 88.19% | +0.32 (+0.13 to +0.49) |
| 7 | `T1_k0_zoom_cot` | `ranker_v3_scalars_ds` | 6 | 3.29 | 86.48% | 88.19% | +0.32 (+0.13 to +0.49) |
| 8 | `T1_k0_zoom_cot` | `ranker_v3_scalars_ds` | 6 | 3.29 | 86.48% | 88.19% | +0.32 (+0.13 to +0.49) |
| 9 | `T1_k0_zoom_ground_cot` | `ranker_v3_scalars_ds` | 9 | 3.61 | 86.56% | 88.34% | +0.46 (+0.25 to +0.66) |
| 10 | `T1_k0_zoom_ground_cot` | `ranker_v3_scalars_ds` | 9 | 3.61 | 86.56% | 88.34% | +0.46 (+0.25 to +0.66) |
| 11 | `T1_k0_everything` | `ranker_v3_scalars_ds` | 11 | 3.65 | 86.57% | 88.28% | +0.41 (+0.19 to +0.62) |
| 12 | `T1_k0_everything` | `ranker_v3_scalars_ds` | 11 | 3.65 | 86.57% | 88.28% | +0.41 (+0.19 to +0.62) |
| 13 | `T1_k0_everything` | `ranker_v3_scalars_ds` | 11 | 3.65 | 86.57% | 88.28% | +0.41 (+0.19 to +0.62) |
| 14 | `T1_k0_everything` | `ranker_v3_scalars_ds` | 11 | 3.65 | 86.57% | 88.28% | +0.41 (+0.19 to +0.62) |
| 16 | `T1_k0_everything` | `ranker_v3_scalars_ds` | 11 | 3.65 | 86.57% | 88.28% | +0.41 (+0.19 to +0.62) |
| 19 | `T1_k0_everything` | `ranker_v3_scalars_ds` | 11 | 3.65 | 86.57% | 88.28% | +0.41 (+0.19 to +0.62) |

## Every configuration on test (pooled accuracy)

| Configuration | greedy | head_best | judge_best | ranker_v3_scalars_ds | ranker_v3_scalars_nods | oracle |
|---|---:|---:|---:|---:|---:|---:|
| T1_k0_zoom | 87.87% | 87.87% | 85.97% | 88.05% | 88.05% | 92.42% |
| T1_k0_vcd | 87.87% | 87.87% | 87.87% | 88.01% | 88.00% | 88.80% |
| T1_k0_first_round | 87.87% | 87.87% | 85.95% | 88.16% | 88.15% | 92.84% |
| T1_k0_ground | 87.87% | 87.87% | 87.02% | 88.17% | 88.19% | 91.31% |
| T1_k0_cot | 87.87% | 87.87% | 86.90% | 88.04% | 88.09% | 91.37% |
| T1_k0_ground_cot | 87.87% | 87.87% | 86.20% | 88.28% | 88.24% | 93.19% |
| T1_k0_zoom_ground | 87.87% | 87.87% | 85.54% | 88.24% | 88.17% | 93.64% |
| T1_k0_everything | 87.87% | 87.87% | 84.85% | 88.28% | 88.36% | 94.90% |
| T1_k0_zoom_cot | 87.87% | 87.87% | 85.24% | 88.19% | 88.15% | 94.13% |
| T1_k0_zoom_ground_cot | 87.87% | 87.87% | 84.87% | 88.34% | 88.36% | 94.67% |
| T1_k0_cot_multi | 87.87% | 87.87% | 86.28% | 88.03% | 88.12% | 92.59% |
| T1_k0_ground_cot_multi | 87.87% | 87.87% | 85.70% | 88.22% | 88.22% | 93.91% |
| T1_k0_zoom_cot_multi | 87.87% | 87.87% | 84.85% | 88.18% | 88.13% | 94.78% |
| T1_k0_everything_multi | 87.87% | 87.87% | 84.50% | 88.29% | 88.36% | 95.39% |
| T1_k5_none | 87.87% | 87.87% | 87.73% | 88.20% | 88.16% | 90.28% |
| T1_k5_zoom | 87.87% | 87.87% | 85.98% | 88.21% | 88.16% | 93.41% |
| T1_k5_vcd | 87.87% | 87.87% | 87.75% | 88.15% | 88.17% | 90.52% |
| T1_k5_first_round | 87.87% | 87.87% | 85.93% | 88.15% | 88.13% | 93.51% |
| T1_k5_ground | 87.87% | 87.87% | 86.96% | 88.22% | 88.25% | 92.51% |
| T1_k5_cot | 87.87% | 87.87% | 86.70% | 88.14% | 88.17% | 92.67% |
| T1_k5_ground_cot | 87.87% | 87.87% | 86.14% | 88.19% | 88.20% | 93.99% |
| T1_k5_zoom_ground | 87.87% | 87.87% | 85.53% | 88.20% | 88.20% | 94.29% |
| T1_k5_everything | 87.87% | 87.87% | 84.86% | 88.18% | 88.17% | 95.20% |
| T1_k5_zoom_cot | 87.87% | 87.87% | 85.19% | 88.11% | 88.16% | 94.76% |
| T1_k5_zoom_ground_cot | 87.87% | 87.87% | 84.88% | 88.20% | 88.21% | 95.15% |
| T1_k5_cot_multi | 87.87% | 87.87% | 86.03% | 88.14% | 88.15% | 93.56% |
| T1_k5_ground_cot_multi | 87.87% | 87.87% | 85.55% | 88.20% | 88.18% | 94.58% |
| T1_k5_zoom_cot_multi | 87.87% | 87.87% | 84.75% | 88.08% | 88.17% | 95.27% |
| T1_k5_everything_multi | 87.87% | 87.87% | 84.48% | 88.17% | 88.19% | 95.65% |
