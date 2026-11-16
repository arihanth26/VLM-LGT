# Candidate ranker v3: reasoning and crops in one ranker, with source and dataset features

Chosen on validation: `T1_k0_everything` with `ranker_v3_scalars_nods` (validation gain +3.05 points). Questions: {'train': 12000, 'val': 6420, 'test': 9517}. Baseline to beat: v2 `T1_k0_cot` scalars_judge, +1.84 on test.

| Dataset | Greedy | Chosen | Difference (95% interval) | Fixed / broken | p | Oracle |
|---|---:|---:|---|---|---:|---:|
| pooled | 82.78% | 85.77% | +2.99 (+2.47 to +3.51) | +467 / -182 | 0.0000 | 93.70% |
| chartqa_full | 76.48% | 82.28% | +5.80 (+4.60 to +7.00) | +194 / -49 | 0.0000 | 91.36% |
| textvqa | 81.52% | 83.80% | +2.28 (+1.24 to +3.28) | +110 / -53 | 0.0000 | 91.88% |
| docvqa | 87.04% | 88.44% | +1.40 (+0.60 to +2.16) | +64 / -29 | 0.0004 | 95.24% |
| scienceqa_img | 86.86% | 89.24% | +2.38 (+1.24 to +3.67) | +99 / -51 | 0.0001 | 96.93% |

## Per-dataset router (chosen on validation, untrained)

| Dataset | Configuration | Selector | Val accuracy | Test difference (95% interval) |
|---|---|---|---:|---|
| chartqa_full | `T1_k0_cot` | `ranker_v3_scalars_nods` | 78.28% | +6.60 (+5.36 to +7.84) |
| textvqa | `T1_k0_everything` | `ranker_v1` | 82.20% | +2.04 (+1.20 to +2.92) |
| docvqa | `T1_k0_zoom_ground` | `ranker_v3_full_nods` | 89.00% | +1.08 (+0.48 to +1.64) |
| scienceqa_img | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 88.73% | +2.38 (+1.29 to +3.57) |
| pooled | | | | +3.06 (+2.56 to +3.53), accuracy 85.84% |

## Accuracy against cost (each row chosen on validation within the budget, read on test)

Cost is model generate calls per question (greedy, samples, correction sources; the grounded crops share one box call). Candidates is the mean number of distinct candidates, each needing one encoding pass and one judge pass (no generation).

| Budget (calls) | Configuration | Selector | Calls | Candidates | Val accuracy | Test accuracy | Test difference (95% interval) |
|---:|---|---|---:|---:|---:|---:|---|
| 2 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.26 | 83.43% | 85.30% | +2.52 (+2.01 to +3.01) |
| 3 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.26 | 83.43% | 85.30% | +2.52 (+2.01 to +3.01) |
| 4 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.26 | 83.43% | 85.30% | +2.52 (+2.01 to +3.01) |
| 5 | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 5 | 1.76 | 83.72% | 85.74% | +2.96 (+2.46 to +3.47) |
| 6 | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 5 | 1.76 | 83.72% | 85.74% | +2.96 (+2.46 to +3.47) |
| 7 | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 5 | 1.76 | 83.72% | 85.74% | +2.96 (+2.46 to +3.47) |
| 8 | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 5 | 1.76 | 83.72% | 85.74% | +2.96 (+2.46 to +3.47) |
| 9 | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 5 | 1.76 | 83.72% | 85.74% | +2.96 (+2.46 to +3.47) |
| 10 | `T1_k0_ground_cot` | `ranker_v3_scalars_nods` | 5 | 1.76 | 83.72% | 85.74% | +2.96 (+2.46 to +3.47) |
| 11 | `T1_k0_everything` | `ranker_v3_scalars_nods` | 11 | 3.72 | 83.83% | 85.77% | +2.99 (+2.47 to +3.51) |
| 12 | `T1_k0_everything` | `ranker_v3_scalars_nods` | 11 | 3.72 | 83.83% | 85.77% | +2.99 (+2.47 to +3.51) |
| 13 | `T1_k0_everything` | `ranker_v3_scalars_nods` | 11 | 3.72 | 83.83% | 85.77% | +2.99 (+2.47 to +3.51) |
| 14 | `T1_k0_everything` | `ranker_v3_scalars_nods` | 11 | 3.72 | 83.83% | 85.77% | +2.99 (+2.47 to +3.51) |
| 16 | `T1_k0_everything` | `ranker_v3_scalars_nods` | 11 | 3.72 | 83.83% | 85.77% | +2.99 (+2.47 to +3.51) |

## Every configuration on test (pooled accuracy)

| Configuration | greedy | head_best | judge_best | ranker_v1 | ranker_v2_full | ranker_v2_full_judge | ranker_v2_scalars_judge | ranker_v3_full_ds | ranker_v3_scalars_ds | ranker_v3_full_nods | ranker_v3_scalars_nods | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1_k0_zoom | 82.78% | 80.51% | 80.33% | 83.83% | 83.86% | 83.89% | 83.76% | 83.84% | 83.64% | 83.79% | 83.62% | 89.47% |
| T1_k0_vcd | 82.78% | 83.02% | 82.80% | 83.11% | 83.11% | 83.16% | 83.01% | 83.12% | 83.06% | 83.15% | 83.07% | 84.46% |
| T1_k0_first_round | 82.78% | 80.56% | 80.39% | 83.76% | 83.89% | 83.94% | 83.86% | 83.82% | 83.79% | 83.81% | 83.69% | 90.22% |
| T1_k0_ground | 82.78% | 82.67% | 81.84% | 83.53% | 83.62% | 83.63% | 83.86% | 83.67% | 83.72% | 83.64% | 83.69% | 87.51% |
| T1_k0_cot | 82.78% | 83.99% | 83.70% | 83.72% | 83.94% | 83.89% | 84.62% | 84.54% | 85.39% | 84.53% | 85.30% | 88.35% |
| T1_k0_ground_cot | 82.78% | 83.06% | 82.12% | 83.84% | 84.08% | 84.04% | 84.21% | 84.65% | 85.86% | 84.68% | 85.74% | 91.12% |
| T1_k0_zoom_ground | 82.78% | 80.33% | 79.79% | 83.89% | 83.90% | 83.97% | 83.79% | 83.91% | 83.89% | 83.90% | 83.92% | 90.96% |
| T1_k0_everything | 82.78% | 80.35% | 79.87% | 83.92% | 84.02% | 84.05% | 83.93% | 84.47% | 85.95% | 84.56% | 85.77% | 93.70% |
| T1_k0_zoom_cot | 82.78% | 80.69% | 80.51% | 83.96% | 84.21% | 84.09% | 83.96% | 84.61% | 85.63% | 84.71% | 85.64% | 92.49% |
| T1_k0_zoom_ground_cot | 82.78% | 80.32% | 79.88% | 83.96% | 84.09% | 84.13% | 84.08% | 84.68% | 85.94% | 84.76% | 85.81% | 93.29% |
| T1_k5_none | 82.78% | 83.35% | 82.64% | 83.69% | 83.65% | 83.77% | 83.63% | 83.72% | 83.55% | 83.72% | 83.53% | 88.12% |
| T1_k5_zoom | 82.78% | 80.48% | 80.32% | 83.86% | 83.91% | 84.03% | 83.99% | 83.92% | 83.82% | 83.91% | 83.89% | 91.86% |
| T1_k5_vcd | 82.78% | 83.32% | 82.60% | 83.51% | 83.53% | 83.65% | 83.43% | 83.59% | 83.53% | 83.65% | 83.62% | 88.43% |
| T1_k5_first_round | 82.78% | 80.49% | 80.33% | 83.85% | 83.83% | 84.01% | 83.94% | 83.86% | 83.77% | 83.83% | 83.81% | 91.99% |
| T1_k5_ground | 82.78% | 82.77% | 81.62% | 83.77% | 83.86% | 83.94% | 84.12% | 83.92% | 83.97% | 83.90% | 84.10% | 90.44% |
| T1_k5_cot | 82.78% | 83.49% | 82.76% | 83.91% | 84.11% | 84.20% | 84.86% | 84.84% | 85.56% | 84.79% | 85.50% | 91.35% |
| T1_k5_ground_cot | 82.78% | 82.79% | 81.62% | 83.98% | 84.01% | 84.06% | 84.29% | 84.56% | 85.78% | 84.60% | 85.73% | 92.83% |
| T1_k5_zoom_ground | 82.78% | 80.32% | 79.79% | 83.89% | 83.84% | 83.97% | 84.09% | 83.92% | 83.91% | 83.90% | 83.97% | 92.70% |
| T1_k5_everything | 82.78% | 80.14% | 79.73% | 83.84% | 83.86% | 84.01% | 83.98% | 84.33% | 85.87% | 84.38% | 85.77% | 94.45% |
| T1_k5_zoom_cot | 82.78% | 80.41% | 80.22% | 83.91% | 83.98% | 84.18% | 84.11% | 84.60% | 85.67% | 84.62% | 85.63% | 93.90% |
| T1_k5_zoom_ground_cot | 82.78% | 80.19% | 79.73% | 83.93% | 83.92% | 84.05% | 84.17% | 84.51% | 85.85% | 84.60% | 85.73% | 94.37% |
