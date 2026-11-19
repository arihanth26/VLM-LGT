# Candidate ranker v3: reasoning and crops in one ranker, with source and dataset features

Chosen on validation: `T1_k0_ground_cot_multi` with `ranker_v3_scalars_ds` (validation gain +3.41 points). Questions: {'train': 12000, 'val': 6420, 'test': 9517}. Baseline to beat: v2 `T1_k0_cot` scalars_judge, +1.84 on test.

| Dataset | Greedy | Chosen | Difference (95% interval) | Fixed / broken | p | Oracle |
|---|---:|---:|---|---|---:|---:|
| pooled | 82.78% | 86.38% | +3.60 (+3.10 to +4.09) | +473 / -130 | 0.0000 | 92.90% |
| chartqa_full | 76.48% | 83.48% | +7.00 (+5.76 to +8.24) | +212 / -37 | 0.0000 | 91.48% |
| textvqa | 81.52% | 84.20% | +2.68 (+1.80 to +3.60) | +96 / -29 | 0.0000 | 89.72% |
| docvqa | 87.04% | 88.40% | +1.36 (+0.68 to +2.04) | +54 / -20 | 0.0001 | 94.04% |
| scienceqa_img | 86.86% | 90.18% | +3.32 (+2.18 to +4.56) | +111 / -44 | 0.0000 | 97.17% |

## Per-dataset router (chosen on validation, untrained)

| Dataset | Configuration | Selector | Val accuracy | Test difference (95% interval) |
|---|---|---|---:|---|
| chartqa_full | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 79.64% | +7.12 (+5.88 to +8.36) |
| textvqa | `T1_k0_everything_multi` | `ranker_v3_full_ds` | 82.80% | +2.76 (+1.84 to +3.64) |
| docvqa | `T1_k0_ground_cot_multi` | `ranker_v2_full_judge` | 89.13% | +1.32 (+0.64 to +1.96) |
| scienceqa_img | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_nods` | 89.00% | +2.78 (+1.69 to +3.97) |
| pooled | | | | +3.53 (+3.03 to +4.02), accuracy 86.31% |

## Accuracy against cost (each row chosen on validation within the budget, read on test)

Cost is model generate calls per question (greedy, samples, correction sources; the grounded crops share one box call). Candidates is the mean number of distinct candidates, each needing one encoding pass and one judge pass (no generation).

| Budget (calls) | Configuration | Selector | Calls | Candidates | Val accuracy | Test accuracy | Test difference (95% interval) |
|---:|---|---|---:|---:|---:|---:|---|
| 2 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.26 | 83.52% | 85.34% | +2.56 (+2.03 to +3.06) |
| 3 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.26 | 83.52% | 85.34% | +2.56 (+2.03 to +3.06) |
| 4 | `T1_k0_cot` | `ranker_v3_scalars_nods` | 2 | 1.26 | 83.52% | 85.34% | +2.56 (+2.03 to +3.06) |
| 5 | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 5 | 1.55 | 84.00% | 86.00% | +3.23 (+2.73 to +3.72) |
| 6 | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 5 | 1.55 | 84.00% | 86.00% | +3.23 (+2.73 to +3.72) |
| 7 | `T1_k0_cot_multi` | `ranker_v3_scalars_ds` | 5 | 1.55 | 84.00% | 86.00% | +3.23 (+2.73 to +3.72) |
| 8 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 9 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 10 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 11 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 12 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 13 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 14 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 16 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |
| 19 | `T1_k0_ground_cot_multi` | `ranker_v3_scalars_ds` | 8 | 2.02 | 84.19% | 86.38% | +3.60 (+3.10 to +4.09) |

## Every configuration on test (pooled accuracy)

| Configuration | greedy | head_best | judge_best | ranker_v1 | ranker_v2_full | ranker_v2_full_judge | ranker_v2_scalars_judge | ranker_v3_full_ds | ranker_v3_scalars_ds | ranker_v3_full_nods | ranker_v3_scalars_nods | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1_k0_zoom | 82.78% | 80.51% | 80.33% | 83.83% | 83.86% | 83.89% | 83.76% | 83.85% | 83.64% | 83.86% | 83.64% | 89.47% |
| T1_k0_vcd | 82.78% | 83.02% | 82.80% | 83.11% | 83.11% | 83.16% | 83.01% | 83.08% | 82.97% | 83.06% | 82.98% | 84.46% |
| T1_k0_first_round | 82.78% | 80.56% | 80.39% | 83.76% | 83.89% | 83.94% | 83.86% | 83.88% | 83.68% | 83.81% | 83.70% | 90.22% |
| T1_k0_ground | 82.78% | 82.67% | 81.84% | 83.53% | 83.62% | 83.63% | 83.86% | 83.67% | 83.76% | 83.70% | 83.77% | 87.51% |
| T1_k0_cot | 82.78% | 83.99% | 83.70% | 83.72% | 83.94% | 83.89% | 84.62% | 84.80% | 85.36% | 84.85% | 85.34% | 88.35% |
| T1_k0_ground_cot | 82.78% | 83.06% | 82.12% | 83.84% | 84.08% | 84.04% | 84.21% | 84.76% | 85.97% | 84.84% | 85.69% | 91.12% |
| T1_k0_zoom_ground | 82.78% | 80.33% | 79.79% | 83.89% | 83.90% | 83.97% | 83.79% | 83.93% | 83.83% | 83.92% | 83.86% | 90.96% |
| T1_k0_everything | 82.78% | 80.35% | 79.87% | 83.92% | 84.02% | 84.05% | 83.93% | 84.60% | 85.90% | 84.70% | 85.77% | 93.70% |
| T1_k0_zoom_cot | 82.78% | 80.69% | 80.51% | 83.96% | 84.21% | 84.09% | 83.96% | 84.84% | 85.73% | 84.90% | 85.65% | 92.49% |
| T1_k0_zoom_ground_cot | 82.78% | 80.32% | 79.88% | 83.96% | 84.09% | 84.13% | 84.08% | 84.81% | 85.91% | 84.92% | 85.76% | 93.29% |
| T1_k0_cot_multi | 82.78% | 83.92% | 83.44% | 84.05% | 84.15% | 84.14% | 84.83% | 85.19% | 86.00% | 85.17% | 85.96% | 91.08% |
| T1_k0_ground_cot_multi | 82.78% | 82.89% | 81.75% | 84.01% | 84.23% | 84.23% | 84.51% | 84.95% | 86.38% | 84.98% | 86.14% | 92.90% |
| T1_k0_zoom_cot_multi | 82.78% | 80.53% | 80.29% | 84.07% | 84.21% | 84.24% | 84.43% | 85.06% | 86.13% | 85.08% | 85.92% | 93.99% |
| T1_k0_everything_multi | 82.78% | 80.15% | 79.57% | 83.94% | 84.00% | 84.10% | 84.11% | 84.74% | 86.24% | 84.83% | 86.01% | 94.84% |
| T1_k5_none | 82.78% | 83.35% | 82.64% | 83.69% | 83.65% | 83.77% | 83.63% | 83.64% | 83.58% | 83.73% | 83.56% | 88.12% |
| T1_k5_zoom | 82.78% | 80.48% | 80.32% | 83.86% | 83.91% | 84.03% | 83.99% | 83.91% | 83.73% | 83.99% | 83.86% | 91.86% |
| T1_k5_vcd | 82.78% | 83.32% | 82.60% | 83.51% | 83.53% | 83.65% | 83.43% | 83.56% | 83.48% | 83.63% | 83.52% | 88.43% |
| T1_k5_first_round | 82.78% | 80.49% | 80.33% | 83.85% | 83.83% | 84.01% | 83.94% | 83.85% | 83.67% | 83.85% | 83.77% | 91.99% |
| T1_k5_ground | 82.78% | 82.77% | 81.62% | 83.77% | 83.86% | 83.94% | 84.12% | 84.04% | 84.02% | 84.02% | 84.04% | 90.44% |
| T1_k5_cot | 82.78% | 83.49% | 82.76% | 83.91% | 84.11% | 84.20% | 84.86% | 85.14% | 85.68% | 85.07% | 85.60% | 91.35% |
| T1_k5_ground_cot | 82.78% | 82.79% | 81.62% | 83.98% | 84.01% | 84.06% | 84.29% | 84.75% | 85.92% | 84.76% | 85.73% | 92.83% |
| T1_k5_zoom_ground | 82.78% | 80.32% | 79.79% | 83.89% | 83.84% | 83.97% | 84.09% | 84.00% | 83.93% | 83.99% | 83.98% | 92.70% |
| T1_k5_everything | 82.78% | 80.14% | 79.73% | 83.84% | 83.86% | 84.01% | 83.98% | 84.50% | 85.80% | 84.52% | 85.71% | 94.45% |
| T1_k5_zoom_cot | 82.78% | 80.41% | 80.22% | 83.91% | 83.98% | 84.18% | 84.11% | 84.83% | 85.73% | 84.88% | 85.71% | 93.90% |
| T1_k5_zoom_ground_cot | 82.78% | 80.19% | 79.73% | 83.93% | 83.92% | 84.05% | 84.17% | 84.72% | 85.87% | 84.73% | 85.72% | 94.37% |
| T1_k5_cot_multi | 82.78% | 83.23% | 82.40% | 84.12% | 84.08% | 84.21% | 84.83% | 85.25% | 86.24% | 85.30% | 86.20% | 93.08% |
| T1_k5_ground_cot_multi | 82.78% | 82.55% | 81.28% | 84.01% | 84.06% | 84.10% | 84.61% | 85.02% | 86.34% | 84.98% | 86.16% | 94.12% |
| T1_k5_zoom_cot_multi | 82.78% | 80.30% | 79.99% | 83.98% | 84.07% | 84.13% | 84.40% | 84.93% | 86.10% | 85.05% | 86.00% | 94.98% |
| T1_k5_everything_multi | 82.78% | 80.06% | 79.51% | 83.90% | 83.89% | 84.05% | 84.23% | 84.68% | 86.11% | 84.74% | 85.90% | 95.37% |
