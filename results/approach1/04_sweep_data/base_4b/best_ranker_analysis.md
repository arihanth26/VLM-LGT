# The best ranker: paired comparisons, where its answers come from, abstention

Test, 9517 questions. Best ranker: ground_cot_multi_esc8b_cot8b, no judge input. Routing setting chosen on validation: {'detector': 'pre_head', 'rule': 'confident', 'budget': 0.05}.

## Paired comparisons (row minus column, same questions)

| Comparison | Pooled | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| best ranker minus routing to 8B | +1.37 (+0.95 to +1.79), p = 4.4e-10 | +3.80 | +0.04 | +0.84 | +0.64 |
| best ranker gated 50% minus routing to 8B | +1.20 (+0.80 to +1.60), p = 2.1e-08 | +3.52 | +0.00 | +0.84 | +0.25 |
| best ranker minus 8B on every question | +1.58 (+1.09 to +2.05), p = 1.9e-11 | +4.04 | +0.48 | +0.24 | +1.54 |
| best ranker minus 2B-only ranker | +1.89 (+1.49 to +2.29), p = 3.1e-19 | +2.88 | +1.52 | +1.40 | +1.74 |
| routing to 8B minus 8B on every question | +0.21 (-0.20 to +0.59), p = 0.32 | +0.24 | +0.44 | -0.60 | +0.89 |
| 2B-only ranker minus 2B greedy | +1.72 (+1.38 to +2.06), p = 4.1e-22 | +3.04 | +1.16 | +1.12 | +1.54 |

## Accuracy of each method

| Method | Test accuracy |
|---|---:|
| 2B greedy | 85.84% |
| 2B-only ranker (no judge) | 87.56% |
| 8B on every question | 87.87% |
| routing to 8B (validation choice) | 88.08% |
| best ranker, no gate | 89.45% |
| best ranker, gated at 50% | 89.28% |

## Where the chosen answer comes from (no gate)

| Origin | Chosen | Right when chosen | Fixed | Broken |
|---|---:|---:|---:|---:|
| greedy kept | 75.8% | 91.05% | 0 | 0 |
| 2B corrections only | 7.6% | 91.10% | 38 | 7 |
| 8B only | 3.2% | 66.33% | 69 | 28 |
| 2B and 8B both | 13.5% | 84.97% | 351 | 79 |

Oracle: a right answer is among the candidates for 94.99% of questions; the ranker picks a right one in 94.17% of those; 527 questions have a right candidate the ranker misses.

## Answering only the most confident questions

| Answers ranked by | accuracy at 50% | accuracy at 80% | accuracy at 90% | accuracy at 100% | AURC |
|---|---:|---:|---:|---:|---:|
| best ranker's answer, by the ranker's probability | 98.05% | 93.34% | 92.32% | 89.45% | 0.0427 |
| best ranker's answer, by the greedy answer's probability | 97.63% | 93.81% | 92.17% | 89.45% | 0.0323 |
| 2B greedy answer, by its own probability | 97.27% | 92.33% | 89.84% | 85.84% | 0.0393 |
