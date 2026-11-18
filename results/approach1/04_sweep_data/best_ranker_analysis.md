# The best ranker: paired comparisons, where its answers come from, abstention

Test, 9517 questions. Best ranker: ground_cot_multi_esc8b_cot8b, no judge input. Routing setting chosen on validation: {'detector': 'token', 'rule': 'confident', 'budget': 0.65}.

## Paired comparisons (row minus column, same questions)

| Comparison | Pooled | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| best ranker minus routing to 8B | +1.24 (+0.78 to +1.68), p = 1.2e-07 | +4.32 | +0.32 | -0.36 | +0.55 |
| best ranker gated 50% minus routing to 8B | +1.02 (+0.56 to +1.46), p = 1.3e-05 | +3.76 | +0.32 | -0.36 | +0.20 |
| best ranker minus 8B on every question | +1.66 (+1.15 to +2.14), p = 2.4e-11 | +4.32 | +1.68 | -0.24 | +0.69 |
| best ranker minus 2B-only ranker | +3.17 (+2.70 to +3.64), p = 6.9e-42 | +3.88 | +1.28 | +2.88 | +5.01 |
| routing to 8B minus 8B on every question | +0.42 (+0.07 to +0.74), p = 0.013 | +0.00 | +1.36 | +0.12 | +0.15 |
| 2B-only ranker minus 2B greedy | +3.58 (+3.08 to +4.09), p = 3.8e-46 | +7.00 | +2.72 | +1.24 | +3.32 |

## Accuracy of each method

| Method | Test accuracy |
|---|---:|
| 2B greedy | 82.78% |
| 2B-only ranker (no judge) | 86.36% |
| 8B on every question | 87.87% |
| routing to 8B (validation choice) | 88.29% |
| best ranker, no gate | 89.53% |
| best ranker, gated at 50% | 89.31% |

## Where the chosen answer comes from (no gate)

| Origin | Chosen | Right when chosen | Fixed | Broken |
|---|---:|---:|---:|---:|
| greedy kept | 82.7% | 93.12% | 0 | 0 |
| 2B corrections only | 2.2% | 55.83% | 48 | 25 |
| 8B only | 4.1% | 60.00% | 149 | 32 |
| 2B and 8B both | 11.1% | 80.25% | 594 | 91 |

Oracle: a right answer is among the candidates for 95.52% of questions; the ranker picks a right one in 93.73% of those; 570 questions have a right candidate the ranker misses.

## Answering only the most confident questions

| Answers ranked by | accuracy at 50% | accuracy at 80% | accuracy at 90% | accuracy at 100% | AURC |
|---|---:|---:|---:|---:|---:|
| best ranker's answer, by the ranker's probability | 98.53% | 94.96% | 93.10% | 89.53% | 0.0327 |
| best ranker's answer, by the greedy answer's probability | 97.54% | 93.51% | 91.62% | 89.53% | 0.0361 |
| 2B greedy answer, by its own probability | 97.12% | 90.82% | 87.59% | 82.78% | 0.0491 |
