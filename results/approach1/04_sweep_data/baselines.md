# Training-free baselines on the correction candidates

Same questions and candidates as the v3 ranker; nothing is selected on test. Calls: generation calls per question (a 4B answer counts 2, an 8B answer 4). Passes add the scoring passes of methods that score candidates.

| Method | Calls | Passes | Test accuracy | Gain over greedy (95% interval) | Fixed / broken | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---:|---:|---:|---|---|---|---|---|---|
| greedy | 1.0 | 1.0 | 82.78% | +0.00 (+0.00 to +0.00) | +0 / -0 | +0.00 | +0.00 | +0.00 | +0.00 |
| model_4b | 2.0 | 2.0 | 85.85% | +3.07 (+2.36 to +3.74) | +711 / -419 | +4.68 | +0.12 | +2.12 | +5.90 |
| model_8b | 4.0 | 4.0 | 87.87% | +5.10 (+4.38 to +5.79) | +821 / -336 | +6.56 | +2.32 | +4.36 | +7.64 |
| reasoning_greedy_vote | 5.0 | 5.0 | 84.52% | +1.74 (+1.15 to +2.34) | +490 / -324 | +6.88 | -3.08 | +0.28 | +3.17 |
| reasoning_vote | 5.0 | 5.0 | 82.94% | +0.16 (-0.58 to +0.85) | +581 / -566 | +6.28 | -5.72 | -1.96 | +2.48 |
| self_consistency_5 | 5.0 | 5.0 | 82.93% | +0.15 (-0.07 to +0.39) | +73 / -59 | +0.12 | +0.32 | +0.08 | +0.05 |
| set_judge_best | 8.0 | 12.0 | 79.50% | -3.28 (-3.96 to -2.62) | +350 / -662 | -1.36 | -7.20 | -3.20 | -0.89 |
| set_token_best | 8.0 | 12.0 | 82.98% | +0.20 (+0.01 to +0.39) | +55 / -36 | +0.08 | +0.36 | +0.16 | +0.20 |
| set_vote | 8.0 | 8.0 | 84.86% | +2.08 (+1.54 to +2.63) | +433 / -235 | +6.48 | -1.56 | +0.88 | +2.63 |
| self_consistency_10 | 10.0 | 10.0 | 82.95% | +0.17 (-0.04 to +0.39) | +63 / -47 | +0.36 | +0.04 | +0.08 | +0.20 |
