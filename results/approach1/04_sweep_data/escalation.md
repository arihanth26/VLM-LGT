# Escalation study: our detector, new fixes

Every setting is chosen on validation and read once on test. A budget sets a fixed threshold on the detector score from validation; on test each question is routed on its own score. Accuracy is pooled over the four datasets.

## Fix: 8b

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 87.87% (+5.10). Oracle (either answer right): 91.40%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget |
|---|---|---|---:|---:|---:|---|---|---|
| best detector and rule at a 20% budget | token | replace | 20% / 19.4% | 1.78 | 86.27% | +3.49 (+2.99 to +4.00) | +478 / -146 | +2.60 (+2.09 to +3.07) |
| best detector and rule at a 30% budget | after_head | confident | 30% / 28.2% | 2.13 | 87.32% | +4.54 (+4.01 to +5.08) | +563 / -131 | +3.01 (+2.52 to +3.50) |
| best detector, rule and budget | token | confident | 65% / 64.6% | 3.58 | 88.29% | +5.52 (+4.93 to +6.10) | +716 / -191 | +2.03 (+1.66 to +2.41) |
| smallest budget keeping 80% of the best validation gain | token | confident | 35% / 34.9% | 2.39 | 87.60% | +4.82 (+4.27 to +5.39) | +624 / -165 | +3.13 (+2.66 to +3.63) |

Per dataset, first headline (20% budget):

| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |
|---|---:|---:|---|---:|
| chartqa_full | 76.48% | 81.52% | +5.04 (+3.96 to +6.16) | 83.04% |
| textvqa | 81.52% | 83.60% | +2.08 (+1.08 to +3.12) | 83.84% |
| docvqa | 87.04% | 89.44% | +2.40 (+1.60 to +3.20) | 91.40% |
| scienceqa_img | 86.86% | 91.52% | +4.66 (+3.57 to +5.85) | 94.50% |

Detector AUROC on test (2B answer right / fix would help):

| Detector | 2B right | Fix helps |
|---|---:|---:|
| pre_head | 0.857 | 0.813 |
| pre_probe | 0.856 | 0.816 |
| after_head | 0.881 | 0.831 |
| after_probe | 0.877 | 0.831 |
| token | 0.847 | 0.829 |

## Fix: up1.5

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 83.36% (+0.58). Oracle (either answer right): 86.15%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget |
|---|---|---|---:|---:|---:|---|---|---|
| best detector and rule at a 20% budget | token | confident | 20% / 19.4% | 1.44 | 83.58% | +0.80 (+0.47 to +1.10) | +157 / -81 | +0.57 (+0.26 to +0.86) |
| best detector and rule at a 30% budget | token | confident | 30% / 29.6% | 1.67 | 83.76% | +0.98 (+0.63 to +1.30) | +191 / -98 | +0.70 (+0.40 to +0.99) |
| best detector, rule and budget | token | confident | 40% / 39.8% | 1.90 | 83.84% | +1.06 (+0.69 to +1.41) | +207 / -106 | +0.78 (+0.48 to +1.06) |
| smallest budget keeping 80% of the best validation gain | token | confident | 25% / 24.4% | 1.55 | 83.71% | +0.94 (+0.59 to +1.26) | +182 / -93 | +0.74 (+0.43 to +1.04) |

Per dataset, first headline (20% budget):

| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |
|---|---:|---:|---|---:|
| chartqa_full | 76.48% | 77.28% | +0.80 (+0.24 to +1.40) | 77.24% |
| textvqa | 81.52% | 82.80% | +1.28 (+0.56 to +2.04) | 81.60% |
| docvqa | 87.04% | 88.04% | +1.00 (+0.48 to +1.52) | 88.60% |
| scienceqa_img | 86.86% | 86.81% | -0.05 (-0.74 to +0.64) | 86.61% |

Detector AUROC on test (2B answer right / fix would help):

| Detector | 2B right | Fix helps |
|---|---:|---:|
| pre_head | 0.857 | 0.781 |
| pre_probe | 0.856 | 0.787 |
| after_head | 0.881 | 0.814 |
| after_probe | 0.877 | 0.812 |
| token | 0.847 | 0.848 |

## Fix: up2

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 83.17% (+0.39). Oracle (either answer right): 86.42%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget |
|---|---|---|---:|---:|---:|---|---|---|
| best detector and rule at a 20% budget | token | confident | 20% / 19.4% | 1.78 | 83.59% | +0.81 (+0.47 to +1.13) | +176 / -99 | +0.57 (+0.25 to +0.87) |
| best detector and rule at a 30% budget | token | confident | 30% / 29.6% | 2.18 | 83.71% | +0.94 (+0.56 to +1.29) | +213 / -124 | +0.71 (+0.38 to +1.01) |
| best detector, rule and budget | token | confident | 40% / 39.8% | 2.59 | 83.74% | +0.97 (+0.58 to +1.36) | +225 / -133 | +0.68 (+0.37 to +0.98) |
| smallest budget keeping 80% of the best validation gain | token | confident | 25% / 24.4% | 1.98 | 83.66% | +0.88 (+0.51 to +1.24) | +200 / -116 | +0.74 (+0.40 to +1.05) |

Per dataset, first headline (20% budget):

| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |
|---|---:|---:|---|---:|
| chartqa_full | 76.48% | 77.44% | +0.96 (+0.28 to +1.64) | 77.16% |
| textvqa | 81.52% | 82.48% | +0.96 (+0.24 to +1.72) | 81.44% |
| docvqa | 87.04% | 87.84% | +0.80 (+0.24 to +1.36) | 88.16% |
| scienceqa_img | 86.86% | 87.31% | +0.45 (-0.30 to +1.19) | 86.56% |

Detector AUROC on test (2B answer right / fix would help):

| Detector | 2B right | Fix helps |
|---|---:|---:|
| pre_head | 0.857 | 0.788 |
| pre_probe | 0.856 | 0.788 |
| after_head | 0.881 | 0.812 |
| after_probe | 0.877 | 0.806 |
| token | 0.847 | 0.853 |

