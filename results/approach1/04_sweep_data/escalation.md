# Escalation study: our detector, new fixes

Every setting is chosen on validation and read once on test. A budget sets a fixed threshold on the detector score from validation; on test each question is routed on its own score. Accuracy is pooled over the four datasets.

## Fix: 4b

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 85.85% (+3.07). Oracle (either answer right): 90.25%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget | Minus fix everything (p) |
|---|---|---|---:|---:|---:|---|---|---|---|
| best detector and rule at a 20% budget | token | confident | 20% / 19.4% | 1.39 | 85.56% | +2.78 (+2.30 to +3.25) | +405 / -140 | +1.95 (+1.49 to +2.43) | -0.28 (-0.79 to +0.21), p = 0.28 |
| best detector and rule at a 30% budget | token | confident | 30% / 29.6% | 1.59 | 86.03% | +3.25 (+2.70 to +3.81) | +509 / -200 | +2.10 (+1.62 to +2.59) | +0.18 (-0.23 to +0.60), p = 0.44 |
| best detector, rule and budget | token | confident | 65% / 64.6% | 2.29 | 86.58% | +3.80 (+3.19 to +4.41) | +623 / -261 | +1.46 (+1.09 to +1.85) | +0.74 (+0.42 to +1.07), p = 9.5e-06 |
| smallest budget keeping 80% of the best validation gain | token | confident | 30% / 29.6% | 1.59 | 86.03% | +3.25 (+2.70 to +3.81) | +509 / -200 | +2.10 (+1.62 to +2.59) | +0.18 (-0.23 to +0.60), p = 0.44 |

Per dataset, first headline (20% budget):

| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |
|---|---:|---:|---|---:|
| chartqa_full | 76.48% | 80.08% | +3.60 (+2.56 to +4.68) | 81.16% |
| textvqa | 81.52% | 83.24% | +1.72 (+0.80 to +2.64) | 81.64% |
| docvqa | 87.04% | 88.92% | +1.88 (+1.16 to +2.64) | 89.16% |
| scienceqa_img | 86.86% | 91.08% | +4.21 (+3.17 to +5.35) | 92.76% |

Detector AUROC on test (2B answer right / fix would help):

| Detector | 2B right | Fix helps |
|---|---:|---:|
| pre_head | 0.857 | 0.805 |
| pre_probe | 0.856 | 0.807 |
| after_head | 0.881 | 0.821 |
| after_probe | 0.877 | 0.822 |
| token | 0.847 | 0.829 |

## Fix: 8b

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 87.87% (+5.10). Oracle (either answer right): 91.40%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget | Minus fix everything (p) |
|---|---|---|---:|---:|---:|---|---|---|---|
| best detector and rule at a 20% budget | token | replace | 20% / 19.4% | 1.78 | 86.27% | +3.49 (+2.99 to +4.00) | +478 / -146 | +2.60 (+2.09 to +3.07) | -1.61 (-2.08 to -1.13), p = 3.4e-11 |
| best detector and rule at a 30% budget | after_head | confident | 30% / 28.2% | 2.13 | 87.32% | +4.54 (+4.01 to +5.08) | +563 / -131 | +3.01 (+2.52 to +3.50) | -0.56 (-0.99 to -0.11), p = 0.016 |
| best detector, rule and budget | token | confident | 65% / 64.6% | 3.58 | 88.29% | +5.52 (+4.93 to +6.10) | +716 / -191 | +2.03 (+1.66 to +2.41) | +0.42 (+0.09 to +0.76), p = 0.013 |
| smallest budget keeping 80% of the best validation gain | token | confident | 35% / 34.9% | 2.39 | 87.60% | +4.82 (+4.27 to +5.39) | +624 / -165 | +3.13 (+2.66 to +3.63) | -0.27 (-0.68 to +0.14), p = 0.19 |

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

## Fix: idefics3_8b

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 68.89% (-13.89). Oracle (either answer right): 88.59%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget | Minus fix everything (p) |
|---|---|---|---:|---:|---:|---|---|---|---|
| best detector and rule at a 20% budget | token | confident | 20% / 19.4% | 1.78 | 83.17% | +0.39 (-0.06 to +0.86) | +272 / -235 | +0.46 (+0.01 to +0.91) | +14.28 (+13.44 to +15.13), p = 5.5e-233 |
| best detector and rule at a 30% budget | pre_probe | confident | 30% / 28.4% | 2.14 | 83.68% | +0.90 (+0.42 to +1.40) | +314 / -228 | +1.06 (+0.62 to +1.49) | +14.79 (+13.97 to +15.65), p = 2.6e-258 |
| best detector, rule and budget | after_head | confident | 45% / 42.9% | 2.72 | 83.76% | +0.98 (+0.45 to +1.51) | +378 / -285 | +1.16 (+0.73 to +1.58) | +14.87 (+14.09 to +15.68), p = 1.9e-285 |
| smallest budget keeping 80% of the best validation gain | after_head | confident | 30% / 28.2% | 2.13 | 83.92% | +1.15 (+0.67 to +1.65) | +331 / -222 | +1.30 (+0.85 to +1.76) | +15.04 (+14.23 to +15.88), p = 3.7e-270 |

Per dataset, first headline (20% budget):

| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |
|---|---:|---:|---|---:|
| chartqa_full | 76.48% | 77.08% | +0.60 (-0.40 to +1.56) | 67.60% |
| textvqa | 81.52% | 81.52% | +0.00 (-0.92 to +0.92) | 64.12% |
| docvqa | 87.04% | 86.28% | -0.76 (-1.52 to -0.04) | 58.64% |
| scienceqa_img | 86.86% | 88.89% | +2.03 (+0.99 to +3.17) | 89.09% |

Detector AUROC on test (2B answer right / fix would help):

| Detector | 2B right | Fix helps |
|---|---:|---:|
| pre_head | 0.857 | 0.790 |
| pre_probe | 0.856 | 0.793 |
| after_head | 0.881 | 0.816 |
| after_probe | 0.877 | 0.814 |
| token | 0.847 | 0.803 |

## Fix: internvl3_8b

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 88.03% (+5.25). Oracle (either answer right): 92.19%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget | Minus fix everything (p) |
|---|---|---|---:|---:|---:|---|---|---|---|
| best detector and rule at a 20% budget | after_head | replace | 20% / 19.1% | 1.76 | 86.97% | +4.19 (+3.65 to +4.73) | +550 / -151 | +3.23 (+2.70 to +3.75) | -1.06 (-1.57 to -0.57), p = 3.7e-05 |
| best detector and rule at a 30% budget | after_head | replace | 30% / 28.2% | 2.13 | 87.53% | +4.75 (+4.14 to +5.37) | +677 / -225 | +3.27 (+2.71 to +3.85) | -0.50 (-0.90 to -0.11), p = 0.017 |
| best detector, rule and budget | token | replace | 55% / 54.7% | 3.19 | 88.43% | +5.65 (+4.94 to +6.33) | +856 / -318 | +3.01 (+2.51 to +3.54) | +0.40 (+0.18 to +0.62), p = 0.0006 |
| smallest budget keeping 80% of the best validation gain | token | replace | 30% / 29.6% | 2.18 | 87.37% | +4.59 (+3.99 to +5.17) | +651 / -214 | +3.11 (+2.54 to +3.68) | -0.66 (-1.07 to -0.25), p = 0.0027 |

Per dataset, first headline (20% budget):

| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |
|---|---:|---:|---|---:|
| chartqa_full | 76.48% | 82.56% | +6.08 (+4.84 to +7.36) | 83.96% |
| textvqa | 81.52% | 83.68% | +2.16 (+1.20 to +3.12) | 83.00% |
| docvqa | 87.04% | 88.56% | +1.52 (+0.84 to +2.20) | 88.96% |
| scienceqa_img | 86.86% | 94.55% | +7.68 (+6.49 to +8.97) | 98.17% |

Detector AUROC on test (2B answer right / fix would help):

| Detector | 2B right | Fix helps |
|---|---:|---:|
| pre_head | 0.857 | 0.810 |
| pre_probe | 0.856 | 0.812 |
| after_head | 0.881 | 0.832 |
| after_probe | 0.877 | 0.830 |
| token | 0.847 | 0.823 |

## Fix: up1.5

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 82.78%. Fix everything: 83.36% (+0.58). Oracle (either answer right): 86.15%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget | Minus fix everything (p) |
|---|---|---|---:|---:|---:|---|---|---|---|
| best detector and rule at a 20% budget | token | confident | 20% / 19.4% | 1.44 | 83.58% | +0.80 (+0.47 to +1.10) | +157 / -81 | +0.57 (+0.26 to +0.86) | +0.22 (-0.16 to +0.61), p = 0.28 |
| best detector and rule at a 30% budget | token | confident | 30% / 29.6% | 1.67 | 83.76% | +0.98 (+0.63 to +1.30) | +191 / -98 | +0.70 (+0.40 to +0.99) | +0.40 (+0.04 to +0.78), p = 0.032 |
| best detector, rule and budget | token | confident | 40% / 39.8% | 1.90 | 83.84% | +1.06 (+0.69 to +1.41) | +207 / -106 | +0.78 (+0.48 to +1.06) | +0.48 (+0.14 to +0.83), p = 0.0065 |
| smallest budget keeping 80% of the best validation gain | token | confident | 25% / 24.4% | 1.55 | 83.71% | +0.94 (+0.59 to +1.26) | +182 / -93 | +0.74 (+0.43 to +1.04) | +0.36 (-0.01 to +0.75), p = 0.062 |

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

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget | Minus fix everything (p) |
|---|---|---|---:|---:|---:|---|---|---|---|
| best detector and rule at a 20% budget | token | confident | 20% / 19.4% | 1.78 | 83.59% | +0.81 (+0.47 to +1.13) | +176 / -99 | +0.57 (+0.25 to +0.87) | +0.42 (+0.02 to +0.83), p = 0.046 |
| best detector and rule at a 30% budget | token | confident | 30% / 29.6% | 2.18 | 83.71% | +0.94 (+0.56 to +1.29) | +213 / -124 | +0.71 (+0.38 to +1.01) | +0.55 (+0.18 to +0.90), p = 0.0043 |
| best detector, rule and budget | token | confident | 40% / 39.8% | 2.59 | 83.74% | +0.97 (+0.58 to +1.36) | +225 / -133 | +0.68 (+0.37 to +0.98) | +0.58 (+0.23 to +0.92), p = 0.0017 |
| smallest budget keeping 80% of the best validation gain | token | confident | 25% / 24.4% | 1.98 | 83.66% | +0.88 (+0.51 to +1.24) | +200 / -116 | +0.74 (+0.40 to +1.05) | +0.49 (+0.11 to +0.86), p = 0.013 |

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

