# Escalation study: our detector, new fixes

Every setting is chosen on validation and read once on test. A budget sets a fixed threshold on the detector score from validation; on test each question is routed on its own score. Accuracy is pooled over the four datasets.

## Fix: 8b

Questions {'val': 6420, 'test': 9517}, missing fix answers {'val': 0, 'test': 0}. 2B greedy on test: 85.84%. Fix everything: 87.87% (+2.04). Oracle (either answer right): 90.95%.

| Setting | Detector | Rule | Budget (val) / share routed (test) | Cost (2B passes) | Test accuracy | Gain (95% interval) | Fixed / broken | Gain over random, same budget |
|---|---|---|---:|---:|---:|---|---|---|
| best detector and rule at a 20% budget | token | confident | 20% / 20.1% | 1.40 | 87.49% | +1.65 (+1.27 to +2.03) | +233 / -76 | +1.25 (+0.90 to +1.60) |
| best detector and rule at a 30% budget | token | confident | 30% / 31.2% | 1.62 | 87.91% | +2.07 (+1.66 to +2.48) | +288 / -91 | +1.44 (+1.10 to +1.80) |
| best detector, rule and budget | token | confident | 65% / 66.3% | 2.33 | 88.07% | +2.24 (+1.81 to +2.67) | +313 / -100 | +0.88 (+0.63 to +1.13) |
| smallest budget keeping 80% of the best validation gain | token | confident | 20% / 20.1% | 1.40 | 87.49% | +1.65 (+1.27 to +2.03) | +233 / -76 | +1.25 (+0.90 to +1.60) |

Per dataset, first headline (20% budget):

| Dataset | Greedy | Chosen | Gain (95% interval) | Fix everything |
|---|---:|---:|---|---:|
| chartqa_full | 81.16% | 82.76% | +1.60 (+0.84 to +2.36) | 83.04% |
| textvqa | 81.64% | 83.76% | +2.12 (+1.32 to +2.96) | 83.84% |
| docvqa | 89.12% | 90.44% | +1.32 (+0.72 to +1.92) | 91.40% |
| scienceqa_img | 92.76% | 94.30% | +1.54 (+0.94 to +2.18) | 94.50% |

Detector AUROC on test (2B answer right / fix would help):

| Detector | 2B right | Fix helps |
|---|---:|---:|
| token | 0.834 | 0.823 |

