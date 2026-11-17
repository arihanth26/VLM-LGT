# Escalation cascade: 2B -> 4B -> 8B against 2B -> 8B

Greedy 2B on test: 82.78%; 4B everywhere 85.85%; 8B everywhere 87.87%. Cost in 2B passes (4B = 2, 8B = 4). Each setting is chosen on validation under the cost cap and read once on test.

| Cost cap | Route | Detector | Rule | To 4B / to 8B (test) | Cost (test) | Test accuracy | Gain (95% interval) | Cascade minus direct |
|---:|---|---|---|---|---:|---:|---|---|
| 1.5 | cascade | token | confident | 24.4% / 0.0% | 1.49 | 85.78% | +3.01 (+2.49 to +3.51) | +0.79 (+0.34 to +1.25) |
| 1.5 | direct | token | confident | 0.0% / 9.4% | 1.38 | 85.00% | +2.22 (+1.85 to +2.59) |  |
| 2 | cascade | token | confident | 34.9% / 6.4% | 1.95 | 86.92% | +4.14 (+3.54 to +4.72) | -0.13 (-0.61 to +0.36) |
| 2 | direct | after_head | confident | 0.0% / 23.9% | 1.96 | 87.04% | +4.27 (+3.78 to +4.77) |  |
| 2.5 | cascade | after_head | confident | 37.7% / 15.3% | 2.36 | 87.55% | +4.77 (+4.19 to +5.36) | -0.05 (-0.45 to +0.37) |
| 2.5 | direct | token | confident | 0.0% / 34.9% | 2.39 | 87.60% | +4.82 (+4.27 to +5.39) |  |
| 3 | cascade | token | confident | 49.7% / 25.9% | 3.03 | 88.04% | +5.26 (+4.64 to +5.87) | -0.13 (-0.45 to +0.21) |
| 3 | direct | token | confident | 0.0% / 49.7% | 2.99 | 88.17% | +5.39 (+4.80 to +5.97) |  |
| 4 | cascade | token | confident | 64.6% / 40.4% | 3.91 | 88.23% | +5.45 (+4.82 to +6.10) | -0.06 (-0.40 to +0.26) |
| 4 | direct | token | confident | 0.0% / 64.6% | 3.58 | 88.29% | +5.52 (+4.93 to +6.10) |  |
