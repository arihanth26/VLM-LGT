# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus two scoring passes per distinct candidate of a corrected question); calls: generation calls only.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.87% vs 85.87%, T1_k0_ground_cot 85.86% vs 85.86%, T1_k0_cot 85.39% vs 85.39%, T1_k0_zoom_cot 85.63% vs 85.63%

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 24.13 | 16.00 | 85.87% | +3.09 (+2.54 to +3.61) | +473 / -179 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 6.39 | 3.92 | 85.32% | +2.54 (+2.11 to +2.98) | +348 / -106 | +1.88 (+1.48 to +2.27) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 85.70% | +2.92 (+2.44 to +3.40) | +421 / -143 | +2.00 (+1.58 to +2.43) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 13.31 | 8.46 | 85.90% | +3.12 (+2.60 to +3.64) | +470 / -173 | +1.72 (+1.33 to +2.10) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 85.70% | +2.92 (+2.44 to +3.40) | +421 / -143 | +2.00 (+1.58 to +2.43) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 8.53 | 5.00 | 85.86% | +3.08 (+2.60 to +3.58) | +440 / -147 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 19.4% | 2.80 | 1.78 | 85.30% | +2.52 (+2.10 to +2.94) | +326 / -86 | +1.90 (+1.49 to +2.27) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 85.68% | +2.90 (+2.45 to +3.36) | +393 / -117 | +2.04 (+1.64 to +2.45) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 5.11 | 2.99 | 85.88% | +3.10 (+2.62 to +3.60) | +437 / -142 | +1.69 (+1.32 to +2.06) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 85.68% | +2.90 (+2.45 to +3.36) | +393 / -117 | +2.04 (+1.64 to +2.45) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 4.52 | 2.00 | 85.39% | +2.62 (+2.11 to +3.11) | +407 / -158 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 84.96% | +2.19 (+1.79 to +2.58) | +295 / -87 | +1.59 (+1.21 to +1.94) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 85.27% | +2.49 (+2.02 to +2.94) | +355 / -118 | +1.72 (+1.33 to +2.12) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.93 | 1.50 | 85.43% | +2.65 (+2.14 to +3.14) | +403 / -151 | +1.48 (+1.11 to +1.83) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 85.27% | +2.49 (+2.02 to +2.94) | +355 / -118 | +1.72 (+1.33 to +2.12) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 12.57 | 6.00 | 85.63% | +2.85 (+2.31 to +3.35) | +447 / -176 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | after_head | 20% / 19.2% | 4.27 | 1.96 | 85.33% | +2.55 (+2.10 to +2.99) | +357 / -114 | +1.87 (+1.47 to +2.28) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 85.55% | +2.77 (+2.30 to +3.24) | +397 / -133 | +1.87 (+1.46 to +2.28) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 7.04 | 3.49 | 85.66% | +2.88 (+2.35 to +3.38) | +443 / -169 | +1.59 (+1.20 to +1.95) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 85.55% | +2.77 (+2.30 to +3.24) | +397 / -133 | +1.87 (+1.46 to +2.28) |
| best under 2 passes per question (any configuration) | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 84.96% | +2.19 (+1.79 to +2.58) | +295 / -87 | +1.59 (+1.21 to +1.94) |
| best under 3 passes per question (any configuration) | T1_k0_cot | token | 50% / 49.7% | 2.93 | 1.50 | 85.43% | +2.65 (+2.14 to +3.14) | +403 / -151 | +1.48 (+1.11 to +1.83) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 85.68% | +2.90 (+2.45 to +3.36) | +393 / -117 | +2.04 (+1.64 to +2.45) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +5.64 (+4.44 to +6.84) | +2.88 (+1.84 to +3.92) | +1.44 (+0.72 to +2.12) | +2.23 (+1.04 to +3.52) |
| best detector at a 20% budget | T1_k5_everything | +4.72 (+3.68 to +5.80) | +2.68 (+1.84 to +3.56) | +0.88 (+0.36 to +1.40) | +1.74 (+0.79 to +2.73) |
| best detector at a 30% budget | T1_k5_everything | +5.36 (+4.24 to +6.48) | +2.88 (+1.96 to +3.84) | +1.12 (+0.48 to +1.72) | +2.18 (+1.09 to +3.37) |
| best detector at a 50% budget | T1_k5_everything | +5.56 (+4.36 to +6.76) | +2.88 (+1.84 to +3.92) | +1.48 (+0.76 to +2.16) | +2.43 (+1.24 to +3.72) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +5.36 (+4.24 to +6.48) | +2.88 (+1.96 to +3.84) | +1.12 (+0.48 to +1.72) | +2.18 (+1.09 to +3.37) |
| correct every question (no gate) | T1_k0_ground_cot | +6.04 (+4.84 to +7.24) | +2.36 (+1.52 to +3.28) | +1.28 (+0.60 to +1.96) | +2.53 (+1.39 to +3.77) |
| best detector at a 20% budget | T1_k0_ground_cot | +5.00 (+4.00 to +6.08) | +2.28 (+1.52 to +3.08) | +0.72 (+0.20 to +1.24) | +1.98 (+1.09 to +2.93) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.68 (+4.56 to +6.80) | +2.40 (+1.60 to +3.24) | +1.00 (+0.32 to +1.60) | +2.43 (+1.44 to +3.57) |
| best detector at a 50% budget | T1_k0_ground_cot | +5.96 (+4.76 to +7.16) | +2.36 (+1.52 to +3.28) | +1.32 (+0.64 to +2.00) | +2.68 (+1.54 to +3.87) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.68 (+4.56 to +6.80) | +2.40 (+1.60 to +3.24) | +1.00 (+0.32 to +1.60) | +2.43 (+1.44 to +3.57) |
| correct every question (no gate) | T1_k0_cot | +6.32 (+5.08 to +7.52) | +1.00 (+0.20 to +1.84) | +0.64 (-0.04 to +1.28) | +2.48 (+1.34 to +3.67) |
| best detector at a 20% budget | T1_k0_cot | +5.12 (+4.12 to +6.16) | +1.24 (+0.56 to +1.96) | +0.44 (-0.04 to +0.92) | +1.88 (+1.04 to +2.83) |
| best detector at a 30% budget | T1_k0_cot | +5.80 (+4.72 to +6.96) | +1.24 (+0.48 to +2.00) | +0.52 (-0.08 to +1.12) | +2.38 (+1.39 to +3.47) |
| best detector at a 50% budget | T1_k0_cot | +6.20 (+5.00 to +7.40) | +1.08 (+0.28 to +1.92) | +0.64 (-0.04 to +1.28) | +2.68 (+1.59 to +3.87) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.80 (+4.72 to +6.96) | +1.24 (+0.48 to +2.00) | +0.52 (-0.08 to +1.12) | +2.38 (+1.39 to +3.47) |
| correct every question (no gate) | T1_k0_zoom_cot | +5.92 (+4.68 to +7.12) | +2.08 (+1.16 to +3.04) | +0.96 (+0.24 to +1.68) | +2.33 (+1.14 to +3.62) |
| best detector at a 20% budget | T1_k0_zoom_cot | +5.64 (+4.48 to +6.76) | +1.36 (+0.56 to +2.16) | +0.68 (+0.24 to +1.16) | +2.53 (+1.49 to +3.62) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.56 (+4.44 to +6.68) | +2.32 (+1.40 to +3.20) | +0.80 (+0.20 to +1.44) | +2.33 (+1.29 to +3.47) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.80 (+4.56 to +7.00) | +2.12 (+1.16 to +3.08) | +1.00 (+0.28 to +1.72) | +2.53 (+1.34 to +3.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.56 (+4.44 to +6.68) | +2.32 (+1.40 to +3.20) | +0.80 (+0.20 to +1.44) | +2.33 (+1.29 to +3.47) |
| best under 2 passes per question (any configuration) | T1_k0_cot | +5.12 (+4.12 to +6.16) | +1.24 (+0.56 to +1.96) | +0.44 (-0.04 to +0.92) | +1.88 (+1.04 to +2.83) |
| best under 3 passes per question (any configuration) | T1_k0_cot | +6.20 (+5.00 to +7.40) | +1.08 (+0.28 to +1.92) | +0.64 (-0.04 to +1.28) | +2.68 (+1.59 to +3.87) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot | +5.68 (+4.56 to +6.80) | +2.40 (+1.60 to +3.24) | +1.00 (+0.32 to +1.60) | +2.43 (+1.44 to +3.57) |
