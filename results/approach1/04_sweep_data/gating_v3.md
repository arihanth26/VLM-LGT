# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus two scoring passes per distinct candidate of a corrected question); calls: generation calls only.

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 22.13 | 14.00 | 85.88% | +3.10 (+2.55 to +3.64) | +474 / -179 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 6.00 | 3.53 | 85.30% | +2.52 (+2.08 to +2.95) | +348 / -108 | +1.86 (+1.46 to +2.25) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 8.25 | 4.84 | 85.70% | +2.92 (+2.44 to +3.40) | +422 / -144 | +1.99 (+1.58 to +2.42) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 12.31 | 7.46 | 85.90% | +3.12 (+2.58 to +3.64) | +470 / -173 | +1.74 (+1.36 to +2.11) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 30% / 29.6% | 8.25 | 4.84 | 85.70% | +2.92 (+2.44 to +3.40) | +422 / -144 | +1.99 (+1.58 to +2.42) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 8.53 | 5.00 | 85.84% | +3.06 (+2.56 to +3.57) | +441 / -150 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 19.4% | 2.80 | 1.78 | 85.27% | +2.49 (+2.07 to +2.92) | +326 / -89 | +1.87 (+1.46 to +2.24) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 85.65% | +2.87 (+2.42 to +3.34) | +393 / -120 | +2.01 (+1.61 to +2.43) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 5.11 | 2.99 | 85.86% | +3.08 (+2.60 to +3.58) | +438 / -145 | +1.66 (+1.30 to +2.04) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 85.65% | +2.87 (+2.42 to +3.34) | +393 / -120 | +2.01 (+1.61 to +2.43) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 4.52 | 2.00 | 85.34% | +2.56 (+2.04 to +3.05) | +410 / -166 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 84.95% | +2.18 (+1.78 to +2.57) | +296 / -89 | +1.59 (+1.21 to +1.95) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 85.24% | +2.46 (+1.98 to +2.90) | +357 / -123 | +1.71 (+1.31 to +2.10) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.93 | 1.50 | 85.39% | +2.62 (+2.11 to +3.11) | +406 / -157 | +1.47 (+1.10 to +1.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 85.24% | +2.46 (+1.98 to +2.90) | +357 / -123 | +1.71 (+1.31 to +2.10) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 12.57 | 6.00 | 85.57% | +2.79 (+2.26 to +3.31) | +443 / -177 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | after_head | 20% / 19.2% | 4.27 | 1.96 | 85.29% | +2.51 (+2.07 to +2.94) | +354 / -115 | +1.84 (+1.44 to +2.25) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 85.49% | +2.71 (+2.24 to +3.15) | +393 / -135 | +1.82 (+1.41 to +2.23) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 7.04 | 3.49 | 85.60% | +2.83 (+2.31 to +3.33) | +439 / -170 | +1.56 (+1.16 to +1.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 85.49% | +2.71 (+2.24 to +3.15) | +393 / -135 | +1.82 (+1.41 to +2.23) |
| best under 2 passes per question (any configuration) | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 84.95% | +2.18 (+1.78 to +2.57) | +296 / -89 | +1.59 (+1.21 to +1.95) |
| best under 3 passes per question (any configuration) | T1_k0_cot | token | 50% / 49.7% | 2.93 | 1.50 | 85.39% | +2.62 (+2.11 to +3.11) | +406 / -157 | +1.47 (+1.10 to +1.82) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot | token | 45% / 45.1% | 4.78 | 2.80 | 85.88% | +3.10 (+2.62 to +3.60) | +434 / -139 | +1.77 (+1.39 to +2.15) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +5.68 (+4.48 to +6.92) | +2.88 (+1.84 to +3.92) | +1.40 (+0.68 to +2.08) | +2.28 (+1.09 to +3.57) |
| best detector at a 20% budget | T1_k5_everything | +4.72 (+3.68 to +5.80) | +2.68 (+1.84 to +3.56) | +0.84 (+0.32 to +1.36) | +1.69 (+0.74 to +2.68) |
| best detector at a 30% budget | T1_k5_everything | +5.36 (+4.24 to +6.52) | +2.88 (+1.92 to +3.84) | +1.16 (+0.56 to +1.80) | +2.13 (+1.04 to +3.32) |
| best detector at a 50% budget | T1_k5_everything | +5.60 (+4.36 to +6.84) | +2.88 (+1.84 to +3.92) | +1.44 (+0.76 to +2.12) | +2.43 (+1.29 to +3.72) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +5.36 (+4.24 to +6.52) | +2.88 (+1.92 to +3.84) | +1.16 (+0.56 to +1.80) | +2.13 (+1.04 to +3.32) |
| correct every question (no gate) | T1_k0_ground_cot | +6.08 (+4.88 to +7.28) | +2.24 (+1.36 to +3.16) | +1.28 (+0.60 to +1.96) | +2.53 (+1.39 to +3.77) |
| best detector at a 20% budget | T1_k0_ground_cot | +5.00 (+4.00 to +6.08) | +2.16 (+1.40 to +2.96) | +0.72 (+0.20 to +1.24) | +1.98 (+1.09 to +2.93) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.68 (+4.56 to +6.80) | +2.28 (+1.44 to +3.16) | +1.00 (+0.32 to +1.60) | +2.43 (+1.44 to +3.57) |
| best detector at a 50% budget | T1_k0_ground_cot | +6.00 (+4.80 to +7.20) | +2.24 (+1.36 to +3.16) | +1.32 (+0.64 to +2.00) | +2.68 (+1.54 to +3.87) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.68 (+4.56 to +6.80) | +2.28 (+1.44 to +3.16) | +1.00 (+0.32 to +1.60) | +2.43 (+1.44 to +3.57) |
| correct every question (no gate) | T1_k0_cot | +6.40 (+5.16 to +7.64) | +0.84 (+0.00 to +1.68) | +0.60 (-0.08 to +1.24) | +2.38 (+1.24 to +3.57) |
| best detector at a 20% budget | T1_k0_cot | +5.16 (+4.16 to +6.20) | +1.20 (+0.48 to +1.92) | +0.40 (-0.08 to +0.88) | +1.88 (+1.04 to +2.83) |
| best detector at a 30% budget | T1_k0_cot | +5.84 (+4.76 to +6.96) | +1.08 (+0.32 to +1.84) | +0.48 (-0.12 to +1.08) | +2.43 (+1.44 to +3.52) |
| best detector at a 50% budget | T1_k0_cot | +6.28 (+5.08 to +7.48) | +0.96 (+0.16 to +1.84) | +0.60 (-0.08 to +1.24) | +2.63 (+1.49 to +3.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.84 (+4.76 to +6.96) | +1.08 (+0.32 to +1.84) | +0.48 (-0.12 to +1.08) | +2.43 (+1.44 to +3.52) |
| correct every question (no gate) | T1_k0_zoom_cot | +5.84 (+4.60 to +7.04) | +1.96 (+1.04 to +2.92) | +1.00 (+0.28 to +1.72) | +2.28 (+1.14 to +3.57) |
| best detector at a 20% budget | T1_k0_zoom_cot | +5.60 (+4.44 to +6.72) | +1.24 (+0.44 to +2.04) | +0.72 (+0.28 to +1.20) | +2.48 (+1.44 to +3.57) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.48 (+4.36 to +6.60) | +2.20 (+1.28 to +3.08) | +0.80 (+0.20 to +1.44) | +2.28 (+1.29 to +3.42) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.72 (+4.48 to +6.92) | +2.00 (+1.08 to +2.92) | +1.04 (+0.32 to +1.76) | +2.48 (+1.34 to +3.72) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.48 (+4.36 to +6.60) | +2.20 (+1.28 to +3.08) | +0.80 (+0.20 to +1.44) | +2.28 (+1.29 to +3.42) |
| best under 2 passes per question (any configuration) | T1_k0_cot | +5.16 (+4.16 to +6.20) | +1.20 (+0.48 to +1.92) | +0.40 (-0.08 to +0.88) | +1.88 (+1.04 to +2.83) |
| best under 3 passes per question (any configuration) | T1_k0_cot | +6.28 (+5.08 to +7.48) | +0.96 (+0.16 to +1.84) | +0.60 (-0.08 to +1.24) | +2.63 (+1.49 to +3.82) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot | +6.04 (+4.84 to +7.24) | +2.32 (+1.44 to +3.20) | +1.32 (+0.64 to +2.00) | +2.63 (+1.54 to +3.82) |
