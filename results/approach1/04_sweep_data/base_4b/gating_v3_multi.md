# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_models.pt`. Greedy on test: 85.84%. Passes: model passes per question (generation calls plus 2 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 87.31% vs 87.31%, T1_k0_ground_cot 87.28% vs 87.28%, T1_k0_cot 87.28% vs 87.28%, T1_k0_zoom_cot 87.28% vs 87.28%, T1_k0_cot_multi 87.50% vs 87.50%, T1_k0_ground_cot_multi 87.56% vs 87.56%



| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Seconds | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 23.74 | 16.00 | 0.00 | 87.31% | +1.47 (+1.13 to +1.80) | +200 / -60 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 20.1% | 6.39 | 4.01 | 0.00 | 87.10% | +1.26 (+0.96 to +1.56) | +166 / -46 | +0.95 (+0.68 to +1.22) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 31.2% | 9.01 | 5.69 | 0.00 | 87.23% | +1.40 (+1.08 to +1.71) | +189 / -56 | +0.99 (+0.73 to +1.25) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 52.4% | 13.74 | 8.86 | 0.00 | 87.31% | +1.47 (+1.15 to +1.80) | +199 / -59 | +0.73 (+0.49 to +0.96) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 25% / 25.3% | 7.62 | 4.80 | 0.00 | 87.22% | +1.39 (+1.08 to +1.70) | +182 / -50 | +1.01 (+0.75 to +1.27) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 8.45 | 5.00 | 0.00 | 87.28% | +1.44 (+1.08 to +1.79) | +220 / -83 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 20.1% | 2.81 | 1.80 | 0.00 | 86.95% | +1.11 (+0.80 to +1.42) | +168 / -62 | +0.89 (+0.60 to +1.19) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 31.2% | 3.70 | 2.25 | 0.00 | 87.15% | +1.31 (+0.98 to +1.65) | +201 / -76 | +1.04 (+0.76 to +1.31) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 52.4% | 5.25 | 3.09 | 0.00 | 87.26% | +1.43 (+1.07 to +1.78) | +219 / -83 | +0.82 (+0.58 to +1.06) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 25% / 25.3% | 3.23 | 2.01 | 0.00 | 87.10% | +1.26 (+0.92 to +1.58) | +188 / -68 | +0.99 (+0.69 to +1.27) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 4.53 | 2.00 | 0.00 | 87.28% | +1.44 (+1.05 to +1.81) | +243 / -106 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 20.1% | 1.85 | 1.20 | 0.00 | 86.99% | +1.16 (+0.81 to +1.47) | +181 / -71 | +1.00 (+0.67 to +1.30) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 31.2% | 2.28 | 1.31 | 0.00 | 87.19% | +1.36 (+1.00 to +1.69) | +218 / -89 | +1.08 (+0.78 to +1.38) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 52.4% | 3.02 | 1.52 | 0.00 | 87.29% | +1.45 (+1.06 to +1.82) | +242 / -104 | +0.96 (+0.69 to +1.23) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 25% / 25.3% | 2.05 | 1.25 | 0.00 | 87.16% | +1.32 (+0.98 to +1.65) | +202 / -76 | +1.09 (+0.78 to +1.40) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 12.59 | 6.00 | 0.00 | 87.28% | +1.44 (+1.09 to +1.78) | +210 / -73 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 20.1% | 3.69 | 2.00 | 0.00 | 87.00% | +1.17 (+0.85 to +1.46) | +167 / -56 | +0.97 (+0.68 to +1.25) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 31.2% | 5.08 | 2.56 | 0.00 | 87.18% | +1.34 (+1.00 to +1.67) | +195 / -67 | +1.04 (+0.77 to +1.30) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 52.4% | 7.55 | 3.62 | 0.00 | 87.29% | +1.45 (+1.10 to +1.79) | +209 / -71 | +0.83 (+0.59 to +1.06) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 20% / 20.1% | 3.69 | 2.00 | 0.00 | 87.00% | +1.17 (+0.85 to +1.46) | +167 / -56 | +0.97 (+0.68 to +1.25) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 8.00 | 5.00 | 0.00 | 87.50% | +1.66 (+1.31 to +2.01) | +229 / -71 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 20.1% | 2.70 | 1.80 | 0.00 | 87.15% | +1.31 (+1.01 to +1.61) | +174 / -49 | +1.03 (+0.76 to +1.32) |
| best detector at a 30% budget | T1_k0_cot_multi | token | 30% / 31.2% | 3.54 | 2.25 | 0.00 | 87.39% | +1.56 (+1.22 to +1.88) | +208 / -60 | +1.20 (+0.92 to +1.48) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 52.4% | 4.99 | 3.09 | 0.00 | 87.51% | +1.67 (+1.32 to +2.02) | +228 / -69 | +0.94 (+0.69 to +1.20) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 25% / 25.3% | 3.10 | 2.01 | 0.00 | 87.31% | +1.47 (+1.16 to +1.79) | +195 / -55 | +1.13 (+0.85 to +1.43) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 11.84 | 8.00 | 0.00 | 87.56% | +1.72 (+1.38 to +2.06) | +232 / -68 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 20.1% | 3.62 | 2.41 | 0.00 | 87.14% | +1.30 (+1.00 to +1.60) | +176 / -52 | +0.98 (+0.70 to +1.27) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 31.2% | 4.91 | 3.19 | 0.00 | 87.45% | +1.62 (+1.28 to +1.94) | +214 / -60 | +1.21 (+0.91 to +1.49) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 52.4% | 7.16 | 4.67 | 0.00 | 87.56% | +1.72 (+1.38 to +2.06) | +230 / -66 | +0.87 (+0.63 to +1.12) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 25% / 25.3% | 4.23 | 2.77 | 0.00 | 87.34% | +1.50 (+1.19 to +1.82) | +199 / -56 | +1.16 (+0.88 to +1.45) |
| best under 2 passes per question (any configuration) | T1_k0_cot | token | 20% / 20.1% | 1.85 | 1.20 | 0.00 | 86.99% | +1.16 (+0.81 to +1.47) | +181 / -71 | +1.00 (+0.67 to +1.30) |
| best under 3 passes per question (any configuration) | T1_k0_cot | token | 35% / 36.6% | 2.48 | 1.37 | 0.00 | 87.30% | +1.46 (+1.09 to +1.81) | +229 / -90 | +1.12 (+0.82 to +1.41) |
| best under 5 passes per question (any configuration) | T1_k0_cot_multi | token | 35% / 36.6% | 3.92 | 2.47 | 0.00 | 87.50% | +1.66 (+1.32 to +2.00) | +218 / -60 | +1.17 (+0.89 to +1.45) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +2.24 (+1.48 to +3.00) | +1.32 (+0.72 to +1.92) | +1.28 (+0.72 to +1.84) | +0.94 (+0.25 to +1.64) |
| best detector at a 20% budget | T1_k5_everything | +1.96 (+1.24 to +2.64) | +1.24 (+0.64 to +1.84) | +1.00 (+0.52 to +1.48) | +0.74 (+0.15 to +1.34) |
| best detector at a 30% budget | T1_k5_everything | +2.08 (+1.32 to +2.80) | +1.32 (+0.72 to +1.92) | +1.24 (+0.68 to +1.80) | +0.84 (+0.15 to +1.49) |
| best detector at a 50% budget | T1_k5_everything | +2.20 (+1.44 to +2.96) | +1.32 (+0.72 to +1.92) | +1.32 (+0.76 to +1.88) | +0.94 (+0.25 to +1.64) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +2.04 (+1.32 to +2.76) | +1.32 (+0.72 to +1.92) | +1.20 (+0.68 to +1.72) | +0.89 (+0.25 to +1.54) |
| correct every question (no gate) | T1_k0_ground_cot | +2.60 (+1.80 to +3.40) | +0.88 (+0.24 to +1.56) | +1.28 (+0.64 to +1.92) | +0.89 (+0.15 to +1.64) |
| best detector at a 20% budget | T1_k0_ground_cot | +1.96 (+1.24 to +2.68) | +0.88 (+0.28 to +1.48) | +0.84 (+0.36 to +1.36) | +0.69 (+0.10 to +1.29) |
| best detector at a 30% budget | T1_k0_ground_cot | +2.24 (+1.48 to +3.04) | +0.92 (+0.28 to +1.56) | +1.12 (+0.56 to +1.72) | +0.89 (+0.20 to +1.59) |
| best detector at a 50% budget | T1_k0_ground_cot | +2.56 (+1.76 to +3.36) | +0.88 (+0.24 to +1.56) | +1.28 (+0.64 to +1.92) | +0.89 (+0.15 to +1.64) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +2.08 (+1.32 to +2.80) | +0.92 (+0.28 to +1.56) | +1.08 (+0.52 to +1.64) | +0.89 (+0.25 to +1.59) |
| correct every question (no gate) | T1_k0_cot | +3.12 (+2.20 to +4.08) | +0.48 (-0.20 to +1.16) | +1.20 (+0.56 to +1.80) | +0.84 (-0.05 to +1.69) |
| best detector at a 20% budget | T1_k0_cot | +2.56 (+1.76 to +3.40) | +0.52 (-0.12 to +1.16) | +0.76 (+0.28 to +1.24) | +0.69 (+0.05 to +1.29) |
| best detector at a 30% budget | T1_k0_cot | +2.84 (+1.96 to +3.72) | +0.52 (-0.12 to +1.16) | +1.00 (+0.44 to +1.56) | +0.99 (+0.30 to +1.69) |
| best detector at a 50% budget | T1_k0_cot | +3.12 (+2.20 to +4.04) | +0.48 (-0.20 to +1.16) | +1.24 (+0.64 to +1.84) | +0.84 (-0.05 to +1.69) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +2.72 (+1.88 to +3.60) | +0.52 (-0.12 to +1.12) | +1.00 (+0.48 to +1.52) | +0.99 (+0.35 to +1.69) |
| correct every question (no gate) | T1_k0_zoom_cot | +2.44 (+1.64 to +3.24) | +0.96 (+0.32 to +1.60) | +1.36 (+0.76 to +1.92) | +0.89 (+0.15 to +1.64) |
| best detector at a 20% budget | T1_k0_zoom_cot | +1.96 (+1.24 to +2.72) | +0.88 (+0.24 to +1.48) | +1.04 (+0.56 to +1.52) | +0.69 (+0.10 to +1.29) |
| best detector at a 30% budget | T1_k0_zoom_cot | +2.16 (+1.40 to +2.92) | +0.96 (+0.36 to +1.60) | +1.28 (+0.72 to +1.84) | +0.89 (+0.20 to +1.59) |
| best detector at a 50% budget | T1_k0_zoom_cot | +2.40 (+1.60 to +3.20) | +0.96 (+0.32 to +1.60) | +1.44 (+0.84 to +2.00) | +0.89 (+0.15 to +1.64) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +1.96 (+1.24 to +2.72) | +0.88 (+0.24 to +1.48) | +1.04 (+0.56 to +1.52) | +0.69 (+0.10 to +1.29) |
| correct every question (no gate) | T1_k0_cot_multi | +3.32 (+2.48 to +4.20) | +0.48 (-0.16 to +1.08) | +1.28 (+0.72 to +1.88) | +1.54 (+0.79 to +2.28) |
| best detector at a 20% budget | T1_k0_cot_multi | +2.76 (+2.00 to +3.56) | +0.64 (+0.04 to +1.20) | +0.80 (+0.32 to +1.28) | +0.99 (+0.44 to +1.54) |
| best detector at a 30% budget | T1_k0_cot_multi | +3.08 (+2.24 to +3.92) | +0.52 (-0.08 to +1.12) | +1.12 (+0.60 to +1.68) | +1.49 (+0.84 to +2.13) |
| best detector at a 50% budget | T1_k0_cot_multi | +3.28 (+2.44 to +4.16) | +0.48 (-0.16 to +1.08) | +1.36 (+0.80 to +1.96) | +1.54 (+0.79 to +2.28) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +2.84 (+2.04 to +3.68) | +0.56 (-0.04 to +1.16) | +1.12 (+0.60 to +1.64) | +1.34 (+0.74 to +1.93) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +3.04 (+2.24 to +3.84) | +1.16 (+0.48 to +1.80) | +1.12 (+0.48 to +1.76) | +1.54 (+0.84 to +2.23) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +2.32 (+1.56 to +3.08) | +1.08 (+0.48 to +1.68) | +0.76 (+0.24 to +1.28) | +0.99 (+0.44 to +1.59) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +2.76 (+1.96 to +3.56) | +1.20 (+0.56 to +1.84) | +1.04 (+0.44 to +1.64) | +1.44 (+0.79 to +2.08) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +3.00 (+2.20 to +3.80) | +1.16 (+0.48 to +1.80) | +1.20 (+0.56 to +1.84) | +1.49 (+0.79 to +2.18) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +2.56 (+1.84 to +3.32) | +1.12 (+0.48 to +1.76) | +1.00 (+0.44 to +1.56) | +1.29 (+0.64 to +1.93) |
| best under 2 passes per question (any configuration) | T1_k0_cot | +2.56 (+1.76 to +3.40) | +0.52 (-0.12 to +1.16) | +0.76 (+0.28 to +1.24) | +0.69 (+0.05 to +1.29) |
| best under 3 passes per question (any configuration) | T1_k0_cot | +3.00 (+2.12 to +3.92) | +0.52 (-0.12 to +1.16) | +1.20 (+0.60 to +1.80) | +1.04 (+0.30 to +1.78) |
| best under 5 passes per question (any configuration) | T1_k0_cot_multi | +3.20 (+2.36 to +4.04) | +0.52 (-0.08 to +1.12) | +1.32 (+0.76 to +1.88) | +1.59 (+0.89 to +2.23) |
