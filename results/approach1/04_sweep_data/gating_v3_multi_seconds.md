# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus 2 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.80% vs 85.80%, T1_k0_ground_cot 85.97% vs 85.97%, T1_k0_cot 85.36% vs 85.36%, T1_k0_zoom_cot 85.73% vs 85.73%, T1_k0_cot_multi 86.00% vs 86.00%, T1_k0_ground_cot_multi 86.38% vs 86.38%

Seconds: estimated wall-clock seconds per question on one H100, from the measured component times (results/latency.json).

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Seconds | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 24.13 | 16.00 | 4.63 | 85.80% | +3.03 (+2.50 to +3.54) | +463 / -175 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 6.39 | 3.92 | 1.05 | 85.19% | +2.42 (+1.98 to +2.84) | +338 / -108 | +1.77 (+1.37 to +2.16) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 1.51 | 85.63% | +2.85 (+2.37 to +3.33) | +412 / -141 | +1.93 (+1.51 to +2.35) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 13.31 | 8.46 | 2.41 | 85.84% | +3.06 (+2.54 to +3.56) | +459 / -168 | +1.71 (+1.33 to +2.08) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 1.51 | 85.63% | +2.85 (+2.37 to +3.33) | +412 / -141 | +1.93 (+1.51 to +2.35) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 8.53 | 5.00 | 2.91 | 85.97% | +3.19 (+2.69 to +3.71) | +457 / -153 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | after_head | 20% / 19.2% | 3.55 | 1.77 | 0.72 | 85.57% | +2.79 (+2.36 to +3.24) | +358 / -92 | +2.11 (+1.73 to +2.52) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 0.97 | 85.78% | +3.01 (+2.53 to +3.48) | +405 / -119 | +2.06 (+1.66 to +2.48) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 5.11 | 2.99 | 1.53 | 85.97% | +3.19 (+2.70 to +3.71) | +453 / -149 | +1.65 (+1.28 to +2.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 0.97 | 85.78% | +3.01 (+2.53 to +3.48) | +405 / -119 | +2.06 (+1.66 to +2.48) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 4.52 | 2.00 | 1.87 | 85.36% | +2.58 (+2.06 to +3.11) | +436 / -190 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 0.47 | 84.95% | +2.18 (+1.76 to +2.56) | +301 / -94 | +1.60 (+1.22 to +1.97) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 0.64 | 85.21% | +2.43 (+1.94 to +2.88) | +366 / -135 | +1.67 (+1.26 to +2.09) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.93 | 1.50 | 1.00 | 85.43% | +2.65 (+2.12 to +3.18) | +432 / -180 | +1.43 (+1.05 to +1.81) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 0.64 | 85.21% | +2.43 (+1.94 to +2.88) | +366 / -135 | +1.67 (+1.26 to +2.09) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 12.57 | 6.00 | 2.69 | 85.73% | +2.95 (+2.41 to +3.46) | +457 / -176 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 19.4% | 3.53 | 1.97 | 0.63 | 85.15% | +2.37 (+1.93 to +2.80) | +327 / -101 | +1.67 (+1.27 to +2.05) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 0.90 | 85.57% | +2.79 (+2.30 to +3.26) | +401 / -135 | +1.87 (+1.46 to +2.28) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 7.04 | 3.49 | 1.41 | 85.76% | +2.98 (+2.45 to +3.49) | +453 / -169 | +1.65 (+1.25 to +2.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 0.90 | 85.57% | +2.79 (+2.30 to +3.26) | +401 / -135 | +1.87 (+1.46 to +2.28) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 8.10 | 5.00 | 3.83 | 86.00% | +3.23 (+2.73 to +3.72) | +450 / -143 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 19.4% | 2.72 | 1.78 | 0.86 | 85.25% | +2.47 (+2.06 to +2.89) | +321 / -86 | +1.72 (+1.34 to +2.09) |
| best detector at a 30% budget | T1_k0_cot_multi | token | 30% / 29.6% | 3.50 | 2.18 | 1.24 | 85.69% | +2.91 (+2.44 to +3.37) | +388 / -111 | +1.91 (+1.52 to +2.31) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 49.7% | 4.93 | 2.99 | 1.99 | 86.03% | +3.25 (+2.75 to +3.74) | +446 / -137 | +1.61 (+1.25 to +1.95) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 35% / 34.9% | 3.89 | 2.39 | 1.44 | 85.92% | +3.14 (+2.66 to +3.61) | +416 / -117 | +1.95 (+1.58 to +2.33) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 12.02 | 8.00 | 4.87 | 86.38% | +3.60 (+3.10 to +4.09) | +473 / -130 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 19.4% | 3.66 | 2.36 | 1.07 | 85.44% | +2.66 (+2.24 to +3.08) | +338 / -85 | +1.89 (+1.50 to +2.27) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 29.6% | 4.85 | 3.07 | 1.56 | 86.01% | +3.24 (+2.76 to +3.69) | +416 / -108 | +2.14 (+1.75 to +2.54) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 49.7% | 7.04 | 4.48 | 2.51 | 86.38% | +3.60 (+3.10 to +4.10) | +469 / -126 | +1.77 (+1.41 to +2.11) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 29.6% | 4.85 | 3.07 | 1.56 | 86.01% | +3.24 (+2.76 to +3.69) | +416 / -108 | +2.14 (+1.75 to +2.54) |
| best under 2 passes per question (any configuration) | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 0.47 | 84.95% | +2.18 (+1.76 to +2.56) | +301 / -94 | +1.60 (+1.22 to +1.97) |
| best under 3 passes per question (any configuration) | T1_k0_cot | token | 40% / 39.8% | 2.59 | 1.40 | 0.83 | 85.44% | +2.66 (+2.13 to +3.16) | +412 / -159 | +1.67 (+1.27 to +2.09) |
| best under 5 passes per question (any configuration) | T1_k0_cot_multi | token | 50% / 49.7% | 4.93 | 2.99 | 1.99 | 86.03% | +3.25 (+2.75 to +3.74) | +446 / -137 | +1.61 (+1.25 to +1.95) |
| best under 0.3 s per question on one H100 (any configuration) | T1_k0_cot | token | 10% / 9.4% | 1.41 | 1.09 | 0.29 | 84.41% | +1.63 (+1.31 to +1.94) | +196 / -41 | +1.29 (+0.97 to +1.62) |
| best under 0.5 s per question on one H100 (any configuration) | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 0.47 | 84.95% | +2.18 (+1.76 to +2.56) | +301 / -94 | +1.60 (+1.22 to +1.97) |
| best under 1 s per question on one H100 (any configuration) | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 0.97 | 85.78% | +3.01 (+2.53 to +3.48) | +405 / -119 | +2.06 (+1.66 to +2.48) |
| best under 2 s per question on one H100 (any configuration) | T1_k0_ground_cot_multi | token | 35% / 34.9% | 5.45 | 3.44 | 1.81 | 86.22% | +3.45 (+2.97 to +3.92) | +440 / -112 | +2.13 (+1.74 to +2.51) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +5.64 (+4.44 to +6.88) | +2.72 (+1.72 to +3.68) | +1.40 (+0.68 to +2.04) | +2.18 (+0.94 to +3.47) |
| best detector at a 20% budget | T1_k5_everything | +4.56 (+3.56 to +5.64) | +2.48 (+1.64 to +3.36) | +0.80 (+0.28 to +1.32) | +1.69 (+0.74 to +2.68) |
| best detector at a 30% budget | T1_k5_everything | +5.24 (+4.12 to +6.36) | +2.76 (+1.84 to +3.68) | +1.12 (+0.48 to +1.72) | +2.13 (+1.04 to +3.27) |
| best detector at a 50% budget | T1_k5_everything | +5.52 (+4.32 to +6.76) | +2.72 (+1.72 to +3.68) | +1.44 (+0.76 to +2.08) | +2.43 (+1.24 to +3.67) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +5.24 (+4.12 to +6.36) | +2.76 (+1.84 to +3.68) | +1.12 (+0.48 to +1.72) | +2.13 (+1.04 to +3.27) |
| correct every question (no gate) | T1_k0_ground_cot | +6.24 (+5.00 to +7.56) | +2.48 (+1.64 to +3.40) | +1.28 (+0.60 to +1.96) | +2.68 (+1.49 to +3.92) |
| best detector at a 20% budget | T1_k0_ground_cot | +5.88 (+4.72 to +7.00) | +1.64 (+0.92 to +2.40) | +0.72 (+0.28 to +1.16) | +2.97 (+1.98 to +4.07) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.84 (+4.72 to +7.00) | +2.52 (+1.72 to +3.36) | +1.04 (+0.40 to +1.64) | +2.53 (+1.54 to +3.62) |
| best detector at a 50% budget | T1_k0_ground_cot | +6.12 (+4.88 to +7.40) | +2.48 (+1.64 to +3.40) | +1.32 (+0.64 to +2.00) | +2.78 (+1.64 to +3.97) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.84 (+4.72 to +7.00) | +2.52 (+1.72 to +3.36) | +1.04 (+0.40 to +1.64) | +2.53 (+1.54 to +3.62) |
| correct every question (no gate) | T1_k0_cot | +6.72 (+5.52 to +8.00) | +0.64 (-0.20 to +1.52) | +0.48 (-0.24 to +1.16) | +2.48 (+1.29 to +3.77) |
| best detector at a 20% budget | T1_k0_cot | +5.20 (+4.16 to +6.28) | +1.12 (+0.40 to +1.84) | +0.40 (-0.08 to +0.88) | +1.93 (+1.09 to +2.88) |
| best detector at a 30% budget | T1_k0_cot | +5.96 (+4.84 to +7.12) | +0.96 (+0.20 to +1.72) | +0.40 (-0.24 to +1.00) | +2.38 (+1.39 to +3.52) |
| best detector at a 50% budget | T1_k0_cot | +6.60 (+5.36 to +7.84) | +0.76 (-0.08 to +1.64) | +0.48 (-0.24 to +1.16) | +2.78 (+1.59 to +4.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.96 (+4.84 to +7.12) | +0.96 (+0.20 to +1.72) | +0.40 (-0.24 to +1.00) | +2.38 (+1.39 to +3.52) |
| correct every question (no gate) | T1_k0_zoom_cot | +6.08 (+4.84 to +7.32) | +2.16 (+1.20 to +3.08) | +0.92 (+0.20 to +1.60) | +2.58 (+1.44 to +3.82) |
| best detector at a 20% budget | T1_k0_zoom_cot | +4.88 (+3.84 to +5.96) | +2.08 (+1.28 to +2.88) | +0.56 (+0.04 to +1.08) | +1.88 (+0.99 to +2.83) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.68 (+4.52 to +6.84) | +2.28 (+1.40 to +3.12) | +0.76 (+0.12 to +1.36) | +2.38 (+1.34 to +3.52) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.96 (+4.72 to +7.20) | +2.20 (+1.24 to +3.12) | +0.96 (+0.24 to +1.68) | +2.78 (+1.63 to +4.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.68 (+4.52 to +6.84) | +2.28 (+1.40 to +3.12) | +0.76 (+0.12 to +1.36) | +2.38 (+1.34 to +3.52) |
| correct every question (no gate) | T1_k0_cot_multi | +7.12 (+5.88 to +8.36) | +1.56 (+0.80 to +2.44) | +1.12 (+0.44 to +1.76) | +3.07 (+1.83 to +4.36) |
| best detector at a 20% budget | T1_k0_cot_multi | +5.40 (+4.32 to +6.48) | +1.64 (+0.96 to +2.36) | +0.72 (+0.20 to +1.24) | +2.03 (+1.09 to +3.07) |
| best detector at a 30% budget | T1_k0_cot_multi | +6.32 (+5.16 to +7.48) | +1.68 (+0.96 to +2.44) | +1.00 (+0.36 to +1.60) | +2.58 (+1.49 to +3.77) |
| best detector at a 50% budget | T1_k0_cot_multi | +7.00 (+5.76 to +8.24) | +1.64 (+0.88 to +2.48) | +1.12 (+0.44 to +1.76) | +3.22 (+2.03 to +4.46) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +6.72 (+5.52 to +7.96) | +1.72 (+1.00 to +2.52) | +1.04 (+0.40 to +1.64) | +3.07 (+1.93 to +4.26) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +7.00 (+5.76 to +8.24) | +2.68 (+1.80 to +3.60) | +1.36 (+0.68 to +2.04) | +3.32 (+2.18 to +4.56) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +5.20 (+4.16 to +6.28) | +2.44 (+1.68 to +3.24) | +0.76 (+0.24 to +1.28) | +2.13 (+1.19 to +3.17) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +6.32 (+5.20 to +7.48) | +2.60 (+1.76 to +3.48) | +1.20 (+0.60 to +1.80) | +2.73 (+1.64 to +3.92) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +6.88 (+5.68 to +8.12) | +2.68 (+1.80 to +3.60) | +1.40 (+0.72 to +2.08) | +3.42 (+2.28 to +4.66) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +6.32 (+5.20 to +7.48) | +2.60 (+1.76 to +3.48) | +1.20 (+0.60 to +1.80) | +2.73 (+1.64 to +3.92) |
| best under 2 passes per question (any configuration) | T1_k0_cot | +5.20 (+4.16 to +6.28) | +1.12 (+0.40 to +1.84) | +0.40 (-0.08 to +0.88) | +1.93 (+1.09 to +2.88) |
| best under 3 passes per question (any configuration) | T1_k0_cot | +6.44 (+5.20 to +7.64) | +0.92 (+0.12 to +1.76) | +0.48 (-0.20 to +1.12) | +2.83 (+1.74 to +4.02) |
| best under 5 passes per question (any configuration) | T1_k0_cot_multi | +7.00 (+5.76 to +8.24) | +1.64 (+0.88 to +2.48) | +1.12 (+0.44 to +1.76) | +3.22 (+2.03 to +4.46) |
| best under 0.3 s per question on one H100 (any configuration) | T1_k0_cot | +3.80 (+2.92 to +4.68) | +0.76 (+0.28 to +1.24) | +0.48 (+0.20 to +0.80) | +1.44 (+0.79 to +2.13) |
| best under 0.5 s per question on one H100 (any configuration) | T1_k0_cot | +5.20 (+4.16 to +6.28) | +1.12 (+0.40 to +1.84) | +0.40 (-0.08 to +0.88) | +1.93 (+1.09 to +2.88) |
| best under 1 s per question on one H100 (any configuration) | T1_k0_ground_cot | +5.84 (+4.72 to +7.00) | +2.52 (+1.72 to +3.36) | +1.04 (+0.40 to +1.64) | +2.53 (+1.54 to +3.62) |
| best under 2 s per question on one H100 (any configuration) | T1_k0_ground_cot_multi | +6.68 (+5.52 to +7.88) | +2.64 (+1.80 to +3.52) | +1.24 (+0.60 to +1.88) | +3.17 (+2.08 to +4.31) |
