# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_esc_cot8b_drop_judge_models.pt`. Greedy on test: 85.84%. Passes: model passes per question (generation calls plus 1 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only. Ranker inputs zeroed: judge.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 87.01% vs 87.01%, T1_k0_ground_cot 87.18% vs 87.18%, T1_k0_cot 87.19% vs 87.19%, T1_k0_zoom_cot 87.08% vs 87.08%, T1_k0_cot_multi 87.48% vs 87.48%, T1_k0_ground_cot_multi 87.48% vs 87.48%, T1_k0_esc8b 87.95% vs 87.95%, T1_k0_ground_cot_multi_esc8b 88.74% vs 88.74%, T1_k0_ground_cot_multi_esc 89.45% vs 89.45%, T1_k0_everything_multi_esc 89.38% vs 89.38%, T1_k0_cot8b 88.09% vs 88.09%, T1_k0_ground_cot_multi_cot8b 88.53% vs 88.53%, T1_k0_ground_cot_multi_esc8b_cot8b 89.45% vs 89.45%



| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Seconds | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 19.87 | 16.00 | 0.00 | 87.01% | +1.18 (+0.86 to +1.50) | +178 / -66 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 20.1% | 5.20 | 4.01 | 0.00 | 86.88% | +1.04 (+0.76 to +1.32) | +148 / -49 | +0.81 (+0.54 to +1.08) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 31.2% | 7.35 | 5.69 | 0.00 | 87.00% | +1.17 (+0.87 to +1.47) | +170 / -59 | +0.90 (+0.64 to +1.17) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 52.4% | 11.30 | 8.86 | 0.00 | 87.01% | +1.18 (+0.86 to +1.50) | +178 / -66 | +0.53 (+0.30 to +0.74) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 15% / 14.5% | 4.10 | 3.18 | 0.00 | 86.72% | +0.88 (+0.63 to +1.13) | +122 / -38 | +0.73 (+0.47 to +0.98) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 6.72 | 5.00 | 0.00 | 87.18% | +1.34 (+1.03 to +1.68) | +205 / -77 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 20.1% | 2.31 | 1.80 | 0.00 | 86.87% | +1.03 (+0.75 to +1.31) | +153 / -55 | +0.83 (+0.55 to +1.10) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 31.2% | 2.97 | 2.25 | 0.00 | 87.09% | +1.25 (+0.95 to +1.57) | +186 / -67 | +0.99 (+0.73 to +1.27) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 52.4% | 4.17 | 3.09 | 0.00 | 87.13% | +1.29 (+0.98 to +1.62) | +200 / -77 | +0.67 (+0.44 to +0.91) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 25% / 25.3% | 2.62 | 2.01 | 0.00 | 87.03% | +1.20 (+0.89 to +1.50) | +173 / -59 | +0.97 (+0.69 to +1.26) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 3.27 | 2.00 | 0.00 | 87.19% | +1.36 (+0.99 to +1.72) | +225 / -96 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 20.1% | 1.53 | 1.20 | 0.00 | 86.94% | +1.10 (+0.79 to +1.41) | +160 / -55 | +0.91 (+0.60 to +1.21) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 31.2% | 1.80 | 1.31 | 0.00 | 87.14% | +1.30 (+0.97 to +1.64) | +195 / -71 | +1.12 (+0.82 to +1.41) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 52.4% | 2.27 | 1.52 | 0.00 | 87.20% | +1.37 (+1.00 to +1.73) | +219 / -89 | +0.90 (+0.65 to +1.17) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 25% / 25.3% | 1.65 | 1.25 | 0.00 | 87.11% | +1.27 (+0.95 to +1.59) | +181 / -60 | +1.05 (+0.75 to +1.34) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 9.29 | 6.00 | 0.00 | 87.08% | +1.24 (+0.94 to +1.54) | +179 / -61 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 20.1% | 2.85 | 2.00 | 0.00 | 86.88% | +1.04 (+0.76 to +1.32) | +146 / -47 | +0.86 (+0.60 to +1.13) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 31.2% | 3.82 | 2.56 | 0.00 | 87.04% | +1.21 (+0.91 to +1.51) | +171 / -56 | +1.00 (+0.74 to +1.26) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 52.4% | 5.58 | 3.62 | 0.00 | 87.09% | +1.25 (+0.96 to +1.57) | +178 / -59 | +0.62 (+0.41 to +0.84) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 20% / 20.1% | 2.85 | 2.00 | 0.00 | 86.88% | +1.04 (+0.76 to +1.32) | +146 / -47 | +0.86 (+0.60 to +1.13) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 6.50 | 5.00 | 0.00 | 87.48% | +1.64 (+1.31 to +1.96) | +219 / -63 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 20.1% | 2.25 | 1.80 | 0.00 | 87.12% | +1.28 (+1.01 to +1.58) | +160 / -38 | +0.97 (+0.69 to +1.25) |
| best detector at a 30% budget | T1_k0_cot_multi | token | 30% / 31.2% | 2.89 | 2.25 | 0.00 | 87.37% | +1.53 (+1.23 to +1.84) | +192 / -46 | +1.19 (+0.90 to +1.46) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 52.4% | 4.04 | 3.09 | 0.00 | 87.48% | +1.64 (+1.32 to +1.96) | +213 / -57 | +0.89 (+0.66 to +1.13) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 30% / 31.2% | 2.89 | 2.25 | 0.00 | 87.37% | +1.53 (+1.23 to +1.84) | +192 / -46 | +1.19 (+0.90 to +1.46) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 9.92 | 8.00 | 0.00 | 87.48% | +1.64 (+1.31 to +1.96) | +217 / -61 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 20.1% | 3.01 | 2.41 | 0.00 | 87.12% | +1.28 (+0.99 to +1.57) | +164 / -42 | +0.92 (+0.64 to +1.20) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 31.2% | 4.05 | 3.19 | 0.00 | 87.38% | +1.54 (+1.24 to +1.84) | +196 / -49 | +1.07 (+0.80 to +1.34) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 52.4% | 5.91 | 4.67 | 0.00 | 87.43% | +1.60 (+1.28 to +1.93) | +212 / -60 | +0.71 (+0.48 to +0.96) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 31.2% | 4.05 | 3.19 | 0.00 | 87.38% | +1.54 (+1.24 to +1.84) | +196 / -49 | +1.07 (+0.80 to +1.34) |
| correct every question (no gate) | T1_k0_esc8b | token | 100% / 100.0% | 4.18 | 3.00 | 0.00 | 87.95% | +2.11 (+1.59 to +2.62) | +417 / -216 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_esc8b | token | 20% / 20.1% | 1.72 | 1.40 | 0.00 | 87.49% | +1.65 (+1.26 to +2.02) | +262 / -105 | +1.28 (+0.90 to +1.64) |
| best detector at a 30% budget | T1_k0_esc8b | token | 30% / 31.2% | 2.08 | 1.62 | 0.00 | 87.98% | +2.14 (+1.70 to +2.59) | +346 / -142 | +1.50 (+1.11 to +1.88) |
| best detector at a 50% budget | T1_k0_esc8b | token | 50% / 52.4% | 2.74 | 2.05 | 0.00 | 87.93% | +2.09 (+1.58 to +2.57) | +399 / -200 | +1.04 (+0.68 to +1.42) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | token | 20% / 20.1% | 1.72 | 1.40 | 0.00 | 87.49% | +1.65 (+1.26 to +2.02) | +262 / -105 | +1.28 (+0.90 to +1.64) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | token | 100% / 100.0% | 12.02 | 10.00 | 0.00 | 88.74% | +2.90 (+2.43 to +3.38) | +425 / -149 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | token | 20% / 20.1% | 3.48 | 2.81 | 0.00 | 87.85% | +2.02 (+1.65 to +2.37) | +270 / -78 | +1.45 (+1.08 to +1.81) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | token | 30% / 31.2% | 4.75 | 3.81 | 0.00 | 88.45% | +2.62 (+2.21 to +3.04) | +356 / -107 | +1.72 (+1.34 to +2.09) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | token | 50% / 52.4% | 7.05 | 5.71 | 0.00 | 88.61% | +2.77 (+2.30 to +3.25) | +407 / -143 | +1.29 (+0.96 to +1.65) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | token | 25% / 25.3% | 4.08 | 3.28 | 0.00 | 88.27% | +2.44 (+2.03 to +2.85) | +324 / -92 | +1.69 (+1.31 to +2.06) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | token | 100% / 100.0% | 20.37 | 18.25 | 0.00 | 89.45% | +3.61 (+3.14 to +4.08) | +458 / -114 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | token | 20% / 20.1% | 5.19 | 4.47 | 0.00 | 88.27% | +2.44 (+2.06 to +2.78) | +292 / -60 | +1.69 (+1.32 to +2.05) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | token | 30% / 31.2% | 7.39 | 6.39 | 0.00 | 89.01% | +3.17 (+2.74 to +3.59) | +378 / -76 | +2.08 (+1.69 to +2.46) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | token | 50% / 52.4% | 11.45 | 10.03 | 0.00 | 89.28% | +3.45 (+2.98 to +3.88) | +435 / -107 | +1.53 (+1.19 to +1.88) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | token | 30% / 31.2% | 7.39 | 6.39 | 0.00 | 89.01% | +3.17 (+2.74 to +3.59) | +378 / -76 | +2.08 (+1.69 to +2.46) |
| correct every question (no gate) | T1_k0_everything_multi_esc | token | 100% / 100.0% | 28.30 | 24.25 | 0.00 | 89.38% | +3.54 (+3.06 to +4.01) | +455 / -118 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | token | 20% / 20.1% | 6.89 | 5.67 | 0.00 | 88.20% | +2.36 (+1.96 to +2.73) | +291 / -66 | +1.58 (+1.20 to +1.94) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | token | 30% / 31.2% | 10.01 | 8.27 | 0.00 | 88.94% | +3.10 (+2.68 to +3.52) | +379 / -84 | +2.00 (+1.62 to +2.36) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | token | 50% / 52.4% | 15.75 | 13.18 | 0.00 | 89.20% | +3.36 (+2.89 to +3.81) | +434 / -114 | +1.49 (+1.15 to +1.85) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | token | 35% / 36.6% | 11.49 | 9.52 | 0.00 | 89.14% | +3.30 (+2.87 to +3.74) | +403 / -89 | +1.96 (+1.59 to +2.33) |
| correct every question (no gate) | T1_k0_cot8b | token | 100% / 100.0% | 4.33 | 3.00 | 0.00 | 88.09% | +2.26 (+1.71 to +2.78) | +439 / -224 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot8b | token | 20% / 20.1% | 1.74 | 1.40 | 0.00 | 87.42% | +1.59 (+1.17 to +1.94) | +259 / -108 | +1.12 (+0.73 to +1.49) |
| best detector at a 30% budget | T1_k0_cot8b | judge | 30% / 26.5% | 2.65 | 1.53 | 0.00 | 88.01% | +2.18 (+1.73 to +2.57) | +312 / -105 | +1.56 (+1.16 to +1.93) |
| best detector at a 50% budget | T1_k0_cot8b | judge | 50% / 46.7% | 3.11 | 1.93 | 0.00 | 88.18% | +2.34 (+1.87 to +2.77) | +364 / -141 | +1.24 (+0.85 to +1.61) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | judge | 25% / 22.7% | 2.56 | 1.45 | 0.00 | 87.81% | +1.98 (+1.54 to +2.35) | +282 / -94 | +1.44 (+1.04 to +1.82) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | token | 100% / 100.0% | 12.04 | 10.00 | 0.00 | 88.53% | +2.69 (+2.14 to +3.23) | +490 / -234 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | token | 20% / 20.1% | 3.47 | 2.81 | 0.00 | 87.80% | +1.96 (+1.53 to +2.34) | +293 / -106 | +1.40 (+0.97 to +1.80) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | judge | 30% / 26.5% | 4.78 | 3.38 | 0.00 | 88.38% | +2.54 (+2.09 to +2.97) | +339 / -97 | +1.77 (+1.34 to +2.18) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | judge | 50% / 46.7% | 6.78 | 5.20 | 0.00 | 88.63% | +2.79 (+2.32 to +3.27) | +398 / -132 | +1.40 (+1.01 to +1.80) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | judge | 25% / 22.7% | 4.40 | 3.04 | 0.00 | 88.13% | +2.29 (+1.87 to +2.68) | +309 / -91 | +1.64 (+1.23 to +2.04) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | token | 100% / 100.0% | 14.12 | 12.00 | 0.00 | 89.45% | +3.61 (+3.14 to +4.08) | +458 / -114 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | token | 20% / 20.1% | 3.93 | 3.21 | 0.00 | 88.27% | +2.44 (+2.06 to +2.78) | +292 / -60 | +1.69 (+1.32 to +2.05) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | token | 30% / 31.2% | 5.44 | 4.44 | 0.00 | 89.01% | +3.17 (+2.74 to +3.59) | +378 / -76 | +2.08 (+1.69 to +2.46) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | token | 50% / 52.4% | 8.18 | 6.76 | 0.00 | 89.28% | +3.45 (+2.98 to +3.88) | +435 / -107 | +1.53 (+1.19 to +1.88) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | token | 30% / 31.2% | 5.44 | 4.44 | 0.00 | 89.01% | +3.17 (+2.74 to +3.59) | +378 / -76 | +2.08 (+1.69 to +2.46) |
| best under 2 passes per question (any configuration) | T1_k0_cot | token | 35% / 36.6% | 1.92 | 1.37 | 0.00 | 87.24% | +1.41 (+1.06 to +1.74) | +207 / -73 | +1.16 (+0.87 to +1.43) |
| best under 3 passes per question (any configuration) | T1_k0_ground_cot_multi_cot8b | token | 15% / 14.5% | 2.82 | 2.31 | 0.00 | 87.56% | +1.72 (+1.34 to +2.06) | +240 / -76 | +1.36 (+0.97 to +1.71) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot_multi_esc8b_cot8b | token | 25% / 25.3% | 4.64 | 3.78 | 0.00 | 88.77% | +2.93 (+2.53 to +3.32) | +346 / -67 | +2.02 (+1.63 to +2.40) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +1.40 (+0.72 to +2.08) | +1.16 (+0.52 to +1.80) | +1.08 (+0.56 to +1.56) | +1.04 (+0.30 to +1.78) |
| best detector at a 20% budget | T1_k5_everything | +1.40 (+0.80 to +2.04) | +1.04 (+0.40 to +1.64) | +0.88 (+0.40 to +1.32) | +0.79 (+0.20 to +1.39) |
| best detector at a 30% budget | T1_k5_everything | +1.44 (+0.80 to +2.12) | +1.16 (+0.52 to +1.76) | +1.04 (+0.52 to +1.52) | +0.99 (+0.30 to +1.69) |
| best detector at a 50% budget | T1_k5_everything | +1.40 (+0.72 to +2.08) | +1.16 (+0.52 to +1.80) | +1.08 (+0.56 to +1.56) | +1.04 (+0.30 to +1.78) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +1.52 (+0.92 to +2.12) | +0.84 (+0.28 to +1.40) | +0.60 (+0.24 to +1.00) | +0.50 (-0.00 to +1.04) |
| correct every question (no gate) | T1_k0_ground_cot | +1.96 (+1.20 to +2.76) | +0.88 (+0.28 to +1.52) | +1.48 (+0.92 to +2.04) | +0.99 (+0.15 to +1.78) |
| best detector at a 20% budget | T1_k0_ground_cot | +1.64 (+0.96 to +2.36) | +0.80 (+0.24 to +1.40) | +0.96 (+0.52 to +1.44) | +0.64 (+0.05 to +1.29) |
| best detector at a 30% budget | T1_k0_ground_cot | +1.84 (+1.12 to +2.60) | +0.92 (+0.32 to +1.52) | +1.28 (+0.72 to +1.80) | +0.89 (+0.15 to +1.64) |
| best detector at a 50% budget | T1_k0_ground_cot | +1.88 (+1.08 to +2.68) | +0.88 (+0.28 to +1.52) | +1.48 (+0.92 to +2.04) | +0.84 (+0.05 to +1.64) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +1.76 (+1.04 to +2.48) | +0.84 (+0.28 to +1.44) | +1.20 (+0.72 to +1.72) | +0.94 (+0.25 to +1.64) |
| correct every question (no gate) | T1_k0_cot | +3.28 (+2.36 to +4.24) | +0.20 (-0.32 to +0.72) | +1.16 (+0.64 to +1.68) | +0.64 (-0.30 to +1.59) |
| best detector at a 20% budget | T1_k0_cot | +2.56 (+1.76 to +3.40) | +0.32 (-0.20 to +0.84) | +0.76 (+0.36 to +1.16) | +0.69 (+0.05 to +1.29) |
| best detector at a 30% budget | T1_k0_cot | +2.96 (+2.08 to +3.88) | +0.24 (-0.28 to +0.76) | +1.04 (+0.56 to +1.52) | +0.89 (+0.15 to +1.64) |
| best detector at a 50% budget | T1_k0_cot | +3.24 (+2.36 to +4.20) | +0.20 (-0.32 to +0.72) | +1.20 (+0.68 to +1.72) | +0.69 (-0.20 to +1.59) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +2.76 (+1.88 to +3.64) | +0.28 (-0.24 to +0.80) | +1.00 (+0.56 to +1.44) | +0.99 (+0.35 to +1.69) |
| correct every question (no gate) | T1_k0_zoom_cot | +1.76 (+1.08 to +2.48) | +0.96 (+0.36 to +1.56) | +1.16 (+0.64 to +1.68) | +1.04 (+0.30 to +1.78) |
| best detector at a 20% budget | T1_k0_zoom_cot | +1.64 (+0.96 to +2.36) | +0.88 (+0.28 to +1.44) | +0.84 (+0.44 to +1.24) | +0.74 (+0.10 to +1.39) |
| best detector at a 30% budget | T1_k0_zoom_cot | +1.72 (+1.04 to +2.44) | +0.96 (+0.36 to +1.56) | +1.08 (+0.60 to +1.56) | +1.04 (+0.35 to +1.74) |
| best detector at a 50% budget | T1_k0_zoom_cot | +1.72 (+1.04 to +2.44) | +0.96 (+0.36 to +1.56) | +1.24 (+0.72 to +1.72) | +1.04 (+0.30 to +1.78) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +1.64 (+0.96 to +2.36) | +0.88 (+0.28 to +1.44) | +0.84 (+0.44 to +1.24) | +0.74 (+0.10 to +1.39) |
| correct every question (no gate) | T1_k0_cot_multi | +3.48 (+2.64 to +4.36) | +0.52 (+0.04 to +1.00) | +1.12 (+0.60 to +1.64) | +1.39 (+0.55 to +2.18) |
| best detector at a 20% budget | T1_k0_cot_multi | +2.88 (+2.12 to +3.68) | +0.56 (+0.08 to +1.04) | +0.76 (+0.36 to +1.20) | +0.84 (+0.25 to +1.44) |
| best detector at a 30% budget | T1_k0_cot_multi | +3.28 (+2.44 to +4.12) | +0.52 (+0.04 to +0.96) | +1.00 (+0.52 to +1.48) | +1.29 (+0.59 to +1.98) |
| best detector at a 50% budget | T1_k0_cot_multi | +3.44 (+2.60 to +4.32) | +0.52 (+0.04 to +1.00) | +1.20 (+0.68 to +1.72) | +1.34 (+0.55 to +2.13) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +3.28 (+2.44 to +4.12) | +0.52 (+0.04 to +0.96) | +1.00 (+0.52 to +1.48) | +1.29 (+0.59 to +1.98) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +2.64 (+1.88 to +3.44) | +1.04 (+0.40 to +1.68) | +1.24 (+0.68 to +1.80) | +1.64 (+0.89 to +2.38) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +2.28 (+1.60 to +3.00) | +0.88 (+0.32 to +1.44) | +0.96 (+0.52 to +1.40) | +0.94 (+0.35 to +1.54) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +2.52 (+1.80 to +3.24) | +1.08 (+0.48 to +1.68) | +1.16 (+0.64 to +1.72) | +1.39 (+0.69 to +2.03) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +2.56 (+1.80 to +3.32) | +1.04 (+0.40 to +1.68) | +1.28 (+0.72 to +1.84) | +1.49 (+0.69 to +2.23) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +2.52 (+1.80 to +3.24) | +1.08 (+0.48 to +1.68) | +1.16 (+0.64 to +1.72) | +1.39 (+0.69 to +2.03) |
| correct every question (no gate) | T1_k0_esc8b | +2.04 (+0.92 to +3.20) | +2.52 (+1.40 to +3.68) | +1.88 (+1.16 to +2.64) | +1.98 (+0.84 to +3.03) |
| best detector at a 20% budget | T1_k0_esc8b | +1.80 (+0.96 to +2.72) | +2.12 (+1.16 to +3.08) | +1.04 (+0.48 to +1.60) | +1.64 (+0.94 to +2.33) |
| best detector at a 30% budget | T1_k0_esc8b | +2.24 (+1.28 to +3.28) | +2.56 (+1.52 to +3.64) | +1.52 (+0.84 to +2.20) | +2.28 (+1.44 to +3.07) |
| best detector at a 50% budget | T1_k0_esc8b | +2.00 (+0.92 to +3.12) | +2.48 (+1.36 to +3.64) | +1.84 (+1.12 to +2.60) | +2.03 (+1.04 to +3.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | +1.80 (+0.96 to +2.72) | +2.12 (+1.16 to +3.08) | +1.04 (+0.48 to +1.60) | +1.64 (+0.94 to +2.33) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | +3.36 (+2.32 to +4.48) | +3.00 (+1.96 to +4.12) | +2.60 (+1.84 to +3.36) | +2.58 (+1.59 to +3.57) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | +2.36 (+1.60 to +3.20) | +2.44 (+1.56 to +3.40) | +1.52 (+0.92 to +2.12) | +1.69 (+0.99 to +2.33) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | +3.00 (+2.12 to +4.00) | +2.92 (+1.92 to +3.96) | +2.20 (+1.52 to +2.92) | +2.28 (+1.49 to +3.07) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | +3.12 (+2.16 to +4.16) | +2.92 (+1.88 to +4.04) | +2.56 (+1.80 to +3.32) | +2.43 (+1.49 to +3.37) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | +2.84 (+2.00 to +3.76) | +2.72 (+1.76 to +3.72) | +2.04 (+1.40 to +2.76) | +2.08 (+1.34 to +2.83) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | +5.92 (+4.84 to +7.04) | +2.68 (+1.72 to +3.68) | +2.52 (+1.76 to +3.28) | +3.27 (+2.28 to +4.21) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | +4.04 (+3.20 to +4.88) | +2.20 (+1.36 to +3.08) | +1.52 (+0.92 to +2.12) | +1.88 (+1.24 to +2.58) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | +5.12 (+4.16 to +6.08) | +2.68 (+1.80 to +3.64) | +2.20 (+1.52 to +2.92) | +2.58 (+1.78 to +3.37) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | +5.64 (+4.60 to +6.68) | +2.64 (+1.64 to +3.64) | +2.52 (+1.76 to +3.28) | +2.88 (+1.93 to +3.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | +5.12 (+4.16 to +6.08) | +2.68 (+1.80 to +3.64) | +2.20 (+1.52 to +2.92) | +2.58 (+1.78 to +3.37) |
| correct every question (no gate) | T1_k0_everything_multi_esc | +5.76 (+4.68 to +6.84) | +2.56 (+1.56 to +3.56) | +2.48 (+1.72 to +3.20) | +3.32 (+2.33 to +4.26) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | +3.96 (+3.12 to +4.80) | +2.08 (+1.20 to +3.00) | +1.44 (+0.84 to +2.04) | +1.88 (+1.24 to +2.58) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | +5.04 (+4.08 to +6.00) | +2.52 (+1.56 to +3.48) | +2.16 (+1.44 to +2.84) | +2.58 (+1.78 to +3.37) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | +5.44 (+4.44 to +6.48) | +2.52 (+1.52 to +3.52) | +2.48 (+1.72 to +3.20) | +2.93 (+1.98 to +3.87) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | +5.36 (+4.40 to +6.36) | +2.60 (+1.64 to +3.56) | +2.40 (+1.68 to +3.16) | +2.73 (+1.88 to +3.57) |
| correct every question (no gate) | T1_k0_cot8b | +5.04 (+3.84 to +6.24) | +0.20 (-0.84 to +1.28) | +1.12 (+0.32 to +1.92) | +2.78 (+1.69 to +3.92) |
| best detector at a 20% budget | T1_k0_cot8b | +3.64 (+2.72 to +4.56) | +0.72 (-0.12 to +1.64) | +0.56 (-0.08 to +1.16) | +1.39 (+0.74 to +2.03) |
| best detector at a 30% budget | T1_k0_cot8b | +4.76 (+3.68 to +5.80) | +0.80 (+0.16 to +1.52) | +0.64 (+0.20 to +1.08) | +2.58 (+1.54 to +3.67) |
| best detector at a 50% budget | T1_k0_cot8b | +5.40 (+4.24 to +6.56) | +0.56 (-0.28 to +1.36) | +0.72 (+0.16 to +1.24) | +2.78 (+1.69 to +3.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | +4.48 (+3.44 to +5.56) | +0.60 (+0.00 to +1.20) | +0.56 (+0.16 to +0.96) | +2.33 (+1.34 to +3.32) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | +5.60 (+4.28 to +6.92) | +0.24 (-0.88 to +1.36) | +2.12 (+1.36 to +2.92) | +2.83 (+1.78 to +3.87) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | +3.96 (+3.00 to +4.96) | +1.16 (+0.24 to +2.12) | +1.24 (+0.56 to +1.84) | +1.39 (+0.74 to +2.08) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | +5.76 (+4.60 to +6.92) | +1.00 (+0.28 to +1.68) | +0.88 (+0.44 to +1.32) | +2.53 (+1.54 to +3.52) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | +6.28 (+5.04 to +7.48) | +0.88 (+0.04 to +1.72) | +1.20 (+0.68 to +1.72) | +2.83 (+1.78 to +3.87) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | +5.44 (+4.32 to +6.52) | +0.68 (+0.04 to +1.32) | +0.80 (+0.40 to +1.24) | +2.23 (+1.29 to +3.22) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | +5.92 (+4.84 to +7.04) | +2.68 (+1.72 to +3.68) | +2.52 (+1.76 to +3.28) | +3.27 (+2.28 to +4.21) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +4.04 (+3.20 to +4.88) | +2.20 (+1.36 to +3.08) | +1.52 (+0.92 to +2.12) | +1.88 (+1.24 to +2.58) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +5.12 (+4.16 to +6.08) | +2.68 (+1.80 to +3.64) | +2.20 (+1.52 to +2.92) | +2.58 (+1.78 to +3.37) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +5.64 (+4.60 to +6.68) | +2.64 (+1.64 to +3.64) | +2.52 (+1.76 to +3.28) | +2.88 (+1.93 to +3.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | +5.12 (+4.16 to +6.08) | +2.68 (+1.80 to +3.64) | +2.20 (+1.52 to +2.92) | +2.58 (+1.78 to +3.37) |
| best under 2 passes per question (any configuration) | T1_k0_cot | +3.20 (+2.32 to +4.12) | +0.24 (-0.28 to +0.76) | +1.16 (+0.68 to +1.68) | +0.94 (+0.20 to +1.74) |
| best under 3 passes per question (any configuration) | T1_k0_ground_cot_multi_cot8b | +3.64 (+2.76 to +4.60) | +1.08 (+0.28 to +1.92) | +1.08 (+0.56 to +1.60) | +0.94 (+0.40 to +1.54) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot_multi_esc8b_cot8b | +4.64 (+3.72 to +5.52) | +2.48 (+1.64 to +3.40) | +2.04 (+1.36 to +2.72) | +2.48 (+1.74 to +3.22) |
