# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_esc_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus two scoring passes per distinct candidate of a corrected question); calls: generation calls only.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.42% vs 85.42%, T1_k0_ground_cot 85.62% vs 85.62%, T1_k0_cot 85.37% vs 85.37%, T1_k0_zoom_cot 85.67% vs 85.67%, T1_k0_cot_multi 86.04% vs 86.04%, T1_k0_ground_cot_multi 86.11% vs 86.11%, T1_k0_esc8b 88.14% vs 88.14%, T1_k0_ground_cot_multi_esc8b 88.14% vs 88.14%, T1_k0_ground_cot_multi_esc 86.69% vs 86.69%, T1_k0_everything_multi_esc 86.56% vs 86.56%

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 24.13 | 16.00 | 85.42% | +2.64 (+2.12 to +3.14) | +421 / -170 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 6.39 | 3.92 | 84.93% | +2.15 (+1.73 to +2.57) | +314 / -109 | +1.61 (+1.22 to +2.00) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 85.24% | +2.46 (+1.99 to +2.93) | +376 / -142 | +1.64 (+1.25 to +2.05) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 13.31 | 8.46 | 85.41% | +2.63 (+2.11 to +3.12) | +417 / -167 | +1.44 (+1.08 to +1.80) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 85.24% | +2.46 (+1.99 to +2.93) | +376 / -142 | +1.64 (+1.25 to +2.05) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 8.53 | 5.00 | 85.62% | +2.84 (+2.33 to +3.35) | +428 / -158 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 19.4% | 2.80 | 1.78 | 85.10% | +2.32 (+1.90 to +2.74) | +313 / -92 | +1.72 (+1.33 to +2.11) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 85.46% | +2.68 (+2.21 to +3.14) | +381 / -126 | +1.86 (+1.47 to +2.27) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 5.11 | 2.99 | 85.62% | +2.84 (+2.33 to +3.34) | +424 / -154 | +1.46 (+1.11 to +1.83) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 85.46% | +2.68 (+2.21 to +3.14) | +381 / -126 | +1.86 (+1.47 to +2.27) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 4.52 | 2.00 | 85.37% | +2.60 (+2.03 to +3.12) | +438 / -191 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 84.95% | +2.18 (+1.75 to +2.56) | +302 / -95 | +1.62 (+1.23 to +1.99) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 85.19% | +2.42 (+1.92 to +2.87) | +370 / -140 | +1.67 (+1.27 to +2.08) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.93 | 1.50 | 85.44% | +2.66 (+2.11 to +3.18) | +434 / -181 | +1.41 (+1.03 to +1.79) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 85.19% | +2.42 (+1.92 to +2.87) | +370 / -140 | +1.67 (+1.27 to +2.08) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 12.57 | 6.00 | 85.67% | +2.89 (+2.34 to +3.39) | +448 / -173 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 19.4% | 3.53 | 1.97 | 85.13% | +2.35 (+1.93 to +2.78) | +325 / -101 | +1.71 (+1.32 to +2.08) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 85.52% | +2.74 (+2.24 to +3.22) | +398 / -137 | +1.89 (+1.48 to +2.30) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 7.04 | 3.49 | 85.67% | +2.89 (+2.36 to +3.39) | +443 / -168 | +1.58 (+1.20 to +1.94) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 85.52% | +2.74 (+2.24 to +3.22) | +398 / -137 | +1.89 (+1.48 to +2.30) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 8.10 | 5.00 | 86.04% | +3.26 (+2.75 to +3.76) | +459 / -149 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 19.4% | 2.72 | 1.78 | 85.28% | +2.50 (+2.08 to +2.92) | +328 / -90 | +1.75 (+1.37 to +2.12) |
| best detector at a 30% budget | T1_k0_cot_multi | token | 30% / 29.6% | 3.50 | 2.18 | 85.72% | +2.94 (+2.47 to +3.40) | +397 / -117 | +1.93 (+1.56 to +2.33) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 49.7% | 4.93 | 2.99 | 86.08% | +3.30 (+2.78 to +3.80) | +455 / -141 | +1.67 (+1.31 to +2.04) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 30% / 29.6% | 3.50 | 2.18 | 85.72% | +2.94 (+2.47 to +3.40) | +397 / -117 | +1.93 (+1.56 to +2.33) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 12.02 | 8.00 | 86.11% | +3.33 (+2.82 to +3.84) | +462 / -145 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 19.4% | 3.66 | 2.36 | 85.33% | +2.55 (+2.13 to +2.98) | +333 / -90 | +1.84 (+1.45 to +2.22) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 29.6% | 4.85 | 3.07 | 85.80% | +3.03 (+2.55 to +3.49) | +406 / -118 | +2.03 (+1.64 to +2.43) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 49.7% | 7.04 | 4.48 | 86.14% | +3.36 (+2.85 to +3.86) | +458 / -138 | +1.67 (+1.31 to +2.03) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 29.6% | 4.85 | 3.07 | 85.80% | +3.03 (+2.55 to +3.49) | +406 / -118 | +2.03 (+1.64 to +2.43) |
| correct every question (no gate) | T1_k0_esc8b | token | 100% / 100.0% | 7.46 | 5.00 | 88.14% | +5.36 (+4.71 to +5.96) | +716 / -206 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_esc8b | token | 20% / 19.4% | 2.43 | 1.78 | 86.33% | +3.55 (+3.07 to +4.03) | +453 / -115 | +2.55 (+2.10 to +3.04) |
| best detector at a 30% budget | T1_k0_esc8b | after_head | 30% / 28.2% | 3.71 | 2.13 | 87.34% | +4.56 (+3.99 to +5.11) | +595 / -161 | +3.05 (+2.57 to +3.54) |
| best detector at a 50% budget | T1_k0_esc8b | token | 50% / 49.7% | 4.41 | 2.99 | 88.07% | +5.30 (+4.67 to +5.89) | +693 / -189 | +2.73 (+2.28 to +3.18) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | token | 40% / 39.8% | 3.79 | 2.59 | 87.91% | +5.13 (+4.52 to +5.71) | +659 / -171 | +2.99 (+2.51 to +3.48) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | token | 100% / 100.0% | 16.25 | 12.00 | 88.14% | +5.36 (+4.76 to +5.91) | +661 / -151 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | token | 20% / 19.4% | 4.57 | 3.14 | 86.47% | +3.69 (+3.22 to +4.16) | +446 / -95 | +2.51 (+2.09 to +2.94) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | after_head | 30% / 28.2% | 6.66 | 4.10 | 87.44% | +4.67 (+4.13 to +5.20) | +569 / -125 | +2.94 (+2.50 to +3.40) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | token | 50% / 49.7% | 9.24 | 6.47 | 88.05% | +5.27 (+4.70 to +5.80) | +645 / -143 | +2.62 (+2.21 to +3.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | token | 40% / 39.8% | 7.79 | 5.38 | 87.88% | +5.11 (+4.55 to +5.63) | +621 / -135 | +2.81 (+2.37 to +3.25) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | token | 100% / 100.0% | 26.69 | 22.25 | 86.69% | +3.91 (+3.41 to +4.39) | +462 / -90 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | token | 20% / 19.4% | 6.71 | 5.13 | 85.72% | +2.94 (+2.53 to +3.36) | +344 / -64 | +1.98 (+1.61 to +2.34) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | token | 30% / 29.6% | 9.40 | 7.28 | 86.32% | +3.54 (+3.08 to +3.99) | +418 / -81 | +2.19 (+1.80 to +2.57) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | token | 50% / 49.7% | 14.52 | 11.56 | 86.66% | +3.88 (+3.38 to +4.36) | +457 / -88 | +1.75 (+1.42 to +2.10) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | token | 35% / 34.9% | 10.78 | 8.41 | 86.49% | +3.71 (+3.24 to +4.17) | +436 / -83 | +2.07 (+1.69 to +2.45) |
| correct every question (no gate) | T1_k0_everything_multi_esc | token | 100% / 100.0% | 36.51 | 28.25 | 86.56% | +3.78 (+3.30 to +4.28) | +451 / -91 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | token | 20% / 19.4% | 8.73 | 6.30 | 85.64% | +2.86 (+2.45 to +3.28) | +337 / -65 | +1.98 (+1.63 to +2.35) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | token | 30% / 29.6% | 12.43 | 9.05 | 86.22% | +3.45 (+2.98 to +3.90) | +410 / -82 | +2.19 (+1.82 to +2.56) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | token | 50% / 49.7% | 19.46 | 14.55 | 86.53% | +3.75 (+3.26 to +4.23) | +446 / -89 | +1.74 (+1.41 to +2.08) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | token | 35% / 34.9% | 14.32 | 10.50 | 86.38% | +3.60 (+3.13 to +4.08) | +427 / -84 | +2.09 (+1.72 to +2.47) |
| best under 2 passes per question (any configuration) | T1_k0_esc8b | token | 10% / 9.4% | 1.71 | 1.38 | 85.02% | +2.24 (+1.87 to +2.62) | +266 / -53 | +1.72 (+1.34 to +2.11) |
| best under 3 passes per question (any configuration) | T1_k0_esc8b | pre_head | 25% / 23.2% | 2.64 | 1.93 | 86.75% | +3.97 (+3.44 to +4.50) | +508 / -130 | +2.68 (+2.20 to +3.16) |
| best under 5 passes per question (any configuration) | T1_k0_esc8b | token | 55% / 54.7% | 4.72 | 3.19 | 88.16% | +5.38 (+4.74 to +5.99) | +706 / -194 | +2.54 (+2.08 to +2.98) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +4.92 (+3.76 to +6.12) | +2.32 (+1.40 to +3.28) | +1.08 (+0.36 to +1.76) | +2.13 (+0.99 to +3.37) |
| best detector at a 20% budget | T1_k5_everything | +4.32 (+3.32 to +5.40) | +2.08 (+1.32 to +2.92) | +0.60 (+0.08 to +1.12) | +1.49 (+0.59 to +2.43) |
| best detector at a 30% budget | T1_k5_everything | +4.72 (+3.64 to +5.84) | +2.36 (+1.52 to +3.24) | +0.76 (+0.12 to +1.36) | +1.88 (+0.84 to +3.02) |
| best detector at a 50% budget | T1_k5_everything | +4.84 (+3.64 to +6.04) | +2.32 (+1.40 to +3.28) | +1.12 (+0.40 to +1.80) | +2.13 (+0.99 to +3.32) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +4.72 (+3.64 to +5.84) | +2.36 (+1.52 to +3.24) | +0.76 (+0.12 to +1.36) | +1.88 (+0.84 to +3.02) |
| correct every question (no gate) | T1_k0_ground_cot | +5.68 (+4.48 to +6.84) | +2.00 (+1.16 to +2.84) | +0.88 (+0.12 to +1.60) | +2.78 (+1.64 to +3.97) |
| best detector at a 20% budget | T1_k0_ground_cot | +4.64 (+3.60 to +5.68) | +2.00 (+1.28 to +2.76) | +0.60 (+0.08 to +1.12) | +1.98 (+1.09 to +2.93) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.40 (+4.28 to +6.48) | +2.12 (+1.32 to +2.92) | +0.76 (+0.08 to +1.40) | +2.38 (+1.39 to +3.47) |
| best detector at a 50% budget | T1_k0_ground_cot | +5.60 (+4.40 to +6.80) | +2.08 (+1.24 to +2.96) | +0.92 (+0.20 to +1.64) | +2.73 (+1.64 to +3.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.40 (+4.28 to +6.48) | +2.12 (+1.32 to +2.92) | +0.76 (+0.08 to +1.40) | +2.38 (+1.39 to +3.47) |
| correct every question (no gate) | T1_k0_cot | +6.48 (+5.24 to +7.76) | +0.64 (-0.24 to +1.56) | +0.36 (-0.40 to +1.04) | +2.97 (+1.83 to +4.21) |
| best detector at a 20% budget | T1_k0_cot | +5.16 (+4.12 to +6.20) | +1.24 (+0.52 to +1.96) | +0.32 (-0.20 to +0.80) | +1.93 (+1.09 to +2.88) |
| best detector at a 30% budget | T1_k0_cot | +5.92 (+4.80 to +7.04) | +0.96 (+0.16 to +1.80) | +0.36 (-0.32 to +1.00) | +2.43 (+1.44 to +3.57) |
| best detector at a 50% budget | T1_k0_cot | +6.40 (+5.16 to +7.64) | +0.76 (-0.12 to +1.68) | +0.44 (-0.32 to +1.12) | +3.12 (+2.03 to +4.36) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.92 (+4.80 to +7.04) | +0.96 (+0.16 to +1.80) | +0.36 (-0.32 to +1.00) | +2.43 (+1.44 to +3.57) |
| correct every question (no gate) | T1_k0_zoom_cot | +5.60 (+4.40 to +6.84) | +2.20 (+1.28 to +3.16) | +1.00 (+0.24 to +1.68) | +2.73 (+1.59 to +3.97) |
| best detector at a 20% budget | T1_k0_zoom_cot | +4.60 (+3.56 to +5.68) | +2.12 (+1.32 to +2.92) | +0.68 (+0.16 to +1.20) | +1.93 (+1.04 to +2.93) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.32 (+4.20 to +6.48) | +2.36 (+1.48 to +3.24) | +0.84 (+0.20 to +1.48) | +2.38 (+1.34 to +3.52) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.48 (+4.24 to +6.72) | +2.28 (+1.36 to +3.24) | +1.04 (+0.28 to +1.76) | +2.73 (+1.59 to +3.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.32 (+4.20 to +6.48) | +2.36 (+1.48 to +3.24) | +0.84 (+0.20 to +1.48) | +2.38 (+1.34 to +3.52) |
| correct every question (no gate) | T1_k0_cot_multi | +6.96 (+5.76 to +8.24) | +1.52 (+0.68 to +2.40) | +1.16 (+0.48 to +1.80) | +3.42 (+2.23 to +4.71) |
| best detector at a 20% budget | T1_k0_cot_multi | +5.32 (+4.24 to +6.40) | +1.68 (+1.00 to +2.44) | +0.76 (+0.24 to +1.28) | +2.18 (+1.24 to +3.22) |
| best detector at a 30% budget | T1_k0_cot_multi | +6.28 (+5.12 to +7.44) | +1.68 (+0.88 to +2.48) | +1.04 (+0.40 to +1.64) | +2.73 (+1.64 to +3.92) |
| best detector at a 50% budget | T1_k0_cot_multi | +6.88 (+5.68 to +8.16) | +1.64 (+0.80 to +2.48) | +1.16 (+0.48 to +1.80) | +3.57 (+2.38 to +4.86) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +6.28 (+5.12 to +7.44) | +1.68 (+0.88 to +2.48) | +1.04 (+0.40 to +1.64) | +2.73 (+1.64 to +3.92) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +6.68 (+5.48 to +7.92) | +2.40 (+1.52 to +3.28) | +1.00 (+0.32 to +1.68) | +3.22 (+2.08 to +4.46) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +4.96 (+3.92 to +6.00) | +2.28 (+1.52 to +3.08) | +0.72 (+0.20 to +1.24) | +2.18 (+1.19 to +3.22) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +5.96 (+4.84 to +7.08) | +2.44 (+1.60 to +3.28) | +0.96 (+0.28 to +1.60) | +2.68 (+1.59 to +3.82) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +6.56 (+5.36 to +7.80) | +2.48 (+1.60 to +3.36) | +1.04 (+0.32 to +1.72) | +3.37 (+2.23 to +4.61) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +5.96 (+4.84 to +7.08) | +2.44 (+1.60 to +3.28) | +0.96 (+0.28 to +1.60) | +2.68 (+1.59 to +3.82) |
| correct every question (no gate) | T1_k0_esc8b | +6.68 (+5.40 to +8.00) | +3.56 (+2.40 to +4.68) | +4.28 (+3.28 to +5.24) | +7.29 (+5.90 to +8.68) |
| best detector at a 20% budget | T1_k0_esc8b | +4.96 (+3.88 to +6.08) | +2.52 (+1.60 to +3.48) | +2.32 (+1.60 to +3.04) | +4.61 (+3.57 to +5.75) |
| best detector at a 30% budget | T1_k0_esc8b | +6.24 (+4.96 to +7.52) | +3.04 (+2.00 to +4.04) | +2.84 (+2.04 to +3.60) | +6.49 (+5.21 to +7.88) |
| best detector at a 50% budget | T1_k0_esc8b | +6.52 (+5.20 to +7.84) | +3.60 (+2.48 to +4.68) | +4.20 (+3.24 to +5.16) | +7.24 (+5.90 to +8.63) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | +6.48 (+5.20 to +7.76) | +3.48 (+2.32 to +4.60) | +3.96 (+3.04 to +4.88) | +6.94 (+5.65 to +8.28) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | +7.64 (+6.36 to +8.92) | +3.80 (+2.72 to +4.88) | +3.60 (+2.76 to +4.44) | +6.64 (+5.30 to +7.98) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | +5.68 (+4.56 to +6.80) | +2.92 (+2.00 to +3.88) | +2.00 (+1.36 to +2.64) | +4.26 (+3.22 to +5.35) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | +7.16 (+5.92 to +8.44) | +3.20 (+2.28 to +4.20) | +2.24 (+1.52 to +2.96) | +6.40 (+5.16 to +7.73) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | +7.28 (+6.04 to +8.56) | +3.84 (+2.76 to +4.92) | +3.60 (+2.76 to +4.44) | +6.64 (+5.30 to +7.98) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | +7.16 (+5.92 to +8.40) | +3.76 (+2.68 to +4.84) | +3.36 (+2.52 to +4.16) | +6.40 (+5.21 to +7.64) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | +5.64 (+4.64 to +6.72) | +3.80 (+2.84 to +4.80) | +2.64 (+1.88 to +3.40) | +3.47 (+2.43 to +4.56) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | +4.48 (+3.52 to +5.44) | +3.20 (+2.40 to +4.08) | +1.40 (+0.84 to +1.96) | +2.63 (+1.69 to +3.57) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | +5.24 (+4.24 to +6.24) | +3.64 (+2.72 to +4.60) | +2.04 (+1.36 to +2.72) | +3.17 (+2.18 to +4.21) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | +5.56 (+4.52 to +6.64) | +3.80 (+2.84 to +4.80) | +2.64 (+1.88 to +3.40) | +3.42 (+2.43 to +4.51) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | +5.48 (+4.44 to +6.52) | +3.72 (+2.80 to +4.68) | +2.28 (+1.56 to +3.00) | +3.27 (+2.28 to +4.31) |
| correct every question (no gate) | T1_k0_everything_multi_esc | +5.44 (+4.44 to +6.52) | +3.52 (+2.52 to +4.52) | +2.72 (+2.00 to +3.44) | +3.37 (+2.38 to +4.46) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | +4.40 (+3.48 to +5.36) | +3.00 (+2.20 to +3.84) | +1.44 (+0.88 to +2.00) | +2.53 (+1.59 to +3.52) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | +5.04 (+4.08 to +6.04) | +3.40 (+2.48 to +4.32) | +2.20 (+1.52 to +2.84) | +3.07 (+2.13 to +4.12) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | +5.36 (+4.36 to +6.44) | +3.52 (+2.52 to +4.52) | +2.72 (+2.00 to +3.44) | +3.32 (+2.33 to +4.41) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | +5.28 (+4.32 to +6.32) | +3.44 (+2.44 to +4.40) | +2.44 (+1.72 to +3.12) | +3.17 (+2.18 to +4.26) |
| best under 2 passes per question (any configuration) | T1_k0_esc8b | +3.72 (+2.88 to +4.60) | +1.56 (+0.92 to +2.20) | +1.16 (+0.68 to +1.68) | +2.58 (+1.69 to +3.47) |
| best under 3 passes per question (any configuration) | T1_k0_esc8b | +5.88 (+4.64 to +7.12) | +2.24 (+1.44 to +3.08) | +2.16 (+1.52 to +2.80) | +6.00 (+4.76 to +7.29) |
| best under 5 passes per question (any configuration) | T1_k0_esc8b | +6.68 (+5.40 to +8.00) | +3.60 (+2.48 to +4.68) | +4.28 (+3.32 to +5.24) | +7.34 (+5.95 to +8.73) |
