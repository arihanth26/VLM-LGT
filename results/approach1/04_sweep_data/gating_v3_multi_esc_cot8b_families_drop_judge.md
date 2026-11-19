# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_esc_cot8b_families_drop_judge_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus 1 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only. Ranker inputs zeroed: judge.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.47% vs 85.47%, T1_k0_ground_cot 85.71% vs 85.71%, T1_k0_cot 85.38% vs 85.38%, T1_k0_zoom_cot 85.60% vs 85.60%, T1_k0_cot_multi 86.12% vs 86.12%, T1_k0_ground_cot_multi 86.27% vs 86.27%, T1_k0_esc8b 88.34% vs 88.34%, T1_k0_ground_cot_multi_esc8b 88.41% vs 88.41%, T1_k0_ground_cot_multi_esc 88.21% vs 88.21%, T1_k0_everything_multi_esc 87.98% vs 87.98%, T1_k0_cot8b 88.23% vs 88.23%, T1_k0_ground_cot_multi_cot8b 88.65% vs 88.65%, T1_k0_ground_cot_multi_esc8b_cot8b 89.56% vs 89.56%, T1_k0_families 88.65% vs 88.65%, T1_k0_esc8b_families 89.78% vs 89.78%, T1_k0_ground_cot_multi_families 89.09% vs 89.09%, T1_k0_ground_cot_multi_esc8b_cot8b_families 89.91% vs 89.91%

Seconds: estimated wall-clock seconds per question on one H100, from the measured component times (results/latency.json).

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Seconds | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 20.07 | 16.00 | 4.42 | 85.47% | +2.69 (+2.20 to +3.19) | +420 / -164 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 5.16 | 3.92 | 0.98 | 85.05% | +2.27 (+1.85 to +2.69) | +320 / -104 | +1.70 (+1.32 to +2.10) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 7.14 | 5.43 | 1.42 | 85.38% | +2.61 (+2.13 to +3.07) | +381 / -133 | +1.84 (+1.44 to +2.25) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 10.88 | 8.46 | 2.28 | 85.52% | +2.74 (+2.26 to +3.22) | +417 / -156 | +1.50 (+1.15 to +1.86) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 25% / 24.4% | 6.13 | 4.66 | 1.20 | 85.26% | +2.48 (+2.04 to +2.92) | +354 / -118 | +1.79 (+1.40 to +2.19) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 6.76 | 5.00 | 2.82 | 85.71% | +2.93 (+2.42 to +3.44) | +436 / -157 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 19.4% | 2.29 | 1.78 | 0.65 | 85.18% | +2.41 (+1.98 to +2.83) | +324 / -95 | +1.78 (+1.38 to +2.16) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 0.93 | 85.52% | +2.74 (+2.27 to +3.19) | +384 / -123 | +1.89 (+1.49 to +2.30) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 4.05 | 2.99 | 1.47 | 85.75% | +2.97 (+2.46 to +3.47) | +432 / -149 | +1.63 (+1.27 to +1.99) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 0.93 | 85.52% | +2.74 (+2.27 to +3.19) | +384 / -123 | +1.89 (+1.49 to +2.30) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 3.26 | 2.00 | 1.80 | 85.38% | +2.61 (+2.06 to +3.14) | +442 / -194 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.51 | 1.19 | 0.45 | 84.91% | +2.13 (+1.71 to +2.54) | +301 / -98 | +1.53 (+1.15 to +1.90) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 0.62 | 85.21% | +2.43 (+1.94 to +2.88) | +364 / -133 | +1.66 (+1.26 to +2.06) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.21 | 1.50 | 0.96 | 85.44% | +2.66 (+2.12 to +3.18) | +430 / -177 | +1.46 (+1.08 to +1.83) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 0.62 | 85.21% | +2.43 (+1.94 to +2.88) | +364 / -133 | +1.66 (+1.26 to +2.06) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 9.29 | 6.00 | 2.52 | 85.60% | +2.83 (+2.32 to +3.31) | +415 / -146 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 19.4% | 2.75 | 1.97 | 0.59 | 85.11% | +2.33 (+1.91 to +2.75) | +315 / -93 | +1.69 (+1.30 to +2.07) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 0.84 | 85.53% | +2.75 (+2.27 to +3.19) | +374 / -112 | +1.93 (+1.53 to +2.33) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 5.26 | 3.49 | 1.32 | 85.66% | +2.88 (+2.37 to +3.35) | +412 / -138 | +1.60 (+1.23 to +1.94) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 0.84 | 85.53% | +2.75 (+2.27 to +3.19) | +374 / -112 | +1.93 (+1.53 to +2.33) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 6.55 | 5.00 | 3.75 | 86.12% | +3.34 (+2.84 to +3.86) | +466 / -148 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 19.4% | 2.25 | 1.78 | 0.83 | 85.34% | +2.56 (+2.15 to +2.98) | +331 / -87 | +1.83 (+1.45 to +2.20) |
| best detector at a 30% budget | T1_k0_cot_multi | token | 30% / 29.6% | 2.84 | 2.18 | 1.20 | 85.73% | +2.95 (+2.48 to +3.40) | +393 / -112 | +1.99 (+1.59 to +2.39) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 49.7% | 3.96 | 2.99 | 1.94 | 86.13% | +3.35 (+2.86 to +3.85) | +455 / -136 | +1.73 (+1.37 to +2.08) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 30% / 29.6% | 2.84 | 2.18 | 1.20 | 85.73% | +2.95 (+2.48 to +3.40) | +393 / -112 | +1.99 (+1.59 to +2.39) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 10.01 | 8.00 | 4.77 | 86.27% | +3.49 (+2.98 to +3.99) | +472 / -140 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 19.4% | 3.01 | 2.36 | 1.04 | 85.43% | +2.65 (+2.24 to +3.08) | +338 / -86 | +1.87 (+1.48 to +2.24) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 1.51 | 85.86% | +3.08 (+2.62 to +3.54) | +403 / -110 | +2.09 (+1.69 to +2.48) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 49.7% | 5.76 | 4.48 | 2.45 | 86.29% | +3.51 (+3.01 to +4.00) | +464 / -130 | +1.81 (+1.45 to +2.15) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 1.51 | 85.86% | +3.08 (+2.62 to +3.54) | +403 / -110 | +2.09 (+1.69 to +2.48) |
| correct every question (no gate) | T1_k0_esc8b | token | 100% / 100.0% | 6.23 | 5.00 | 0.35 | 88.34% | +5.56 (+4.88 to +6.18) | +751 / -222 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_esc8b | judge | 20% / 18.3% | 2.81 | 1.73 | 0.21 | 85.42% | +2.64 (+2.18 to +3.08) | +358 / -107 | +1.62 (+1.17 to +2.05) |
| best detector at a 30% budget | T1_k0_esc8b | after_head | 30% / 28.2% | 3.28 | 2.13 | 0.23 | 87.43% | +4.65 (+4.10 to +5.23) | +615 / -172 | +3.01 (+2.51 to +3.51) |
| best detector at a 50% budget | T1_k0_esc8b | token | 50% / 49.7% | 3.70 | 2.99 | 0.24 | 88.21% | +5.43 (+4.78 to +6.02) | +716 / -199 | +2.72 (+2.25 to +3.17) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | token | 40% / 39.8% | 3.19 | 2.59 | 0.22 | 88.04% | +5.26 (+4.64 to +5.85) | +679 / -178 | +2.96 (+2.48 to +3.46) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | token | 100% / 100.0% | 14.13 | 12.00 | 4.93 | 88.41% | +5.63 (+5.00 to +6.20) | +724 / -188 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | token | 20% / 19.4% | 3.86 | 3.14 | 1.07 | 86.52% | +3.74 (+3.26 to +4.21) | +465 / -109 | +2.60 (+2.14 to +3.06) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | after_head | 30% / 28.2% | 5.74 | 4.10 | 1.54 | 87.48% | +4.70 (+4.13 to +5.24) | +606 / -159 | +2.99 (+2.51 to +3.48) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | token | 50% / 49.7% | 7.85 | 6.47 | 2.53 | 88.26% | +5.48 (+4.86 to +6.06) | +697 / -175 | +2.72 (+2.27 to +3.16) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | token | 40% / 39.8% | 6.58 | 5.38 | 2.06 | 88.05% | +5.27 (+4.69 to +5.81) | +662 / -160 | +2.90 (+2.43 to +3.38) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | token | 100% / 100.0% | 24.62 | 22.25 | 7.67 | 88.21% | +5.43 (+4.85 to +5.99) | +612 / -95 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | after_head | 20% / 19.2% | 6.69 | 5.08 | 1.64 | 87.04% | +4.27 (+3.80 to +4.73) | +467 / -61 | +3.10 (+2.65 to +3.53) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | after_head | 30% / 28.2% | 8.77 | 6.99 | 2.31 | 87.64% | +4.86 (+4.35 to +5.38) | +544 / -81 | +3.20 (+2.77 to +3.65) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | token | 50% / 49.7% | 13.13 | 11.56 | 3.89 | 88.16% | +5.38 (+4.83 to +5.93) | +602 / -90 | +2.54 (+2.16 to +2.93) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | after_head | 35% / 33.2% | 9.91 | 8.06 | 2.69 | 87.80% | +5.02 (+4.50 to +5.55) | +561 / -83 | +2.98 (+2.58 to +3.39) |
| correct every question (no gate) | T1_k0_everything_multi_esc | token | 100% / 100.0% | 32.52 | 28.25 | 9.14 | 87.98% | +5.20 (+4.65 to +5.75) | +594 / -99 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | after_head | 20% / 19.2% | 8.25 | 6.23 | 1.92 | 86.91% | +4.13 (+3.67 to +4.59) | +461 / -68 | +3.04 (+2.61 to +3.46) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | after_head | 30% / 28.2% | 11.03 | 8.68 | 2.73 | 87.44% | +4.67 (+4.15 to +5.19) | +531 / -87 | +3.09 (+2.67 to +3.51) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | token | 50% / 49.7% | 17.09 | 14.55 | 4.63 | 87.92% | +5.14 (+4.60 to +5.68) | +584 / -95 | +2.44 (+2.06 to +2.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | after_head | 35% / 33.2% | 12.57 | 10.05 | 3.18 | 87.58% | +4.80 (+4.29 to +5.33) | +546 / -89 | +2.88 (+2.47 to +3.29) |
| correct every question (no gate) | T1_k0_cot8b | token | 100% / 100.0% | 6.37 | 5.00 | 2.18 | 88.23% | +5.45 (+4.77 to +6.12) | +781 / -262 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot8b | judge | 20% / 18.3% | 2.82 | 1.73 | 0.54 | 86.24% | +3.46 (+2.99 to +3.90) | +416 / -87 | +2.30 (+1.86 to +2.75) |
| best detector at a 30% budget | T1_k0_cot8b | after_head | 30% / 28.2% | 3.31 | 2.13 | 0.75 | 87.78% | +5.00 (+4.39 to +5.60) | +665 / -189 | +3.34 (+2.86 to +3.87) |
| best detector at a 50% budget | T1_k0_cot8b | token | 50% / 49.7% | 3.76 | 2.99 | 1.15 | 88.09% | +5.32 (+4.67 to +5.93) | +734 / -228 | +2.56 (+2.09 to +3.04) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | token | 40% / 39.8% | 3.24 | 2.59 | 0.95 | 87.96% | +5.18 (+4.54 to +5.75) | +687 / -194 | +2.93 (+2.45 to +3.45) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | token | 100% / 100.0% | 14.18 | 12.00 | 6.76 | 88.65% | +5.87 (+5.20 to +6.47) | +766 / -207 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | judge | 20% / 18.3% | 4.33 | 3.01 | 1.38 | 86.37% | +3.59 (+3.12 to +4.03) | +420 / -78 | +2.36 (+1.92 to +2.82) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | after_head | 30% / 28.2% | 5.75 | 4.10 | 2.05 | 88.06% | +5.29 (+4.70 to +5.84) | +661 / -158 | +3.48 (+3.01 to +3.97) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | token | 50% / 49.7% | 7.87 | 6.47 | 3.44 | 88.43% | +5.65 (+5.02 to +6.24) | +726 / -188 | +2.65 (+2.20 to +3.09) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | token | 40% / 39.8% | 6.59 | 5.38 | 2.79 | 88.20% | +5.42 (+4.83 to +5.98) | +682 / -166 | +2.97 (+2.52 to +3.47) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | token | 100% / 100.0% | 18.27 | 16.00 | 6.92 | 89.56% | +6.78 (+6.12 to +7.37) | +782 / -137 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | after_head | 20% / 19.2% | 5.43 | 3.88 | 1.49 | 87.87% | +5.10 (+4.56 to +5.60) | +569 / -84 | +3.69 (+3.18 to +4.16) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | after_head | 30% / 28.2% | 6.94 | 5.23 | 2.10 | 88.73% | +5.95 (+5.37 to +6.48) | +677 / -111 | +3.90 (+3.39 to +4.38) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | token | 50% / 49.7% | 9.94 | 8.46 | 3.52 | 89.32% | +6.55 (+5.92 to +7.12) | +750 / -127 | +3.11 (+2.65 to +3.57) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | token | 40% / 39.8% | 8.26 | 6.98 | 2.85 | 89.05% | +6.27 (+5.69 to +6.84) | +709 / -112 | +3.44 (+2.95 to +3.93) |
| correct every question (no gate) | T1_k0_families | token | 100% / 100.0% | 11.02 | 9.00 | 0.71 | 88.65% | +5.87 (+5.23 to +6.48) | +766 / -207 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_families | after_head | 20% / 19.2% | 3.84 | 2.54 | 0.29 | 86.90% | +4.12 (+3.64 to +4.63) | +508 / -116 | +3.08 (+2.62 to +3.55) |
| best detector at a 30% budget | T1_k0_families | after_head | 30% / 28.2% | 4.66 | 3.25 | 0.34 | 87.60% | +4.82 (+4.28 to +5.37) | +611 / -152 | +3.22 (+2.73 to +3.70) |
| best detector at a 50% budget | T1_k0_families | after_head | 50% / 47.7% | 6.42 | 4.81 | 0.44 | 88.35% | +5.57 (+4.94 to +6.17) | +719 / -189 | +2.70 (+2.25 to +3.14) |
| smallest budget keeping 90% of the full validation gain | T1_k0_families | after_head | 45% / 43.2% | 6.02 | 4.45 | 0.42 | 88.28% | +5.51 (+4.90 to +6.10) | +706 / -182 | +2.93 (+2.46 to +3.39) |
| correct every question (no gate) | T1_k0_esc8b_families | token | 100% / 100.0% | 15.16 | 13.00 | 0.88 | 89.78% | +7.00 (+6.34 to +7.63) | +846 / -180 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_esc8b_families | after_head | 20% / 19.2% | 4.68 | 3.30 | 0.32 | 87.56% | +4.78 (+4.27 to +5.31) | +557 / -102 | +3.47 (+2.98 to +3.95) |
| best detector at a 30% budget | T1_k0_esc8b_families | after_head | 30% / 28.2% | 5.88 | 4.38 | 0.39 | 88.39% | +5.61 (+5.04 to +6.17) | +667 / -133 | +3.59 (+3.11 to +4.08) |
| best detector at a 50% budget | T1_k0_esc8b_families | token | 50% / 49.7% | 8.25 | 6.97 | 0.51 | 89.48% | +6.70 (+6.07 to +7.33) | +797 / -159 | +3.18 (+2.72 to +3.66) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b_families | token | 45% / 45.1% | 7.60 | 6.41 | 0.47 | 89.31% | +6.54 (+5.92 to +7.15) | +777 / -155 | +3.35 (+2.88 to +3.84) |
| correct every question (no gate) | T1_k0_ground_cot_multi_families | token | 100% / 100.0% | 18.91 | 16.00 | 5.30 | 89.09% | +6.32 (+5.64 to +6.96) | +801 / -200 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_families | after_head | 20% / 19.2% | 5.56 | 3.88 | 1.18 | 87.33% | +4.55 (+4.06 to +5.06) | +544 / -111 | +3.40 (+2.93 to +3.92) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_families | after_head | 30% / 28.2% | 7.12 | 5.23 | 1.64 | 88.03% | +5.25 (+4.70 to +5.80) | +650 / -150 | +3.52 (+3.05 to +4.02) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_families | after_head | 50% / 47.7% | 10.41 | 8.15 | 2.64 | 88.81% | +6.03 (+5.38 to +6.65) | +757 / -183 | +2.97 (+2.52 to +3.44) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_families | after_head | 45% / 43.2% | 9.67 | 7.48 | 2.41 | 88.76% | +5.98 (+5.36 to +6.59) | +746 / -177 | +3.25 (+2.78 to +3.72) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b_families | token | 100% / 100.0% | 27.14 | 24.00 | 7.45 | 89.91% | +7.13 (+6.48 to +7.78) | +863 / -184 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b_families | after_head | 20% / 19.2% | 7.20 | 5.42 | 1.60 | 87.64% | +4.86 (+4.34 to +5.37) | +572 / -109 | +3.48 (+3.01 to +3.97) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b_families | after_head | 30% / 28.2% | 9.50 | 7.48 | 2.25 | 88.56% | +5.78 (+5.20 to +6.33) | +688 / -138 | +3.72 (+3.22 to +4.21) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b_families | token | 50% / 49.7% | 14.44 | 12.43 | 3.79 | 89.69% | +6.91 (+6.26 to +7.53) | +818 / -160 | +3.25 (+2.76 to +3.73) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b_families | after_head | 45% / 43.2% | 13.28 | 10.93 | 3.34 | 89.48% | +6.70 (+6.06 to +7.30) | +796 / -158 | +3.44 (+2.96 to +3.90) |
| best under 2 passes per question (any configuration) | T1_k0_cot8b | token | 15% / 14.4% | 1.83 | 1.58 | 0.42 | 86.03% | +3.25 (+2.81 to +3.71) | +392 / -83 | +2.43 (+2.00 to +2.88) |
| best under 3 passes per question (any configuration) | T1_k0_cot8b | token | 35% / 34.9% | 2.97 | 2.39 | 0.84 | 87.73% | +4.95 (+4.32 to +5.51) | +653 / -182 | +2.96 (+2.44 to +3.46) |
| best under 5 passes per question (any configuration) | T1_k0_cot8b | token | 65% / 64.6% | 4.54 | 3.58 | 1.46 | 88.27% | +5.50 (+4.83 to +6.16) | +774 / -251 | +1.75 (+1.36 to +2.16) |
| best under 0.3 s per question on one H100 (any configuration) | T1_k0_esc8b | token | 60% / 59.5% | 4.20 | 3.38 | 0.26 | 88.32% | +5.54 (+4.86 to +6.14) | +739 / -212 | +2.14 (+1.71 to +2.57) |
| best under 0.5 s per question on one H100 (any configuration) | T1_k0_esc8b_families | token | 45% / 45.1% | 7.60 | 6.41 | 0.47 | 89.31% | +6.54 (+5.92 to +7.15) | +777 / -155 | +3.35 (+2.88 to +3.84) |
| best under 1 s per question on one H100 (any configuration) | T1_k0_esc8b_families | after_head | 90% / 89.7% | 13.82 | 11.76 | 0.81 | 89.79% | +7.01 (+6.35 to +7.64) | +846 / -179 | +0.83 (+0.60 to +1.04) |
| best under 2 s per question on one H100 (any configuration) | T1_k0_esc8b_families | after_head | 90% / 89.7% | 13.82 | 11.76 | 0.81 | 89.79% | +7.01 (+6.35 to +7.64) | +846 / -179 | +0.83 (+0.60 to +1.04) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +5.36 (+4.20 to +6.52) | +2.32 (+1.40 to +3.32) | +0.96 (+0.36 to +1.56) | +1.98 (+0.79 to +3.27) |
| best detector at a 20% budget | T1_k5_everything | +4.52 (+3.48 to +5.60) | +2.16 (+1.40 to +3.00) | +0.52 (+0.08 to +0.96) | +1.78 (+0.79 to +2.78) |
| best detector at a 30% budget | T1_k5_everything | +5.12 (+4.00 to +6.24) | +2.36 (+1.48 to +3.28) | +0.72 (+0.16 to +1.24) | +2.13 (+1.04 to +3.32) |
| best detector at a 50% budget | T1_k5_everything | +5.28 (+4.12 to +6.44) | +2.32 (+1.40 to +3.32) | +0.96 (+0.36 to +1.56) | +2.33 (+1.19 to +3.57) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +4.84 (+3.76 to +5.92) | +2.32 (+1.48 to +3.20) | +0.76 (+0.24 to +1.24) | +1.88 (+0.84 to +2.97) |
| correct every question (no gate) | T1_k0_ground_cot | +5.96 (+4.72 to +7.16) | +2.16 (+1.28 to +3.04) | +1.00 (+0.40 to +1.60) | +2.53 (+1.29 to +3.82) |
| best detector at a 20% budget | T1_k0_ground_cot | +5.00 (+3.92 to +6.08) | +2.00 (+1.28 to +2.80) | +0.52 (+0.08 to +0.92) | +2.03 (+1.09 to +3.02) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.56 (+4.40 to +6.68) | +2.12 (+1.32 to +2.96) | +0.80 (+0.28 to +1.32) | +2.43 (+1.34 to +3.52) |
| best detector at a 50% budget | T1_k0_ground_cot | +5.88 (+4.64 to +7.08) | +2.16 (+1.28 to +3.04) | +1.00 (+0.40 to +1.60) | +2.83 (+1.64 to +4.07) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.56 (+4.40 to +6.68) | +2.12 (+1.32 to +2.96) | +0.80 (+0.28 to +1.32) | +2.43 (+1.34 to +3.52) |
| correct every question (no gate) | T1_k0_cot | +6.60 (+5.32 to +7.88) | +0.68 (-0.16 to +1.56) | +0.64 (-0.04 to +1.28) | +2.48 (+1.19 to +3.82) |
| best detector at a 20% budget | T1_k0_cot | +5.16 (+4.12 to +6.24) | +1.00 (+0.28 to +1.76) | +0.44 (-0.00 to +0.92) | +1.88 (+0.99 to +2.83) |
| best detector at a 30% budget | T1_k0_cot | +5.96 (+4.80 to +7.16) | +0.84 (+0.08 to +1.60) | +0.52 (-0.08 to +1.08) | +2.38 (+1.39 to +3.52) |
| best detector at a 50% budget | T1_k0_cot | +6.32 (+5.08 to +7.64) | +0.80 (-0.04 to +1.68) | +0.64 (-0.04 to +1.28) | +2.93 (+1.74 to +4.21) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.96 (+4.80 to +7.16) | +0.84 (+0.08 to +1.60) | +0.52 (-0.08 to +1.08) | +2.38 (+1.39 to +3.52) |
| correct every question (no gate) | T1_k0_zoom_cot | +5.76 (+4.60 to +6.92) | +2.20 (+1.36 to +3.08) | +0.96 (+0.40 to +1.52) | +2.28 (+1.09 to +3.52) |
| best detector at a 20% budget | T1_k0_zoom_cot | +4.84 (+3.80 to +5.92) | +2.00 (+1.28 to +2.76) | +0.48 (+0.04 to +0.92) | +1.93 (+0.99 to +2.88) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.44 (+4.32 to +6.56) | +2.24 (+1.44 to +3.08) | +0.84 (+0.36 to +1.36) | +2.43 (+1.44 to +3.52) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.68 (+4.48 to +6.88) | +2.20 (+1.36 to +3.08) | +0.96 (+0.40 to +1.52) | +2.63 (+1.49 to +3.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.44 (+4.32 to +6.56) | +2.24 (+1.44 to +3.08) | +0.84 (+0.36 to +1.36) | +2.43 (+1.44 to +3.52) |
| correct every question (no gate) | T1_k0_cot_multi | +6.96 (+5.72 to +8.24) | +1.80 (+1.04 to +2.64) | +1.12 (+0.52 to +1.72) | +3.52 (+2.18 to +4.96) |
| best detector at a 20% budget | T1_k0_cot_multi | +5.56 (+4.48 to +6.64) | +1.68 (+1.00 to +2.40) | +0.76 (+0.28 to +1.24) | +2.18 (+1.19 to +3.22) |
| best detector at a 30% budget | T1_k0_cot_multi | +6.44 (+5.28 to +7.60) | +1.76 (+1.04 to +2.52) | +0.92 (+0.32 to +1.52) | +2.63 (+1.49 to +3.87) |
| best detector at a 50% budget | T1_k0_cot_multi | +6.80 (+5.56 to +8.04) | +1.84 (+1.08 to +2.64) | +1.12 (+0.52 to +1.72) | +3.72 (+2.48 to +5.11) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +6.44 (+5.28 to +7.60) | +1.76 (+1.04 to +2.52) | +0.92 (+0.32 to +1.52) | +2.63 (+1.49 to +3.87) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +6.84 (+5.60 to +8.08) | +2.36 (+1.52 to +3.24) | +1.04 (+0.44 to +1.64) | +3.77 (+2.43 to +5.16) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +5.36 (+4.28 to +6.44) | +2.16 (+1.40 to +2.96) | +0.68 (+0.24 to +1.12) | +2.33 (+1.29 to +3.37) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +6.16 (+5.00 to +7.32) | +2.32 (+1.52 to +3.16) | +0.88 (+0.32 to +1.44) | +2.93 (+1.78 to +4.16) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +6.72 (+5.48 to +7.96) | +2.40 (+1.56 to +3.28) | +1.04 (+0.44 to +1.64) | +3.97 (+2.73 to +5.35) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +6.16 (+5.00 to +7.32) | +2.32 (+1.52 to +3.16) | +0.88 (+0.32 to +1.44) | +2.93 (+1.78 to +4.16) |
| correct every question (no gate) | T1_k0_esc8b | +6.84 (+5.52 to +8.20) | +3.88 (+2.68 to +5.04) | +4.32 (+3.32 to +5.28) | +7.59 (+6.10 to +9.07) |
| best detector at a 20% budget | T1_k0_esc8b | +4.04 (+2.96 to +5.12) | +0.44 (-0.04 to +0.92) | +0.80 (+0.40 to +1.24) | +5.90 (+4.56 to +7.24) |
| best detector at a 30% budget | T1_k0_esc8b | +6.40 (+5.04 to +7.68) | +3.20 (+2.12 to +4.24) | +2.80 (+2.00 to +3.56) | +6.59 (+5.30 to +7.98) |
| best detector at a 50% budget | T1_k0_esc8b | +6.60 (+5.24 to +7.92) | +3.92 (+2.72 to +5.04) | +4.16 (+3.20 to +5.12) | +7.44 (+6.05 to +8.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | +6.48 (+5.20 to +7.76) | +3.84 (+2.68 to +5.00) | +3.92 (+3.00 to +4.84) | +7.19 (+5.90 to +8.58) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | +7.40 (+6.04 to +8.72) | +4.16 (+3.04 to +5.32) | +3.80 (+2.88 to +4.68) | +7.54 (+6.10 to +8.97) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | +5.32 (+4.24 to +6.48) | +3.00 (+2.04 to +4.00) | +2.20 (+1.56 to +2.88) | +4.61 (+3.57 to +5.75) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | +6.76 (+5.48 to +8.04) | +3.36 (+2.36 to +4.40) | +2.44 (+1.72 to +3.12) | +6.59 (+5.30 to +7.98) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | +6.96 (+5.64 to +8.28) | +4.16 (+3.04 to +5.32) | +3.80 (+2.92 to +4.68) | +7.39 (+6.00 to +8.78) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | +6.76 (+5.48 to +8.08) | +4.04 (+2.96 to +5.16) | +3.56 (+2.72 to +4.40) | +7.09 (+5.80 to +8.43) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | +8.40 (+7.12 to +9.68) | +4.44 (+3.44 to +5.48) | +3.04 (+2.24 to +3.80) | +5.95 (+4.71 to +7.29) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | +7.36 (+6.16 to +8.48) | +3.32 (+2.52 to +4.16) | +1.52 (+1.04 to +2.04) | +5.01 (+3.87 to +6.20) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | +8.08 (+6.84 to +9.32) | +3.72 (+2.80 to +4.64) | +2.08 (+1.44 to +2.72) | +5.75 (+4.56 to +7.09) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | +8.24 (+7.00 to +9.48) | +4.44 (+3.44 to +5.48) | +3.04 (+2.28 to +3.80) | +5.90 (+4.66 to +7.19) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | +8.24 (+6.96 to +9.48) | +3.88 (+2.92 to +4.80) | +2.24 (+1.60 to +2.88) | +5.90 (+4.66 to +7.24) |
| correct every question (no gate) | T1_k0_everything_multi_esc | +7.96 (+6.76 to +9.20) | +4.36 (+3.36 to +5.36) | +2.92 (+2.12 to +3.64) | +5.65 (+4.36 to +7.04) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | +7.08 (+5.92 to +8.20) | +3.20 (+2.40 to +4.04) | +1.52 (+1.04 to +2.04) | +4.86 (+3.72 to +6.05) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | +7.68 (+6.48 to +8.92) | +3.60 (+2.72 to +4.52) | +2.04 (+1.40 to +2.68) | +5.50 (+4.26 to +6.84) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | +7.84 (+6.64 to +9.04) | +4.36 (+3.36 to +5.36) | +2.92 (+2.16 to +3.64) | +5.50 (+4.26 to +6.79) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | +7.80 (+6.60 to +9.00) | +3.76 (+2.80 to +4.72) | +2.16 (+1.52 to +2.80) | +5.65 (+4.36 to +6.99) |
| correct every question (no gate) | T1_k0_cot8b | +10.32 (+8.84 to +11.76) | +1.04 (-0.12 to +2.20) | +2.92 (+1.92 to +3.92) | +8.03 (+6.59 to +9.62) |
| best detector at a 20% budget | T1_k0_cot8b | +7.12 (+5.96 to +8.24) | +0.40 (-0.08 to +0.92) | +0.68 (+0.24 to +1.12) | +6.15 (+4.86 to +7.54) |
| best detector at a 30% budget | T1_k0_cot8b | +10.12 (+8.72 to +11.52) | +1.16 (+0.16 to +2.16) | +2.00 (+1.16 to +2.80) | +7.14 (+5.80 to +8.58) |
| best detector at a 50% budget | T1_k0_cot8b | +9.88 (+8.48 to +11.28) | +1.48 (+0.36 to +2.60) | +2.84 (+1.84 to +3.84) | +7.49 (+6.15 to +9.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | +9.64 (+8.28 to +11.00) | +1.80 (+0.76 to +2.84) | +2.80 (+1.88 to +3.72) | +6.79 (+5.55 to +8.23) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | +10.84 (+9.44 to +12.24) | +2.16 (+1.08 to +3.32) | +2.96 (+2.00 to +3.84) | +7.93 (+6.49 to +9.47) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | +7.48 (+6.32 to +8.60) | +0.68 (+0.20 to +1.20) | +0.64 (+0.24 to +1.04) | +6.05 (+4.76 to +7.44) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | +10.36 (+8.96 to +11.76) | +2.16 (+1.24 to +3.16) | +1.96 (+1.20 to +2.72) | +6.99 (+5.60 to +8.48) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | +10.24 (+8.92 to +11.60) | +2.24 (+1.20 to +3.36) | +3.04 (+2.16 to +3.92) | +7.44 (+6.05 to +8.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | +9.88 (+8.52 to +11.20) | +2.52 (+1.52 to +3.60) | +2.84 (+1.96 to +3.68) | +6.69 (+5.40 to +8.08) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | +10.84 (+9.48 to +12.24) | +4.20 (+3.12 to +5.32) | +4.00 (+3.08 to +4.84) | +8.38 (+6.99 to +9.82) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +9.08 (+7.80 to +10.32) | +3.08 (+2.24 to +3.96) | +2.16 (+1.60 to +2.76) | +6.30 (+5.06 to +7.59) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +10.28 (+8.92 to +11.64) | +3.60 (+2.68 to +4.56) | +2.76 (+2.00 to +3.48) | +7.44 (+6.10 to +8.82) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +10.24 (+8.92 to +11.56) | +4.20 (+3.12 to +5.32) | +4.00 (+3.08 to +4.88) | +8.03 (+6.59 to +9.47) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | +9.88 (+8.56 to +11.20) | +4.16 (+3.12 to +5.24) | +3.72 (+2.84 to +4.60) | +7.59 (+6.35 to +8.92) |
| correct every question (no gate) | T1_k0_families | +6.88 (+5.60 to +8.16) | +4.00 (+2.76 to +5.20) | +2.64 (+1.60 to +3.64) | +10.96 (+9.52 to +12.44) |
| best detector at a 20% budget | T1_k0_families | +5.36 (+4.20 to +6.52) | +2.44 (+1.52 to +3.32) | +1.44 (+0.80 to +2.12) | +7.98 (+6.79 to +9.27) |
| best detector at a 30% budget | T1_k0_families | +6.20 (+5.00 to +7.40) | +2.96 (+1.96 to +4.00) | +1.52 (+0.72 to +2.32) | +9.52 (+8.23 to +10.96) |
| best detector at a 50% budget | T1_k0_families | +6.80 (+5.52 to +8.08) | +3.72 (+2.52 to +4.84) | +2.16 (+1.20 to +3.12) | +10.56 (+9.17 to +12.05) |
| smallest budget keeping 90% of the full validation gain | T1_k0_families | +6.76 (+5.48 to +8.04) | +3.64 (+2.48 to +4.72) | +2.16 (+1.24 to +3.08) | +10.41 (+9.07 to +11.90) |
| correct every question (no gate) | T1_k0_esc8b_families | +8.32 (+6.92 to +9.68) | +5.20 (+3.92 to +6.36) | +4.40 (+3.32 to +5.48) | +10.81 (+9.42 to +12.30) |
| best detector at a 20% budget | T1_k0_esc8b_families | +6.56 (+5.32 to +7.76) | +3.24 (+2.32 to +4.20) | +2.08 (+1.40 to +2.80) | +7.83 (+6.64 to +9.12) |
| best detector at a 30% budget | T1_k0_esc8b_families | +7.40 (+6.08 to +8.72) | +3.88 (+2.84 to +4.96) | +2.68 (+1.88 to +3.52) | +9.17 (+7.88 to +10.56) |
| best detector at a 50% budget | T1_k0_esc8b_families | +7.84 (+6.48 to +9.16) | +5.24 (+4.00 to +6.40) | +4.44 (+3.44 to +5.44) | +9.92 (+8.58 to +11.35) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b_families | +7.84 (+6.52 to +9.16) | +5.16 (+3.92 to +6.32) | +4.40 (+3.40 to +5.44) | +9.27 (+7.98 to +10.71) |
| correct every question (no gate) | T1_k0_ground_cot_multi_families | +8.28 (+6.84 to +9.64) | +4.52 (+3.28 to +5.72) | +2.60 (+1.60 to +3.60) | +10.71 (+9.32 to +12.20) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_families | +6.60 (+5.32 to +7.80) | +2.88 (+1.96 to +3.84) | +1.40 (+0.80 to +2.04) | +7.98 (+6.79 to +9.27) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_families | +7.44 (+6.12 to +8.72) | +3.44 (+2.44 to +4.52) | +1.44 (+0.68 to +2.24) | +9.52 (+8.23 to +10.96) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_families | +8.20 (+6.76 to +9.56) | +4.24 (+3.04 to +5.40) | +2.16 (+1.24 to +3.08) | +10.36 (+9.02 to +11.85) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_families | +8.12 (+6.72 to +9.44) | +4.16 (+3.04 to +5.32) | +2.16 (+1.28 to +3.04) | +10.31 (+8.97 to +11.80) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b_families | +8.88 (+7.44 to +10.28) | +5.04 (+3.76 to +6.28) | +4.36 (+3.36 to +5.40) | +11.01 (+9.57 to +12.50) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b_families | +7.12 (+5.84 to +8.36) | +3.00 (+2.08 to +4.00) | +2.08 (+1.40 to +2.76) | +7.83 (+6.64 to +9.17) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b_families | +8.04 (+6.64 to +9.40) | +3.72 (+2.68 to +4.76) | +2.68 (+1.88 to +3.48) | +9.37 (+8.08 to +10.81) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b_families | +8.64 (+7.20 to +9.96) | +5.04 (+3.84 to +6.24) | +4.40 (+3.40 to +5.40) | +10.21 (+8.82 to +11.70) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b_families | +8.76 (+7.32 to +10.16) | +4.76 (+3.60 to +5.88) | +3.72 (+2.80 to +4.68) | +10.26 (+8.87 to +11.75) |
| best under 2 passes per question (any configuration) | T1_k0_cot8b | +6.60 (+5.52 to +7.72) | +1.64 (+0.84 to +2.44) | +1.32 (+0.80 to +1.84) | +3.47 (+2.53 to +4.51) |
| best under 3 passes per question (any configuration) | T1_k0_cot8b | +9.48 (+8.16 to +10.84) | +1.84 (+0.84 to +2.84) | +2.52 (+1.60 to +3.44) | +6.20 (+5.01 to +7.54) |
| best under 5 passes per question (any configuration) | T1_k0_cot8b | +10.24 (+8.80 to +11.68) | +1.16 (+0.00 to +2.32) | +2.96 (+1.96 to +3.96) | +8.13 (+6.69 to +9.72) |
| best under 0.3 s per question on one H100 (any configuration) | T1_k0_esc8b | +6.72 (+5.40 to +8.08) | +3.92 (+2.72 to +5.08) | +4.36 (+3.36 to +5.32) | +7.54 (+6.10 to +9.02) |
| best under 0.5 s per question on one H100 (any configuration) | T1_k0_esc8b_families | +7.84 (+6.52 to +9.16) | +5.16 (+3.92 to +6.32) | +4.40 (+3.40 to +5.44) | +9.27 (+7.98 to +10.71) |
| best under 1 s per question on one H100 (any configuration) | T1_k0_esc8b_families | +8.32 (+6.92 to +9.68) | +5.20 (+3.92 to +6.36) | +4.44 (+3.36 to +5.48) | +10.81 (+9.42 to +12.30) |
| best under 2 s per question on one H100 (any configuration) | T1_k0_esc8b_families | +8.32 (+6.92 to +9.68) | +5.20 (+3.92 to +6.36) | +4.44 (+3.36 to +5.48) | +10.81 (+9.42 to +12.30) |
