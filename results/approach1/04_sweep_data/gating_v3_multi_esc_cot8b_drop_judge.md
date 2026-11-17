# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_esc_cot8b_drop_judge_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus 1 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only. Ranker inputs zeroed: judge.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.50% vs 85.50%, T1_k0_ground_cot 85.69% vs 85.69%, T1_k0_cot 85.39% vs 85.39%, T1_k0_zoom_cot 85.52% vs 85.52%, T1_k0_cot_multi 86.07% vs 86.07%, T1_k0_ground_cot_multi 86.26% vs 86.26%, T1_k0_esc8b 88.35% vs 88.35%, T1_k0_ground_cot_multi_esc8b 88.49% vs 88.49%, T1_k0_ground_cot_multi_esc 88.21% vs 88.21%, T1_k0_everything_multi_esc 88.17% vs 88.17%, T1_k0_cot8b 88.26% vs 88.26%, T1_k0_ground_cot_multi_cot8b 88.65% vs 88.65%, T1_k0_ground_cot_multi_esc8b_cot8b 89.53% vs 89.53%

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 20.07 | 16.00 | 85.50% | +2.72 (+2.20 to +3.26) | +451 / -192 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 5.16 | 3.92 | 85.07% | +2.29 (+1.83 to +2.73) | +338 / -120 | +1.69 (+1.28 to +2.08) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 7.14 | 5.43 | 85.43% | +2.65 (+2.16 to +3.13) | +404 / -152 | +1.84 (+1.42 to +2.26) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 10.88 | 8.46 | 85.56% | +2.78 (+2.27 to +3.30) | +447 / -182 | +1.51 (+1.13 to +1.87) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 25% / 24.4% | 6.13 | 4.66 | 85.32% | +2.54 (+2.07 to +3.01) | +375 / -133 | +1.79 (+1.38 to +2.20) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 6.76 | 5.00 | 85.69% | +2.91 (+2.41 to +3.43) | +440 / -163 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 19.4% | 2.29 | 1.78 | 85.23% | +2.45 (+2.02 to +2.88) | +327 / -94 | +1.83 (+1.42 to +2.22) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 85.57% | +2.79 (+2.32 to +3.26) | +388 / -122 | +1.98 (+1.59 to +2.40) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 4.05 | 2.99 | 85.75% | +2.97 (+2.49 to +3.48) | +436 / -153 | +1.62 (+1.25 to +1.99) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 85.57% | +2.79 (+2.32 to +3.26) | +388 / -122 | +1.98 (+1.59 to +2.40) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 3.26 | 2.00 | 85.39% | +2.62 (+2.08 to +3.15) | +433 / -184 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.51 | 1.19 | 84.91% | +2.13 (+1.72 to +2.53) | +299 / -96 | +1.53 (+1.16 to +1.90) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 85.18% | +2.41 (+1.93 to +2.86) | +361 / -132 | +1.62 (+1.22 to +2.02) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.21 | 1.50 | 85.43% | +2.65 (+2.12 to +3.16) | +424 / -172 | +1.46 (+1.08 to +1.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 85.18% | +2.41 (+1.93 to +2.86) | +361 / -132 | +1.62 (+1.22 to +2.02) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 9.29 | 6.00 | 85.52% | +2.74 (+2.24 to +3.25) | +426 / -165 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 19.4% | 2.75 | 1.97 | 85.06% | +2.28 (+1.86 to +2.70) | +316 / -99 | +1.68 (+1.29 to +2.06) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 85.41% | +2.63 (+2.15 to +3.07) | +379 / -129 | +1.85 (+1.44 to +2.26) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 5.26 | 3.49 | 85.57% | +2.79 (+2.30 to +3.29) | +422 / -156 | +1.56 (+1.18 to +1.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 85.41% | +2.63 (+2.15 to +3.07) | +379 / -129 | +1.85 (+1.44 to +2.26) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 6.55 | 5.00 | 86.07% | +3.29 (+2.77 to +3.79) | +461 / -148 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 19.4% | 2.25 | 1.78 | 85.31% | +2.53 (+2.12 to +2.95) | +328 / -87 | +1.80 (+1.42 to +2.18) |
| best detector at a 30% budget | T1_k0_cot_multi | after_head | 30% / 28.2% | 3.47 | 2.13 | 85.86% | +3.08 (+2.57 to +3.57) | +431 / -138 | +2.08 (+1.68 to +2.49) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 49.7% | 3.96 | 2.99 | 86.08% | +3.30 (+2.79 to +3.78) | +451 / -137 | +1.69 (+1.32 to +2.05) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 35% / 34.9% | 3.14 | 2.39 | 85.95% | +3.17 (+2.71 to +3.65) | +417 / -115 | +1.99 (+1.59 to +2.37) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 10.01 | 8.00 | 86.26% | +3.48 (+2.96 to +3.99) | +482 / -151 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 19.4% | 3.01 | 2.36 | 85.44% | +2.66 (+2.24 to +3.10) | +345 / -92 | +1.89 (+1.50 to +2.27) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 85.87% | +3.09 (+2.61 to +3.56) | +412 / -118 | +2.09 (+1.69 to +2.50) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 49.7% | 5.76 | 4.48 | 86.28% | +3.50 (+2.98 to +4.00) | +474 / -141 | +1.80 (+1.42 to +2.16) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 85.87% | +3.09 (+2.61 to +3.56) | +412 / -118 | +2.09 (+1.69 to +2.50) |
| correct every question (no gate) | T1_k0_esc8b | token | 100% / 100.0% | 6.23 | 5.00 | 88.35% | +5.57 (+4.90 to +6.19) | +749 / -219 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_esc8b | token | 20% / 19.4% | 2.10 | 1.78 | 86.46% | +3.68 (+3.19 to +4.15) | +467 / -117 | +2.66 (+2.21 to +3.14) |
| best detector at a 30% budget | T1_k0_esc8b | after_head | 30% / 28.2% | 3.28 | 2.13 | 87.45% | +4.68 (+4.11 to +5.24) | +615 / -170 | +3.02 (+2.53 to +3.52) |
| best detector at a 50% budget | T1_k0_esc8b | token | 50% / 49.7% | 3.70 | 2.99 | 88.24% | +5.46 (+4.80 to +6.06) | +717 / -197 | +2.75 (+2.28 to +3.20) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | token | 40% / 39.8% | 3.19 | 2.59 | 88.07% | +5.30 (+4.68 to +5.89) | +680 / -176 | +2.99 (+2.51 to +3.49) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | token | 100% / 100.0% | 14.13 | 12.00 | 88.49% | +5.72 (+5.05 to +6.30) | +739 / -195 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | token | 20% / 19.4% | 3.86 | 3.14 | 86.55% | +3.77 (+3.28 to +4.25) | +470 / -111 | +2.64 (+2.19 to +3.11) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | after_head | 30% / 28.2% | 5.74 | 4.10 | 87.54% | +4.76 (+4.18 to +5.31) | +616 / -163 | +3.07 (+2.58 to +3.56) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | token | 50% / 49.7% | 7.85 | 6.47 | 88.37% | +5.59 (+4.95 to +6.17) | +712 / -180 | +2.77 (+2.31 to +3.23) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | token | 40% / 39.8% | 6.58 | 5.38 | 88.18% | +5.40 (+4.80 to +5.96) | +676 / -162 | +3.01 (+2.51 to +3.50) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | token | 100% / 100.0% | 24.62 | 22.25 | 88.21% | +5.43 (+4.88 to +5.99) | +615 / -98 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | after_head | 20% / 19.2% | 6.69 | 5.08 | 87.08% | +4.30 (+3.81 to +4.75) | +471 / -62 | +3.15 (+2.71 to +3.57) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | after_head | 30% / 28.2% | 8.77 | 6.99 | 87.63% | +4.85 (+4.33 to +5.37) | +546 / -84 | +3.22 (+2.79 to +3.66) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | token | 50% / 49.7% | 13.13 | 11.56 | 88.14% | +5.36 (+4.81 to +5.91) | +604 / -94 | +2.55 (+2.17 to +2.94) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | token | 35% / 34.9% | 9.67 | 8.41 | 87.79% | +5.01 (+4.48 to +5.54) | +561 / -84 | +2.99 (+2.57 to +3.43) |
| correct every question (no gate) | T1_k0_everything_multi_esc | token | 100% / 100.0% | 32.52 | 28.25 | 88.17% | +5.39 (+4.82 to +5.96) | +616 / -103 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | after_head | 20% / 19.2% | 8.25 | 6.23 | 87.05% | +4.28 (+3.80 to +4.74) | +474 / -67 | +3.14 (+2.72 to +3.57) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | after_head | 30% / 28.2% | 11.03 | 8.68 | 87.60% | +4.82 (+4.30 to +5.34) | +548 / -89 | +3.19 (+2.76 to +3.64) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | token | 50% / 49.7% | 17.09 | 14.55 | 88.07% | +5.30 (+4.76 to +5.84) | +603 / -99 | +2.50 (+2.11 to +2.90) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | token | 35% / 34.9% | 12.49 | 10.50 | 87.77% | +4.99 (+4.45 to +5.51) | +561 / -86 | +2.99 (+2.58 to +3.43) |
| correct every question (no gate) | T1_k0_cot8b | token | 100% / 100.0% | 6.37 | 5.00 | 88.26% | +5.48 (+4.80 to +6.14) | +779 / -257 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot8b | judge | 20% / 18.3% | 2.82 | 1.73 | 86.26% | +3.48 (+3.01 to +3.93) | +415 / -84 | +2.35 (+1.90 to +2.82) |
| best detector at a 30% budget | T1_k0_cot8b | after_head | 30% / 28.2% | 3.31 | 2.13 | 87.86% | +5.09 (+4.49 to +5.67) | +663 / -179 | +3.43 (+2.93 to +3.94) |
| best detector at a 50% budget | T1_k0_cot8b | token | 50% / 49.7% | 3.76 | 2.99 | 88.15% | +5.37 (+4.71 to +5.99) | +734 / -223 | +2.61 (+2.13 to +3.08) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | token | 40% / 39.8% | 3.24 | 2.59 | 88.03% | +5.25 (+4.61 to +5.83) | +689 / -189 | +2.99 (+2.51 to +3.51) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | token | 100% / 100.0% | 14.18 | 12.00 | 88.65% | +5.87 (+5.20 to +6.47) | +782 / -223 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | judge | 20% / 18.3% | 4.33 | 3.01 | 86.38% | +3.60 (+3.13 to +4.05) | +422 / -79 | +2.42 (+1.98 to +2.88) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | after_head | 30% / 28.2% | 5.75 | 4.10 | 88.08% | +5.31 (+4.73 to +5.86) | +669 / -164 | +3.54 (+3.07 to +4.04) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | token | 50% / 49.7% | 7.87 | 6.47 | 88.45% | +5.67 (+5.02 to +6.25) | +741 / -201 | +2.69 (+2.22 to +3.15) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | token | 40% / 39.8% | 6.59 | 5.38 | 88.23% | +5.45 (+4.85 to +6.02) | +696 / -177 | +3.03 (+2.54 to +3.52) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | token | 100% / 100.0% | 18.27 | 16.00 | 89.53% | +6.76 (+6.10 to +7.37) | +791 / -148 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | after_head | 20% / 19.2% | 5.43 | 3.88 | 87.87% | +5.10 (+4.55 to +5.60) | +575 / -90 | +3.69 (+3.19 to +4.17) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | after_head | 30% / 28.2% | 6.94 | 5.23 | 88.66% | +5.88 (+5.31 to +6.43) | +682 / -122 | +3.82 (+3.33 to +4.32) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | token | 50% / 49.7% | 9.94 | 8.46 | 89.31% | +6.54 (+5.89 to +7.15) | +760 / -138 | +3.09 (+2.63 to +3.54) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | token | 40% / 39.8% | 8.26 | 6.98 | 89.06% | +6.28 (+5.70 to +6.85) | +720 / -122 | +3.41 (+2.92 to +3.91) |
| best under 2 passes per question (any configuration) | T1_k0_cot8b | pre_head | 15% / 13.8% | 1.79 | 1.55 | 86.32% | +3.54 (+3.07 to +3.98) | +425 / -88 | +2.74 (+2.27 to +3.20) |
| best under 3 passes per question (any configuration) | T1_k0_cot8b | token | 35% / 34.9% | 2.97 | 2.39 | 87.80% | +5.02 (+4.39 to +5.58) | +654 / -176 | +3.04 (+2.51 to +3.53) |
| best under 5 passes per question (any configuration) | T1_k0_cot8b | token | 60% / 59.5% | 4.28 | 3.38 | 88.27% | +5.50 (+4.82 to +6.14) | +766 / -243 | +2.12 (+1.70 to +2.55) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +5.32 (+4.08 to +6.56) | +2.36 (+1.40 to +3.36) | +1.00 (+0.36 to +1.64) | +2.08 (+0.79 to +3.42) |
| best detector at a 20% budget | T1_k5_everything | +4.48 (+3.36 to +5.60) | +2.24 (+1.44 to +3.12) | +0.64 (+0.16 to +1.12) | +1.69 (+0.69 to +2.73) |
| best detector at a 30% budget | T1_k5_everything | +5.08 (+3.96 to +6.24) | +2.44 (+1.48 to +3.40) | +0.84 (+0.28 to +1.40) | +2.13 (+1.04 to +3.37) |
| best detector at a 50% budget | T1_k5_everything | +5.24 (+4.04 to +6.48) | +2.40 (+1.40 to +3.44) | +1.04 (+0.40 to +1.68) | +2.38 (+1.09 to +3.62) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +4.84 (+3.72 to +6.00) | +2.44 (+1.56 to +3.36) | +0.92 (+0.40 to +1.48) | +1.83 (+0.74 to +2.97) |
| correct every question (no gate) | T1_k0_ground_cot | +5.76 (+4.56 to +7.00) | +2.08 (+1.20 to +3.00) | +1.20 (+0.56 to +1.84) | +2.53 (+1.29 to +3.82) |
| best detector at a 20% budget | T1_k0_ground_cot | +4.84 (+3.80 to +5.92) | +2.04 (+1.28 to +2.84) | +0.72 (+0.24 to +1.16) | +2.13 (+1.19 to +3.12) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.40 (+4.28 to +6.52) | +2.16 (+1.36 to +3.04) | +1.00 (+0.44 to +1.56) | +2.58 (+1.49 to +3.72) |
| best detector at a 50% budget | T1_k0_ground_cot | +5.68 (+4.48 to +6.92) | +2.12 (+1.24 to +3.04) | +1.24 (+0.60 to +1.88) | +2.83 (+1.64 to +4.07) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.40 (+4.28 to +6.52) | +2.16 (+1.36 to +3.04) | +1.00 (+0.44 to +1.56) | +2.58 (+1.49 to +3.72) |
| correct every question (no gate) | T1_k0_cot | +6.60 (+5.40 to +7.84) | +0.60 (-0.20 to +1.48) | +0.60 (-0.08 to +1.24) | +2.68 (+1.39 to +3.92) |
| best detector at a 20% budget | T1_k0_cot | +5.20 (+4.16 to +6.28) | +0.96 (+0.24 to +1.72) | +0.40 (-0.04 to +0.88) | +1.93 (+1.04 to +2.83) |
| best detector at a 30% budget | T1_k0_cot | +5.96 (+4.84 to +7.12) | +0.80 (+0.04 to +1.60) | +0.48 (-0.12 to +1.08) | +2.38 (+1.39 to +3.52) |
| best detector at a 50% budget | T1_k0_cot | +6.36 (+5.12 to +7.64) | +0.72 (-0.08 to +1.60) | +0.60 (-0.08 to +1.24) | +2.97 (+1.78 to +4.16) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.96 (+4.84 to +7.12) | +0.80 (+0.04 to +1.60) | +0.48 (-0.12 to +1.08) | +2.38 (+1.39 to +3.52) |
| correct every question (no gate) | T1_k0_zoom_cot | +5.48 (+4.28 to +6.68) | +2.12 (+1.24 to +3.04) | +1.00 (+0.36 to +1.68) | +2.28 (+1.09 to +3.52) |
| best detector at a 20% budget | T1_k0_zoom_cot | +4.68 (+3.64 to +5.76) | +1.92 (+1.16 to +2.68) | +0.52 (+0.04 to +1.00) | +1.93 (+0.99 to +2.93) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.24 (+4.12 to +6.36) | +2.16 (+1.28 to +3.00) | +0.72 (+0.16 to +1.32) | +2.33 (+1.24 to +3.47) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.40 (+4.20 to +6.56) | +2.16 (+1.28 to +3.08) | +1.00 (+0.36 to +1.68) | +2.58 (+1.39 to +3.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.24 (+4.12 to +6.36) | +2.16 (+1.28 to +3.00) | +0.72 (+0.16 to +1.32) | +2.33 (+1.24 to +3.47) |
| correct every question (no gate) | T1_k0_cot_multi | +6.76 (+5.52 to +8.04) | +1.76 (+1.00 to +2.60) | +1.08 (+0.44 to +1.72) | +3.62 (+2.33 to +5.01) |
| best detector at a 20% budget | T1_k0_cot_multi | +5.36 (+4.32 to +6.48) | +1.72 (+1.04 to +2.40) | +0.72 (+0.24 to +1.20) | +2.28 (+1.29 to +3.37) |
| best detector at a 30% budget | T1_k0_cot_multi | +6.72 (+5.52 to +7.96) | +1.36 (+0.64 to +2.12) | +0.76 (+0.24 to +1.32) | +3.57 (+2.28 to +4.91) |
| best detector at a 50% budget | T1_k0_cot_multi | +6.60 (+5.40 to +7.88) | +1.84 (+1.08 to +2.68) | +1.08 (+0.44 to +1.72) | +3.77 (+2.53 to +5.11) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +6.56 (+5.40 to +7.80) | +1.88 (+1.16 to +2.64) | +1.00 (+0.36 to +1.60) | +3.27 (+2.08 to +4.51) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +6.68 (+5.44 to +7.92) | +2.44 (+1.56 to +3.36) | +1.16 (+0.52 to +1.80) | +3.67 (+2.38 to +5.11) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +5.24 (+4.16 to +6.32) | +2.32 (+1.52 to +3.12) | +0.72 (+0.24 to +1.20) | +2.28 (+1.29 to +3.37) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +6.04 (+4.88 to +7.24) | +2.48 (+1.64 to +3.32) | +1.00 (+0.40 to +1.60) | +2.78 (+1.59 to +4.02) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +6.56 (+5.32 to +7.80) | +2.48 (+1.60 to +3.40) | +1.20 (+0.56 to +1.84) | +3.82 (+2.58 to +5.16) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +6.04 (+4.88 to +7.24) | +2.48 (+1.64 to +3.32) | +1.00 (+0.40 to +1.60) | +2.78 (+1.59 to +4.02) |
| correct every question (no gate) | T1_k0_esc8b | +6.88 (+5.52 to +8.24) | +3.92 (+2.72 to +5.08) | +4.40 (+3.40 to +5.40) | +7.44 (+6.00 to +8.87) |
| best detector at a 20% budget | T1_k0_esc8b | +5.04 (+3.96 to +6.16) | +2.80 (+1.88 to +3.80) | +2.44 (+1.72 to +3.20) | +4.61 (+3.57 to +5.75) |
| best detector at a 30% budget | T1_k0_esc8b | +6.40 (+5.04 to +7.68) | +3.24 (+2.20 to +4.28) | +2.84 (+2.04 to +3.60) | +6.59 (+5.30 to +7.98) |
| best detector at a 50% budget | T1_k0_esc8b | +6.64 (+5.28 to +7.96) | +3.96 (+2.76 to +5.04) | +4.28 (+3.28 to +5.28) | +7.34 (+5.95 to +8.73) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | +6.52 (+5.24 to +7.80) | +3.88 (+2.72 to +5.00) | +4.00 (+3.04 to +4.96) | +7.14 (+5.85 to +8.48) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | +7.28 (+5.96 to +8.60) | +4.28 (+3.12 to +5.44) | +4.12 (+3.12 to +5.04) | +7.54 (+6.10 to +8.97) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | +5.20 (+4.12 to +6.32) | +3.12 (+2.16 to +4.12) | +2.32 (+1.64 to +3.04) | +4.61 (+3.57 to +5.75) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | +6.64 (+5.36 to +7.96) | +3.52 (+2.48 to +4.56) | +2.64 (+1.84 to +3.36) | +6.59 (+5.30 to +7.98) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | +6.88 (+5.60 to +8.20) | +4.28 (+3.16 to +5.44) | +4.16 (+3.20 to +5.04) | +7.39 (+6.00 to +8.78) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | +6.72 (+5.48 to +8.00) | +4.20 (+3.08 to +5.32) | +3.92 (+3.00 to +4.80) | +7.09 (+5.80 to +8.43) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | +8.24 (+7.00 to +9.48) | +4.44 (+3.44 to +5.48) | +3.20 (+2.36 to +4.00) | +5.95 (+4.66 to +7.29) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | +7.36 (+6.20 to +8.48) | +3.32 (+2.52 to +4.16) | +1.64 (+1.08 to +2.20) | +5.01 (+3.87 to +6.25) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | +8.00 (+6.80 to +9.20) | +3.72 (+2.80 to +4.64) | +2.12 (+1.44 to +2.80) | +5.75 (+4.51 to +7.09) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | +8.04 (+6.84 to +9.28) | +4.44 (+3.44 to +5.48) | +3.20 (+2.40 to +3.96) | +5.85 (+4.61 to +7.19) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | +7.84 (+6.64 to +9.04) | +4.32 (+3.36 to +5.32) | +2.72 (+2.00 to +3.44) | +5.21 (+4.07 to +6.45) |
| correct every question (no gate) | T1_k0_everything_multi_esc | +8.32 (+7.08 to +9.56) | +4.32 (+3.28 to +5.32) | +3.24 (+2.40 to +4.00) | +5.75 (+4.46 to +7.14) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | +7.40 (+6.28 to +8.56) | +3.20 (+2.40 to +4.04) | +1.68 (+1.16 to +2.20) | +4.96 (+3.82 to +6.15) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | +8.04 (+6.84 to +9.24) | +3.64 (+2.72 to +4.56) | +2.16 (+1.48 to +2.84) | +5.60 (+4.31 to +6.94) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | +8.08 (+6.84 to +9.32) | +4.32 (+3.28 to +5.32) | +3.24 (+2.44 to +4.00) | +5.60 (+4.36 to +6.89) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | +7.88 (+6.72 to +9.08) | +4.24 (+3.24 to +5.24) | +2.80 (+2.04 to +3.52) | +5.06 (+3.92 to +6.25) |
| correct every question (no gate) | T1_k0_cot8b | +10.40 (+8.96 to +11.84) | +1.28 (+0.16 to +2.40) | +2.84 (+1.80 to +3.88) | +7.88 (+6.45 to +9.47) |
| best detector at a 20% budget | T1_k0_cot8b | +7.08 (+5.92 to +8.20) | +0.52 (+0.04 to +1.00) | +0.68 (+0.24 to +1.12) | +6.15 (+4.81 to +7.54) |
| best detector at a 30% budget | T1_k0_cot8b | +10.24 (+8.84 to +11.68) | +1.36 (+0.40 to +2.32) | +2.00 (+1.16 to +2.80) | +7.14 (+5.80 to +8.58) |
| best detector at a 50% budget | T1_k0_cot8b | +9.96 (+8.56 to +11.36) | +1.68 (+0.60 to +2.80) | +2.80 (+1.80 to +3.80) | +7.44 (+6.10 to +8.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | +9.68 (+8.32 to +10.96) | +2.04 (+1.04 to +3.08) | +2.80 (+1.88 to +3.76) | +6.79 (+5.55 to +8.23) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | +10.76 (+9.36 to +12.16) | +2.08 (+0.96 to +3.28) | +3.04 (+2.04 to +4.00) | +8.03 (+6.59 to +9.57) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | +7.44 (+6.28 to +8.56) | +0.68 (+0.20 to +1.20) | +0.68 (+0.28 to +1.08) | +6.10 (+4.76 to +7.49) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | +10.32 (+8.96 to +11.72) | +2.12 (+1.16 to +3.08) | +2.04 (+1.24 to +2.84) | +7.09 (+5.75 to +8.53) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | +10.16 (+8.84 to +11.52) | +2.20 (+1.12 to +3.32) | +3.16 (+2.20 to +4.12) | +7.54 (+6.20 to +9.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | +9.80 (+8.48 to +11.12) | +2.48 (+1.44 to +3.56) | +3.00 (+2.04 to +3.88) | +6.79 (+5.50 to +8.18) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | +10.88 (+9.52 to +12.24) | +4.00 (+2.88 to +5.16) | +4.12 (+3.16 to +5.00) | +8.33 (+6.89 to +9.77) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +9.20 (+7.92 to +10.44) | +2.96 (+2.12 to +3.84) | +2.16 (+1.56 to +2.76) | +6.30 (+5.06 to +7.59) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +10.32 (+9.00 to +11.64) | +3.40 (+2.48 to +4.40) | +2.72 (+1.92 to +3.44) | +7.39 (+6.05 to +8.78) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +10.32 (+9.00 to +11.68) | +4.00 (+2.88 to +5.12) | +4.12 (+3.16 to +5.00) | +7.98 (+6.54 to +9.37) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | +9.96 (+8.68 to +11.28) | +3.96 (+2.88 to +5.04) | +3.88 (+3.00 to +4.76) | +7.59 (+6.35 to +8.92) |
| best under 2 passes per question (any configuration) | T1_k0_cot8b | +8.44 (+7.16 to +9.76) | +0.36 (-0.20 to +0.96) | +1.00 (+0.48 to +1.52) | +4.56 (+3.52 to +5.70) |
| best under 3 passes per question (any configuration) | T1_k0_cot8b | +9.48 (+8.16 to +10.80) | +2.08 (+1.12 to +3.08) | +2.56 (+1.64 to +3.48) | +6.20 (+5.01 to +7.54) |
| best under 5 passes per question (any configuration) | T1_k0_cot8b | +10.24 (+8.80 to +11.68) | +1.48 (+0.40 to +2.60) | +2.92 (+1.92 to +3.92) | +7.78 (+6.40 to +9.32) |
