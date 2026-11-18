# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_esc_cot8b_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus 2 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.36% vs 85.36%, T1_k0_ground_cot 85.55% vs 85.55%, T1_k0_cot 85.42% vs 85.42%, T1_k0_zoom_cot 85.62% vs 85.62%, T1_k0_cot_multi 86.01% vs 86.01%, T1_k0_ground_cot_multi 86.08% vs 86.08%, T1_k0_esc8b 88.25% vs 88.25%, T1_k0_ground_cot_multi_esc8b 88.26% vs 88.26%, T1_k0_ground_cot_multi_esc 87.55% vs 87.55%, T1_k0_everything_multi_esc 87.38% vs 87.38%, T1_k0_cot8b 88.23% vs 88.23%, T1_k0_ground_cot_multi_cot8b 88.48% vs 88.48%, T1_k0_ground_cot_multi_esc8b_cot8b 89.31% vs 89.31%

Seconds: estimated wall-clock seconds per question on one H100, from the measured component times (results/latency.json).

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Seconds | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 24.13 | 16.00 | 4.63 | 85.36% | +2.58 (+2.07 to +3.09) | +418 / -172 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 6.39 | 3.92 | 1.05 | 84.92% | +2.14 (+1.71 to +2.56) | +314 / -110 | +1.59 (+1.19 to +1.96) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 1.51 | 85.19% | +2.42 (+1.93 to +2.89) | +375 / -145 | +1.61 (+1.21 to +2.02) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 13.31 | 8.46 | 2.41 | 85.35% | +2.57 (+2.06 to +3.07) | +414 / -169 | +1.39 (+1.03 to +1.73) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 30% / 29.6% | 8.84 | 5.43 | 1.51 | 85.19% | +2.42 (+1.93 to +2.89) | +375 / -145 | +1.61 (+1.21 to +2.02) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 8.53 | 5.00 | 2.91 | 85.55% | +2.77 (+2.26 to +3.27) | +419 / -155 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 19.4% | 2.80 | 1.78 | 0.68 | 85.09% | +2.31 (+1.90 to +2.72) | +308 / -88 | +1.75 (+1.37 to +2.12) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 0.97 | 85.41% | +2.63 (+2.16 to +3.08) | +373 / -123 | +1.84 (+1.45 to +2.24) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 5.11 | 2.99 | 1.53 | 85.54% | +2.76 (+2.26 to +3.26) | +414 / -151 | +1.45 (+1.10 to +1.80) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 3.62 | 2.18 | 0.97 | 85.41% | +2.63 (+2.16 to +3.08) | +373 / -123 | +1.84 (+1.45 to +2.24) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 4.52 | 2.00 | 1.87 | 85.42% | +2.64 (+2.08 to +3.15) | +439 / -188 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.83 | 1.19 | 0.47 | 84.96% | +2.19 (+1.77 to +2.57) | +302 / -94 | +1.63 (+1.24 to +2.00) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 0.64 | 85.22% | +2.44 (+1.95 to +2.89) | +369 / -137 | +1.67 (+1.27 to +2.07) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.93 | 1.50 | 1.00 | 85.47% | +2.69 (+2.14 to +3.21) | +435 / -179 | +1.41 (+1.02 to +1.79) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 2.22 | 1.30 | 0.64 | 85.22% | +2.44 (+1.95 to +2.89) | +369 / -137 | +1.67 (+1.27 to +2.07) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 12.57 | 6.00 | 2.69 | 85.62% | +2.84 (+2.30 to +3.34) | +440 / -170 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 19.4% | 3.53 | 1.97 | 0.63 | 85.16% | +2.39 (+1.95 to +2.81) | +326 / -99 | +1.77 (+1.38 to +2.12) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 0.90 | 85.50% | +2.72 (+2.22 to +3.19) | +394 / -135 | +1.88 (+1.47 to +2.29) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 7.04 | 3.49 | 1.41 | 85.59% | +2.82 (+2.29 to +3.32) | +435 / -167 | +1.54 (+1.16 to +1.90) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 4.76 | 2.48 | 0.90 | 85.50% | +2.72 (+2.22 to +3.19) | +394 / -135 | +1.88 (+1.47 to +2.29) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 8.10 | 5.00 | 3.83 | 86.01% | +3.24 (+2.72 to +3.75) | +459 / -151 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 19.4% | 2.72 | 1.78 | 0.86 | 85.26% | +2.48 (+2.06 to +2.90) | +327 / -91 | +1.74 (+1.36 to +2.11) |
| best detector at a 30% budget | T1_k0_cot_multi | token | 30% / 29.6% | 3.50 | 2.18 | 1.24 | 85.68% | +2.90 (+2.43 to +3.37) | +396 / -120 | +1.90 (+1.51 to +2.30) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 49.7% | 4.93 | 2.99 | 1.99 | 86.05% | +3.27 (+2.75 to +3.77) | +455 / -144 | +1.65 (+1.29 to +2.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 30% / 29.6% | 3.50 | 2.18 | 1.24 | 85.68% | +2.90 (+2.43 to +3.37) | +396 / -120 | +1.90 (+1.51 to +2.30) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 12.02 | 8.00 | 4.87 | 86.08% | +3.30 (+2.79 to +3.80) | +457 / -143 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 19.4% | 3.66 | 2.36 | 1.07 | 85.32% | +2.54 (+2.13 to +2.97) | +330 / -88 | +1.85 (+1.46 to +2.22) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 29.6% | 4.85 | 3.07 | 1.56 | 85.77% | +2.99 (+2.53 to +3.47) | +401 / -116 | +2.03 (+1.65 to +2.43) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 49.7% | 7.04 | 4.48 | 2.51 | 86.11% | +3.33 (+2.82 to +3.81) | +453 / -136 | +1.70 (+1.34 to +2.07) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 29.6% | 4.85 | 3.07 | 1.56 | 85.77% | +2.99 (+2.53 to +3.47) | +401 / -116 | +2.03 (+1.65 to +2.43) |
| correct every question (no gate) | T1_k0_esc8b | token | 100% / 100.0% | 7.46 | 5.00 | 0.41 | 88.25% | +5.47 (+4.81 to +6.08) | +733 / -212 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_esc8b | token | 20% / 19.4% | 2.43 | 1.78 | 0.18 | 86.40% | +3.63 (+3.14 to +4.11) | +461 / -116 | +2.60 (+2.14 to +3.09) |
| best detector at a 30% budget | T1_k0_esc8b | after_head | 30% / 28.2% | 3.71 | 2.13 | 0.25 | 87.40% | +4.62 (+4.06 to +5.18) | +605 / -165 | +3.03 (+2.55 to +3.53) |
| best detector at a 50% budget | T1_k0_esc8b | token | 50% / 49.7% | 4.41 | 2.99 | 0.28 | 88.18% | +5.40 (+4.75 to +6.00) | +706 / -192 | +2.75 (+2.28 to +3.20) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | token | 40% / 39.8% | 3.79 | 2.59 | 0.25 | 88.00% | +5.22 (+4.60 to +5.80) | +671 / -174 | +3.01 (+2.52 to +3.50) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | token | 100% / 100.0% | 16.25 | 12.00 | 5.04 | 88.26% | +5.48 (+4.90 to +6.04) | +678 / -156 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | token | 20% / 19.4% | 4.57 | 3.14 | 1.11 | 86.53% | +3.75 (+3.29 to +4.22) | +449 / -92 | +2.57 (+2.15 to +3.01) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | after_head | 30% / 28.2% | 6.66 | 4.10 | 1.59 | 87.49% | +4.71 (+4.16 to +5.25) | +577 / -129 | +2.93 (+2.48 to +3.39) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | token | 50% / 49.7% | 9.24 | 6.47 | 2.60 | 88.13% | +5.35 (+4.76 to +5.88) | +656 / -147 | +2.56 (+2.13 to +2.97) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | token | 40% / 39.8% | 7.79 | 5.38 | 2.12 | 87.96% | +5.18 (+4.61 to +5.71) | +630 / -137 | +2.79 (+2.35 to +3.26) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | token | 100% / 100.0% | 26.98 | 22.25 | 7.79 | 87.55% | +4.77 (+4.22 to +5.29) | +552 / -98 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | token | 20% / 19.4% | 6.81 | 5.13 | 1.65 | 86.26% | +3.48 (+3.03 to +3.93) | +398 / -67 | +2.43 (+2.03 to +2.84) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | token | 30% / 29.6% | 9.54 | 7.28 | 2.43 | 86.99% | +4.21 (+3.72 to +4.71) | +488 / -87 | +2.71 (+2.29 to +3.13) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | token | 50% / 49.7% | 14.71 | 11.56 | 3.98 | 87.52% | +4.74 (+4.20 to +5.25) | +546 / -95 | +2.21 (+1.84 to +2.57) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | token | 35% / 34.9% | 10.93 | 8.41 | 2.84 | 87.23% | +4.46 (+3.95 to +4.96) | +514 / -90 | +2.61 (+2.20 to +3.03) |
| correct every question (no gate) | T1_k0_everything_multi_esc | token | 100% / 100.0% | 36.78 | 28.25 | 9.36 | 87.38% | +4.60 (+4.09 to +5.12) | +535 / -97 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | token | 20% / 19.4% | 8.82 | 6.30 | 1.96 | 86.15% | +3.37 (+2.93 to +3.84) | +388 / -67 | +2.35 (+1.96 to +2.75) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | token | 30% / 29.6% | 12.56 | 9.05 | 2.91 | 86.90% | +4.12 (+3.63 to +4.60) | +478 / -86 | +2.66 (+2.26 to +3.07) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | token | 50% / 49.7% | 19.64 | 14.55 | 4.76 | 87.36% | +4.58 (+4.07 to +5.10) | +530 / -94 | +2.14 (+1.79 to +2.51) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | token | 35% / 34.9% | 14.47 | 10.50 | 3.40 | 87.11% | +4.33 (+3.81 to +4.83) | +501 / -89 | +2.54 (+2.15 to +2.94) |
| correct every question (no gate) | T1_k0_cot8b | token | 100% / 100.0% | 7.75 | 5.00 | 2.25 | 88.23% | +5.45 (+4.80 to +6.06) | +748 / -229 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot8b | judge | 20% / 18.3% | 3.09 | 1.73 | 0.55 | 86.25% | +3.47 (+2.99 to +3.90) | +410 / -80 | +2.31 (+1.86 to +2.77) |
| best detector at a 30% budget | T1_k0_cot8b | after_head | 30% / 28.2% | 3.78 | 2.13 | 0.77 | 87.90% | +5.12 (+4.53 to +5.68) | +642 / -155 | +3.47 (+2.99 to +3.96) |
| best detector at a 50% budget | T1_k0_cot8b | token | 50% / 49.7% | 4.53 | 2.99 | 1.19 | 88.14% | +5.36 (+4.72 to +5.95) | +714 / -204 | +2.65 (+2.18 to +3.11) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | token | 40% / 39.8% | 3.88 | 2.59 | 0.98 | 88.00% | +5.22 (+4.59 to +5.79) | +671 / -174 | +3.04 (+2.54 to +3.55) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | token | 100% / 100.0% | 16.37 | 12.00 | 6.88 | 88.48% | +5.71 (+5.06 to +6.27) | +728 / -185 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | judge | 20% / 18.3% | 4.83 | 3.01 | 1.41 | 86.35% | +3.57 (+3.12 to +3.99) | +408 / -68 | +2.32 (+1.88 to +2.76) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | after_head | 30% / 28.2% | 6.68 | 4.10 | 2.10 | 88.01% | +5.23 (+4.68 to +5.78) | +633 / -135 | +3.40 (+2.93 to +3.88) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | token | 50% / 49.7% | 9.26 | 6.47 | 3.51 | 88.32% | +5.54 (+4.92 to +6.11) | +697 / -170 | +2.57 (+2.13 to +3.03) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | token | 40% / 39.8% | 7.80 | 5.38 | 2.85 | 88.15% | +5.37 (+4.77 to +5.93) | +663 / -152 | +2.93 (+2.47 to +3.40) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | token | 100% / 100.0% | 20.55 | 16.00 | 7.04 | 89.31% | +6.54 (+5.89 to +7.10) | +746 / -124 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | after_head | 20% / 19.2% | 6.18 | 3.88 | 1.53 | 87.78% | +5.00 (+4.49 to +5.48) | +551 / -75 | +3.63 (+3.15 to +4.09) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | after_head | 30% / 28.2% | 7.93 | 5.23 | 2.15 | 88.57% | +5.79 (+5.22 to +6.34) | +649 / -98 | +3.79 (+3.30 to +4.26) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | token | 50% / 49.7% | 11.42 | 8.46 | 3.60 | 89.12% | +6.35 (+5.73 to +6.88) | +721 / -117 | +2.99 (+2.54 to +3.41) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | token | 40% / 39.8% | 9.55 | 6.98 | 2.92 | 88.89% | +6.12 (+5.54 to +6.65) | +691 / -109 | +3.36 (+2.90 to +3.82) |
| best under 2 passes per question (any configuration) | T1_k0_cot8b | token | 10% / 9.4% | 1.71 | 1.38 | 0.32 | 85.34% | +2.56 (+2.19 to +2.95) | +292 / -48 | +1.96 (+1.59 to +2.35) |
| best under 3 passes per question (any configuration) | T1_k0_cot8b | pre_head | 25% / 23.2% | 2.70 | 1.93 | 0.62 | 87.51% | +4.73 (+4.17 to +5.25) | +570 / -120 | +3.29 (+2.82 to +3.80) |
| best under 5 passes per question (any configuration) | T1_k0_cot8b | token | 55% / 54.7% | 4.86 | 3.19 | 1.30 | 88.26% | +5.48 (+4.82 to +6.09) | +734 / -212 | +2.55 (+2.10 to +3.01) |
| best under 0.3 s per question on one H100 (any configuration) | T1_k0_esc8b | token | 55% / 54.7% | 4.72 | 3.19 | 0.29 | 88.27% | +5.50 (+4.84 to +6.09) | +720 / -197 | +2.58 (+2.12 to +3.04) |
| best under 0.5 s per question on one H100 (any configuration) | T1_k0_esc8b | token | 60% / 59.5% | 5.02 | 3.38 | 0.30 | 88.27% | +5.50 (+4.83 to +6.10) | +726 / -203 | +2.16 (+1.72 to +2.57) |
| best under 1 s per question on one H100 (any configuration) | T1_k0_cot8b | token | 40% / 39.8% | 3.88 | 2.59 | 0.98 | 88.00% | +5.22 (+4.59 to +5.79) | +671 / -174 | +3.04 (+2.54 to +3.55) |
| best under 2 s per question on one H100 (any configuration) | T1_k0_cot8b | token | 55% / 54.7% | 4.86 | 3.19 | 1.30 | 88.26% | +5.48 (+4.82 to +6.09) | +734 / -212 | +2.55 (+2.10 to +3.01) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +4.80 (+3.64 to +6.00) | +2.28 (+1.32 to +3.24) | +1.00 (+0.28 to +1.64) | +2.18 (+1.04 to +3.42) |
| best detector at a 20% budget | T1_k5_everything | +4.24 (+3.20 to +5.32) | +2.08 (+1.28 to +2.92) | +0.56 (+0.04 to +1.04) | +1.59 (+0.64 to +2.53) |
| best detector at a 30% budget | T1_k5_everything | +4.60 (+3.52 to +5.72) | +2.32 (+1.44 to +3.20) | +0.68 (+0.00 to +1.24) | +1.98 (+0.94 to +3.12) |
| best detector at a 50% budget | T1_k5_everything | +4.72 (+3.60 to +5.92) | +2.28 (+1.32 to +3.24) | +1.04 (+0.32 to +1.68) | +2.18 (+1.04 to +3.37) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +4.60 (+3.52 to +5.72) | +2.32 (+1.44 to +3.20) | +0.68 (+0.00 to +1.24) | +1.98 (+0.94 to +3.12) |
| correct every question (no gate) | T1_k0_ground_cot | +5.72 (+4.52 to +6.88) | +1.96 (+1.12 to +2.80) | +0.80 (+0.08 to +1.52) | +2.58 (+1.49 to +3.77) |
| best detector at a 20% budget | T1_k0_ground_cot | +4.72 (+3.72 to +5.80) | +1.96 (+1.24 to +2.72) | +0.56 (+0.08 to +1.04) | +1.93 (+1.04 to +2.88) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.40 (+4.28 to +6.48) | +2.12 (+1.32 to +2.92) | +0.64 (-0.04 to +1.24) | +2.28 (+1.29 to +3.37) |
| best detector at a 50% budget | T1_k0_ground_cot | +5.60 (+4.40 to +6.80) | +2.04 (+1.20 to +2.88) | +0.84 (+0.12 to +1.52) | +2.53 (+1.44 to +3.72) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.40 (+4.28 to +6.48) | +2.12 (+1.32 to +2.92) | +0.64 (-0.04 to +1.24) | +2.28 (+1.29 to +3.37) |
| correct every question (no gate) | T1_k0_cot | +6.60 (+5.36 to +7.88) | +0.68 (-0.20 to +1.60) | +0.40 (-0.36 to +1.08) | +2.93 (+1.78 to +4.21) |
| best detector at a 20% budget | T1_k0_cot | +5.16 (+4.12 to +6.20) | +1.28 (+0.60 to +2.00) | +0.32 (-0.20 to +0.80) | +1.93 (+1.09 to +2.88) |
| best detector at a 30% budget | T1_k0_cot | +5.92 (+4.80 to +7.04) | +1.00 (+0.20 to +1.80) | +0.40 (-0.28 to +1.04) | +2.43 (+1.44 to +3.57) |
| best detector at a 50% budget | T1_k0_cot | +6.48 (+5.28 to +7.76) | +0.80 (-0.04 to +1.68) | +0.48 (-0.28 to +1.16) | +3.07 (+1.93 to +4.31) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.92 (+4.80 to +7.04) | +1.00 (+0.20 to +1.80) | +0.40 (-0.28 to +1.04) | +2.43 (+1.44 to +3.57) |
| correct every question (no gate) | T1_k0_zoom_cot | +5.52 (+4.28 to +6.72) | +2.24 (+1.32 to +3.20) | +0.96 (+0.20 to +1.64) | +2.58 (+1.44 to +3.77) |
| best detector at a 20% budget | T1_k0_zoom_cot | +4.68 (+3.64 to +5.76) | +2.16 (+1.36 to +3.00) | +0.68 (+0.16 to +1.20) | +1.93 (+1.04 to +2.88) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.28 (+4.16 to +6.44) | +2.36 (+1.48 to +3.28) | +0.80 (+0.16 to +1.44) | +2.38 (+1.34 to +3.52) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.40 (+4.16 to +6.60) | +2.28 (+1.36 to +3.24) | +1.00 (+0.24 to +1.68) | +2.53 (+1.39 to +3.72) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.28 (+4.16 to +6.44) | +2.36 (+1.48 to +3.28) | +0.80 (+0.16 to +1.44) | +2.38 (+1.34 to +3.52) |
| correct every question (no gate) | T1_k0_cot_multi | +7.08 (+5.88 to +8.36) | +1.40 (+0.56 to +2.28) | +1.16 (+0.48 to +1.80) | +3.32 (+2.13 to +4.61) |
| best detector at a 20% budget | T1_k0_cot_multi | +5.32 (+4.24 to +6.40) | +1.64 (+0.96 to +2.40) | +0.76 (+0.24 to +1.28) | +2.13 (+1.19 to +3.17) |
| best detector at a 30% budget | T1_k0_cot_multi | +6.32 (+5.20 to +7.52) | +1.56 (+0.80 to +2.36) | +1.04 (+0.40 to +1.64) | +2.63 (+1.54 to +3.82) |
| best detector at a 50% budget | T1_k0_cot_multi | +6.96 (+5.76 to +8.24) | +1.52 (+0.68 to +2.36) | +1.16 (+0.48 to +1.80) | +3.47 (+2.28 to +4.71) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +6.32 (+5.20 to +7.52) | +1.56 (+0.80 to +2.36) | +1.04 (+0.40 to +1.64) | +2.63 (+1.54 to +3.82) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +6.68 (+5.48 to +7.92) | +2.40 (+1.52 to +3.28) | +1.00 (+0.32 to +1.68) | +3.07 (+1.93 to +4.36) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +4.96 (+3.92 to +6.00) | +2.32 (+1.56 to +3.12) | +0.68 (+0.16 to +1.20) | +2.13 (+1.14 to +3.17) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +5.92 (+4.80 to +7.08) | +2.44 (+1.60 to +3.28) | +0.92 (+0.28 to +1.52) | +2.63 (+1.54 to +3.77) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +6.56 (+5.36 to +7.80) | +2.48 (+1.60 to +3.36) | +1.04 (+0.36 to +1.72) | +3.22 (+2.08 to +4.46) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +5.92 (+4.80 to +7.08) | +2.44 (+1.60 to +3.28) | +0.92 (+0.28 to +1.52) | +2.63 (+1.54 to +3.77) |
| correct every question (no gate) | T1_k0_esc8b | +6.72 (+5.40 to +8.08) | +3.76 (+2.56 to +4.88) | +4.32 (+3.32 to +5.28) | +7.49 (+6.10 to +8.92) |
| best detector at a 20% budget | T1_k0_esc8b | +5.00 (+3.92 to +6.12) | +2.72 (+1.76 to +3.72) | +2.36 (+1.64 to +3.08) | +4.61 (+3.57 to +5.75) |
| best detector at a 30% budget | T1_k0_esc8b | +6.24 (+4.96 to +7.56) | +3.20 (+2.16 to +4.20) | +2.80 (+2.00 to +3.56) | +6.64 (+5.35 to +8.03) |
| best detector at a 50% budget | T1_k0_esc8b | +6.60 (+5.28 to +7.92) | +3.80 (+2.68 to +4.88) | +4.20 (+3.24 to +5.16) | +7.39 (+6.00 to +8.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | +6.52 (+5.24 to +7.80) | +3.68 (+2.52 to +4.80) | +3.96 (+3.04 to +4.88) | +7.09 (+5.80 to +8.43) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | +7.44 (+6.16 to +8.76) | +3.92 (+2.76 to +5.04) | +3.68 (+2.80 to +4.56) | +7.24 (+5.90 to +8.58) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | +5.32 (+4.24 to +6.44) | +3.08 (+2.16 to +4.08) | +2.08 (+1.44 to +2.72) | +4.71 (+3.67 to +5.80) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | +6.92 (+5.72 to +8.20) | +3.36 (+2.36 to +4.36) | +2.24 (+1.52 to +2.96) | +6.69 (+5.45 to +7.98) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | +6.92 (+5.68 to +8.16) | +3.96 (+2.84 to +5.08) | +3.68 (+2.80 to +4.52) | +7.19 (+5.85 to +8.53) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | +6.76 (+5.52 to +8.00) | +3.88 (+2.80 to +5.00) | +3.44 (+2.60 to +4.28) | +6.99 (+5.80 to +8.28) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | +7.56 (+6.40 to +8.72) | +3.88 (+2.88 to +4.88) | +2.84 (+2.04 to +3.60) | +4.81 (+3.67 to +6.05) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | +5.80 (+4.80 to +6.84) | +3.32 (+2.48 to +4.24) | +1.36 (+0.80 to +1.92) | +3.42 (+2.43 to +4.46) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | +6.84 (+5.72 to +7.92) | +3.68 (+2.76 to +4.64) | +2.08 (+1.40 to +2.80) | +4.26 (+3.17 to +5.40) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | +7.44 (+6.28 to +8.60) | +3.88 (+2.88 to +4.88) | +2.84 (+2.08 to +3.64) | +4.81 (+3.67 to +6.00) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | +7.24 (+6.08 to +8.36) | +3.76 (+2.80 to +4.72) | +2.32 (+1.60 to +3.04) | +4.51 (+3.37 to +5.70) |
| correct every question (no gate) | T1_k0_everything_multi_esc | +7.32 (+6.20 to +8.52) | +3.80 (+2.80 to +4.76) | +2.72 (+1.92 to +3.48) | +4.56 (+3.47 to +5.75) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | +5.76 (+4.76 to +6.80) | +3.20 (+2.36 to +4.08) | +1.32 (+0.76 to +1.88) | +3.17 (+2.23 to +4.21) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | +6.80 (+5.68 to +7.88) | +3.64 (+2.68 to +4.60) | +2.04 (+1.36 to +2.72) | +3.97 (+2.93 to +5.11) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | +7.24 (+6.12 to +8.40) | +3.80 (+2.80 to +4.76) | +2.72 (+1.96 to +3.48) | +4.56 (+3.47 to +5.75) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | +7.08 (+5.96 to +8.20) | +3.72 (+2.76 to +4.68) | +2.28 (+1.56 to +2.96) | +4.21 (+3.12 to +5.40) |
| correct every question (no gate) | T1_k0_cot8b | +10.12 (+8.72 to +11.56) | +1.40 (+0.32 to +2.48) | +2.80 (+1.80 to +3.80) | +7.98 (+6.59 to +9.47) |
| best detector at a 20% budget | T1_k0_cot8b | +7.12 (+5.96 to +8.24) | +0.40 (-0.08 to +0.88) | +0.68 (+0.24 to +1.12) | +6.20 (+4.91 to +7.59) |
| best detector at a 30% budget | T1_k0_cot8b | +10.04 (+8.72 to +11.44) | +1.64 (+0.72 to +2.52) | +2.00 (+1.16 to +2.80) | +7.19 (+5.85 to +8.63) |
| best detector at a 50% budget | T1_k0_cot8b | +9.96 (+8.64 to +11.32) | +1.60 (+0.56 to +2.64) | +2.80 (+1.80 to +3.80) | +7.49 (+6.10 to +8.97) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot8b | +9.72 (+8.44 to +11.08) | +1.84 (+0.88 to +2.88) | +2.80 (+1.84 to +3.80) | +6.84 (+5.60 to +8.23) |
| correct every question (no gate) | T1_k0_ground_cot_multi_cot8b | +10.88 (+9.52 to +12.20) | +2.00 (+0.92 to +3.12) | +2.84 (+1.92 to +3.76) | +7.44 (+6.15 to +8.78) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_cot8b | +7.56 (+6.40 to +8.72) | +0.52 (+0.04 to +1.04) | +0.60 (+0.24 to +1.00) | +6.10 (+4.91 to +7.44) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_cot8b | +10.44 (+9.16 to +11.84) | +2.04 (+1.16 to +2.96) | +1.84 (+1.08 to +2.56) | +6.94 (+5.70 to +8.28) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_cot8b | +10.24 (+8.92 to +11.64) | +2.16 (+1.08 to +3.28) | +2.88 (+1.96 to +3.80) | +7.19 (+5.85 to +8.53) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_cot8b | +10.00 (+8.72 to +11.32) | +2.44 (+1.40 to +3.48) | +2.68 (+1.80 to +3.56) | +6.59 (+5.30 to +7.88) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b_cot8b | +10.76 (+9.40 to +12.12) | +4.00 (+2.92 to +5.08) | +3.80 (+2.88 to +4.68) | +7.83 (+6.54 to +9.22) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +9.04 (+7.76 to +10.32) | +2.96 (+2.16 to +3.80) | +2.04 (+1.44 to +2.64) | +6.20 (+5.06 to +7.44) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +10.24 (+8.92 to +11.60) | +3.44 (+2.56 to +4.40) | +2.48 (+1.76 to +3.20) | +7.29 (+6.05 to +8.63) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b_cot8b | +10.16 (+8.88 to +11.52) | +4.04 (+2.96 to +5.12) | +3.76 (+2.88 to +4.68) | +7.68 (+6.40 to +9.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b_cot8b | +9.88 (+8.60 to +11.20) | +4.04 (+3.00 to +5.08) | +3.48 (+2.60 to +4.36) | +7.29 (+6.10 to +8.63) |
| best under 2 passes per question (any configuration) | T1_k0_cot8b | +5.32 (+4.32 to +6.36) | +1.24 (+0.64 to +1.84) | +1.04 (+0.64 to +1.48) | +2.68 (+1.83 to +3.57) |
| best under 3 passes per question (any configuration) | T1_k0_cot8b | +9.72 (+8.40 to +11.08) | +1.24 (+0.52 to +2.00) | +1.52 (+0.84 to +2.20) | +6.84 (+5.55 to +8.18) |
| best under 5 passes per question (any configuration) | T1_k0_cot8b | +10.12 (+8.76 to +11.56) | +1.56 (+0.52 to +2.60) | +2.88 (+1.88 to +3.88) | +7.83 (+6.49 to +9.32) |
| best under 0.3 s per question on one H100 (any configuration) | T1_k0_esc8b | +6.76 (+5.48 to +8.12) | +3.80 (+2.68 to +4.88) | +4.32 (+3.36 to +5.28) | +7.49 (+6.10 to +8.92) |
| best under 0.5 s per question on one H100 (any configuration) | T1_k0_esc8b | +6.72 (+5.40 to +8.08) | +3.80 (+2.68 to +4.88) | +4.36 (+3.40 to +5.32) | +7.49 (+6.10 to +8.92) |
| best under 1 s per question on one H100 (any configuration) | T1_k0_cot8b | +9.72 (+8.44 to +11.08) | +1.84 (+0.88 to +2.88) | +2.80 (+1.84 to +3.80) | +6.84 (+5.60 to +8.23) |
| best under 2 s per question on one H100 (any configuration) | T1_k0_cot8b | +10.12 (+8.76 to +11.56) | +1.56 (+0.52 to +2.60) | +2.88 (+1.88 to +3.88) | +7.83 (+6.49 to +9.32) |
