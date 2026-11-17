# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_esc_drop_judge_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus 1 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only. Ranker inputs zeroed: judge.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.64% vs 85.64%, T1_k0_ground_cot 85.75% vs 85.75%, T1_k0_cot 85.34% vs 85.34%, T1_k0_zoom_cot 85.57% vs 85.57%, T1_k0_cot_multi 86.10% vs 86.10%, T1_k0_ground_cot_multi 86.28% vs 86.28%, T1_k0_esc8b 88.38% vs 88.38%, T1_k0_ground_cot_multi_esc8b 88.50% vs 88.50%, T1_k0_ground_cot_multi_esc 87.13% vs 87.13%, T1_k0_everything_multi_esc 87.09% vs 87.09%

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 20.07 | 16.00 | 85.64% | +2.86 (+2.34 to +3.37) | +447 / -175 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 5.16 | 3.92 | 85.14% | +2.36 (+1.92 to +2.81) | +334 / -109 | +1.70 (+1.30 to +2.11) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 7.14 | 5.43 | 85.53% | +2.75 (+2.27 to +3.24) | +401 / -139 | +1.89 (+1.48 to +2.31) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 10.88 | 8.46 | 85.70% | +2.92 (+2.42 to +3.43) | +444 / -166 | +1.60 (+1.23 to +1.94) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 25% / 24.4% | 6.13 | 4.66 | 85.42% | +2.64 (+2.19 to +3.09) | +372 / -121 | +1.84 (+1.44 to +2.25) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 6.76 | 5.00 | 85.75% | +2.97 (+2.47 to +3.49) | +433 / -150 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 19.4% | 2.29 | 1.78 | 85.24% | +2.46 (+2.02 to +2.89) | +322 / -88 | +1.82 (+1.42 to +2.21) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 85.60% | +2.83 (+2.35 to +3.28) | +383 / -114 | +1.99 (+1.59 to +2.39) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 4.05 | 2.99 | 85.79% | +3.02 (+2.52 to +3.51) | +429 / -142 | +1.64 (+1.27 to +2.01) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 85.60% | +2.83 (+2.35 to +3.28) | +383 / -114 | +1.99 (+1.59 to +2.39) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 3.26 | 2.00 | 85.34% | +2.56 (+2.03 to +3.08) | +418 / -174 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.51 | 1.19 | 84.85% | +2.07 (+1.67 to +2.47) | +290 / -93 | +1.45 (+1.07 to +1.82) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 85.13% | +2.35 (+1.89 to +2.79) | +350 / -126 | +1.60 (+1.20 to +1.98) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.21 | 1.50 | 85.38% | +2.61 (+2.07 to +3.11) | +412 / -164 | +1.46 (+1.08 to +1.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 85.13% | +2.35 (+1.89 to +2.79) | +350 / -126 | +1.60 (+1.20 to +1.98) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 9.29 | 6.00 | 85.57% | +2.79 (+2.29 to +3.29) | +417 / -151 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | after_head | 20% / 19.2% | 3.52 | 1.96 | 85.32% | +2.54 (+2.08 to +2.97) | +349 / -107 | +1.87 (+1.47 to +2.28) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 85.48% | +2.70 (+2.22 to +3.15) | +375 / -118 | +1.86 (+1.46 to +2.26) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 5.26 | 3.49 | 85.62% | +2.84 (+2.34 to +3.32) | +414 / -144 | +1.58 (+1.21 to +1.92) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 85.48% | +2.70 (+2.22 to +3.15) | +375 / -118 | +1.86 (+1.46 to +2.26) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 6.55 | 5.00 | 86.10% | +3.32 (+2.82 to +3.82) | +453 / -137 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 19.4% | 2.25 | 1.78 | 85.31% | +2.53 (+2.13 to +2.94) | +325 / -84 | +1.80 (+1.41 to +2.15) |
| best detector at a 30% budget | T1_k0_cot_multi | after_head | 30% / 28.2% | 3.47 | 2.13 | 85.93% | +3.15 (+2.67 to +3.65) | +428 / -128 | +2.15 (+1.77 to +2.56) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 49.7% | 3.96 | 2.99 | 86.10% | +3.32 (+2.84 to +3.81) | +445 / -129 | +1.68 (+1.31 to +2.03) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | after_head | 30% / 28.2% | 3.47 | 2.13 | 85.93% | +3.15 (+2.67 to +3.65) | +428 / -128 | +2.15 (+1.77 to +2.56) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 10.01 | 8.00 | 86.28% | +3.50 (+3.01 to +4.01) | +471 / -138 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 19.4% | 3.01 | 2.36 | 85.50% | +2.72 (+2.31 to +3.16) | +346 / -87 | +1.92 (+1.52 to +2.30) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 85.91% | +3.13 (+2.68 to +3.59) | +409 / -111 | +2.10 (+1.70 to +2.51) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 49.7% | 5.76 | 4.48 | 86.28% | +3.50 (+2.99 to +4.01) | +464 / -131 | +1.80 (+1.44 to +2.15) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 85.91% | +3.13 (+2.68 to +3.59) | +409 / -111 | +2.10 (+1.70 to +2.51) |
| correct every question (no gate) | T1_k0_esc8b | token | 100% / 100.0% | 6.23 | 5.00 | 88.38% | +5.60 (+4.94 to +6.21) | +741 / -208 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_esc8b | token | 20% / 19.4% | 2.10 | 1.78 | 86.42% | +3.65 (+3.17 to +4.13) | +463 / -116 | +2.62 (+2.16 to +3.09) |
| best detector at a 30% budget | T1_k0_esc8b | after_head | 30% / 28.2% | 3.28 | 2.13 | 87.46% | +4.69 (+4.12 to +5.24) | +613 / -167 | +3.05 (+2.55 to +3.56) |
| best detector at a 50% budget | T1_k0_esc8b | token | 50% / 49.7% | 3.70 | 2.99 | 88.26% | +5.48 (+4.83 to +6.08) | +712 / -190 | +2.79 (+2.32 to +3.24) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | token | 40% / 39.8% | 3.19 | 2.59 | 88.12% | +5.34 (+4.73 to +5.93) | +677 / -169 | +3.06 (+2.57 to +3.55) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | token | 100% / 100.0% | 14.13 | 12.00 | 88.50% | +5.73 (+5.08 to +6.32) | +742 / -197 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | token | 20% / 19.4% | 3.86 | 3.14 | 86.58% | +3.80 (+3.31 to +4.30) | +474 / -112 | +2.68 (+2.22 to +3.15) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | after_head | 30% / 28.2% | 5.74 | 4.10 | 87.54% | +4.76 (+4.18 to +5.32) | +618 / -165 | +3.09 (+2.61 to +3.58) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | token | 50% / 49.7% | 7.85 | 6.47 | 88.38% | +5.60 (+4.97 to +6.19) | +715 / -182 | +2.83 (+2.35 to +3.29) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | token | 40% / 39.8% | 6.58 | 5.38 | 88.17% | +5.39 (+4.79 to +5.96) | +679 / -166 | +3.03 (+2.53 to +3.51) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | token | 100% / 100.0% | 24.47 | 22.25 | 87.13% | +4.35 (+3.84 to +4.88) | +520 / -106 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | token | 20% / 19.4% | 5.92 | 5.13 | 86.06% | +3.28 (+2.86 to +3.72) | +376 / -64 | +2.28 (+1.90 to +2.70) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | token | 30% / 29.6% | 8.34 | 7.28 | 86.71% | +3.93 (+3.46 to +4.43) | +459 / -85 | +2.56 (+2.15 to +2.95) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | token | 50% / 49.7% | 13.04 | 11.56 | 87.13% | +4.35 (+3.86 to +4.87) | +513 / -99 | +2.04 (+1.67 to +2.40) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | token | 30% / 29.6% | 8.34 | 7.28 | 86.71% | +3.93 (+3.46 to +4.43) | +459 / -85 | +2.56 (+2.15 to +2.95) |
| correct every question (no gate) | T1_k0_everything_multi_esc | token | 100% / 100.0% | 32.38 | 28.25 | 87.09% | +4.31 (+3.78 to +4.84) | +519 / -109 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | after_head | 20% / 19.2% | 8.19 | 6.23 | 86.14% | +3.36 (+2.93 to +3.82) | +396 / -76 | +2.41 (+2.00 to +2.82) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | token | 30% / 29.6% | 10.74 | 9.05 | 86.68% | +3.90 (+3.43 to +4.38) | +459 / -88 | +2.55 (+2.15 to +2.95) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | token | 50% / 49.7% | 17.00 | 14.55 | 87.08% | +4.30 (+3.78 to +4.82) | +512 / -103 | +2.01 (+1.65 to +2.37) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | token | 30% / 29.6% | 10.74 | 9.05 | 86.68% | +3.90 (+3.43 to +4.38) | +459 / -88 | +2.55 (+2.15 to +2.95) |
| best under 2 passes per question (any configuration) | T1_k0_esc8b | token | 15% / 14.4% | 1.82 | 1.58 | 85.77% | +2.99 (+2.56 to +3.44) | +376 / -91 | +2.25 (+1.82 to +2.70) |
| best under 3 passes per question (any configuration) | T1_k0_esc8b | token | 35% / 34.9% | 2.93 | 2.39 | 87.83% | +5.05 (+4.44 to +5.61) | +645 / -164 | +3.12 (+2.60 to +3.61) |
| best under 5 passes per question (any configuration) | T1_k0_esc8b | token | 60% / 59.5% | 4.20 | 3.38 | 88.39% | +5.61 (+4.95 to +6.21) | +734 / -200 | +2.22 (+1.79 to +2.64) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +5.40 (+4.16 to +6.60) | +2.64 (+1.68 to +3.68) | +1.04 (+0.40 to +1.68) | +2.23 (+0.99 to +3.57) |
| best detector at a 20% budget | T1_k5_everything | +4.56 (+3.44 to +5.64) | +2.36 (+1.52 to +3.24) | +0.56 (+0.08 to +1.04) | +1.88 (+0.89 to +2.93) |
| best detector at a 30% budget | T1_k5_everything | +5.16 (+4.00 to +6.28) | +2.60 (+1.68 to +3.52) | +0.84 (+0.28 to +1.40) | +2.33 (+1.24 to +3.57) |
| best detector at a 50% budget | T1_k5_everything | +5.32 (+4.12 to +6.52) | +2.64 (+1.68 to +3.68) | +1.08 (+0.44 to +1.72) | +2.58 (+1.39 to +3.92) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +4.96 (+3.84 to +6.04) | +2.56 (+1.72 to +3.48) | +0.88 (+0.36 to +1.40) | +2.03 (+0.99 to +3.17) |
| correct every question (no gate) | T1_k0_ground_cot | +5.92 (+4.72 to +7.12) | +2.24 (+1.40 to +3.12) | +1.04 (+0.44 to +1.64) | +2.63 (+1.44 to +3.92) |
| best detector at a 20% budget | T1_k0_ground_cot | +5.00 (+3.96 to +6.08) | +2.04 (+1.32 to +2.80) | +0.52 (+0.08 to +0.96) | +2.23 (+1.29 to +3.17) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.60 (+4.48 to +6.72) | +2.20 (+1.40 to +3.04) | +0.80 (+0.28 to +1.32) | +2.68 (+1.64 to +3.77) |
| best detector at a 50% budget | T1_k0_ground_cot | +5.84 (+4.64 to +7.04) | +2.24 (+1.40 to +3.12) | +1.04 (+0.44 to +1.64) | +2.93 (+1.74 to +4.16) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.60 (+4.48 to +6.72) | +2.20 (+1.40 to +3.04) | +0.80 (+0.28 to +1.32) | +2.68 (+1.64 to +3.77) |
| correct every question (no gate) | T1_k0_cot | +6.40 (+5.16 to +7.68) | +0.64 (-0.16 to +1.44) | +0.56 (-0.08 to +1.16) | +2.68 (+1.44 to +3.92) |
| best detector at a 20% budget | T1_k0_cot | +5.08 (+4.00 to +6.16) | +0.88 (+0.20 to +1.56) | +0.40 (-0.08 to +0.88) | +1.88 (+0.99 to +2.83) |
| best detector at a 30% budget | T1_k0_cot | +5.84 (+4.72 to +7.00) | +0.80 (+0.12 to +1.52) | +0.44 (-0.16 to +1.04) | +2.33 (+1.34 to +3.47) |
| best detector at a 50% budget | T1_k0_cot | +6.24 (+5.00 to +7.52) | +0.72 (-0.04 to +1.52) | +0.56 (-0.08 to +1.16) | +2.97 (+1.83 to +4.21) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +5.84 (+4.72 to +7.00) | +0.80 (+0.12 to +1.52) | +0.44 (-0.16 to +1.04) | +2.33 (+1.34 to +3.47) |
| correct every question (no gate) | T1_k0_zoom_cot | +5.72 (+4.52 to +6.92) | +2.24 (+1.40 to +3.12) | +0.96 (+0.40 to +1.56) | +2.13 (+0.99 to +3.37) |
| best detector at a 20% budget | T1_k0_zoom_cot | +5.64 (+4.48 to +6.72) | +1.52 (+0.76 to +2.24) | +0.56 (+0.16 to +1.00) | +2.43 (+1.39 to +3.52) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.36 (+4.20 to +6.48) | +2.28 (+1.48 to +3.12) | +0.80 (+0.28 to +1.32) | +2.28 (+1.24 to +3.37) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.64 (+4.44 to +6.84) | +2.24 (+1.40 to +3.12) | +0.96 (+0.40 to +1.56) | +2.43 (+1.29 to +3.62) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.36 (+4.20 to +6.48) | +2.28 (+1.48 to +3.12) | +0.80 (+0.28 to +1.32) | +2.28 (+1.24 to +3.37) |
| correct every question (no gate) | T1_k0_cot_multi | +6.88 (+5.68 to +8.12) | +1.76 (+1.00 to +2.56) | +1.08 (+0.48 to +1.68) | +3.62 (+2.38 to +5.01) |
| best detector at a 20% budget | T1_k0_cot_multi | +5.40 (+4.36 to +6.48) | +1.68 (+1.04 to +2.36) | +0.72 (+0.24 to +1.20) | +2.28 (+1.29 to +3.37) |
| best detector at a 30% budget | T1_k0_cot_multi | +6.84 (+5.64 to +8.08) | +1.48 (+0.76 to +2.20) | +0.80 (+0.28 to +1.32) | +3.57 (+2.33 to +4.96) |
| best detector at a 50% budget | T1_k0_cot_multi | +6.72 (+5.52 to +7.96) | +1.80 (+1.04 to +2.56) | +1.08 (+0.48 to +1.68) | +3.77 (+2.58 to +5.11) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +6.84 (+5.64 to +8.08) | +1.48 (+0.76 to +2.20) | +0.80 (+0.28 to +1.32) | +3.57 (+2.33 to +4.96) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +6.72 (+5.52 to +7.92) | +2.60 (+1.76 to +3.48) | +1.08 (+0.48 to +1.72) | +3.62 (+2.33 to +5.01) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +5.36 (+4.28 to +6.40) | +2.36 (+1.60 to +3.16) | +0.68 (+0.20 to +1.16) | +2.43 (+1.44 to +3.52) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +6.16 (+5.00 to +7.28) | +2.48 (+1.64 to +3.36) | +0.92 (+0.36 to +1.48) | +2.93 (+1.78 to +4.16) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +6.60 (+5.40 to +7.84) | +2.60 (+1.76 to +3.48) | +1.08 (+0.48 to +1.72) | +3.77 (+2.53 to +5.11) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +6.16 (+5.00 to +7.28) | +2.48 (+1.64 to +3.36) | +0.92 (+0.36 to +1.48) | +2.93 (+1.78 to +4.16) |
| correct every question (no gate) | T1_k0_esc8b | +6.88 (+5.52 to +8.28) | +4.04 (+2.88 to +5.16) | +4.36 (+3.36 to +5.36) | +7.49 (+6.05 to +8.93) |
| best detector at a 20% budget | T1_k0_esc8b | +5.00 (+3.92 to +6.12) | +2.80 (+1.88 to +3.80) | +2.36 (+1.64 to +3.12) | +4.61 (+3.57 to +5.75) |
| best detector at a 30% budget | T1_k0_esc8b | +6.44 (+5.16 to +7.72) | +3.28 (+2.24 to +4.28) | +2.80 (+2.00 to +3.56) | +6.59 (+5.30 to +7.98) |
| best detector at a 50% budget | T1_k0_esc8b | +6.64 (+5.32 to +7.92) | +4.04 (+2.88 to +5.16) | +4.24 (+3.28 to +5.20) | +7.39 (+6.00 to +8.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_esc8b | +6.56 (+5.32 to +7.84) | +4.00 (+2.84 to +5.08) | +4.00 (+3.08 to +4.96) | +7.14 (+5.85 to +8.48) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc8b | +7.44 (+6.04 to +8.84) | +4.28 (+3.12 to +5.44) | +3.96 (+3.00 to +4.88) | +7.59 (+6.15 to +9.02) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc8b | +5.40 (+4.32 to +6.52) | +3.12 (+2.16 to +4.16) | +2.20 (+1.52 to +2.88) | +4.66 (+3.57 to +5.80) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc8b | +6.76 (+5.44 to +8.08) | +3.52 (+2.48 to +4.60) | +2.52 (+1.72 to +3.24) | +6.59 (+5.30 to +7.98) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc8b | +7.04 (+5.68 to +8.40) | +4.28 (+3.16 to +5.40) | +4.00 (+3.04 to +4.88) | +7.44 (+6.00 to +8.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc8b | +6.84 (+5.56 to +8.16) | +4.20 (+3.08 to +5.32) | +3.72 (+2.76 to +4.60) | +7.14 (+5.85 to +8.48) |
| correct every question (no gate) | T1_k0_ground_cot_multi_esc | +6.08 (+5.00 to +7.16) | +4.08 (+3.08 to +5.12) | +2.84 (+2.04 to +3.56) | +4.41 (+3.22 to +5.60) |
| best detector at a 20% budget | T1_k0_ground_cot_multi_esc | +4.96 (+4.00 to +5.96) | +3.44 (+2.60 to +4.36) | +1.64 (+1.08 to +2.20) | +3.02 (+2.03 to +4.07) |
| best detector at a 30% budget | T1_k0_ground_cot_multi_esc | +5.72 (+4.68 to +6.76) | +3.88 (+2.96 to +4.84) | +2.28 (+1.60 to +2.92) | +3.82 (+2.78 to +4.96) |
| best detector at a 50% budget | T1_k0_ground_cot_multi_esc | +6.04 (+4.96 to +7.12) | +4.12 (+3.12 to +5.16) | +2.84 (+2.04 to +3.56) | +4.41 (+3.27 to +5.60) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi_esc | +5.72 (+4.68 to +6.76) | +3.88 (+2.96 to +4.84) | +2.28 (+1.60 to +2.92) | +3.82 (+2.78 to +4.96) |
| correct every question (no gate) | T1_k0_everything_multi_esc | +5.96 (+4.84 to +7.12) | +4.16 (+3.16 to +5.16) | +2.72 (+1.96 to +3.44) | +4.41 (+3.27 to +5.60) |
| best detector at a 20% budget | T1_k0_everything_multi_esc | +5.24 (+4.16 to +6.28) | +3.08 (+2.28 to +3.88) | +1.24 (+0.72 to +1.76) | +4.02 (+2.93 to +5.11) |
| best detector at a 30% budget | T1_k0_everything_multi_esc | +5.60 (+4.56 to +6.64) | +3.96 (+3.04 to +4.92) | +2.20 (+1.52 to +2.84) | +3.82 (+2.78 to +4.96) |
| best detector at a 50% budget | T1_k0_everything_multi_esc | +5.88 (+4.76 to +7.00) | +4.20 (+3.20 to +5.20) | +2.72 (+1.96 to +3.44) | +4.41 (+3.32 to +5.60) |
| smallest budget keeping 90% of the full validation gain | T1_k0_everything_multi_esc | +5.60 (+4.56 to +6.64) | +3.96 (+3.04 to +4.92) | +2.20 (+1.52 to +2.84) | +3.82 (+2.78 to +4.96) |
| best under 2 passes per question (any configuration) | T1_k0_esc8b | +4.44 (+3.40 to +5.48) | +2.40 (+1.56 to +3.24) | +1.56 (+0.96 to +2.16) | +3.72 (+2.73 to +4.81) |
| best under 3 passes per question (any configuration) | T1_k0_esc8b | +6.36 (+5.12 to +7.64) | +3.92 (+2.76 to +5.00) | +3.60 (+2.68 to +4.52) | +6.64 (+5.45 to +7.98) |
| best under 5 passes per question (any configuration) | T1_k0_esc8b | +6.80 (+5.48 to +8.16) | +4.08 (+2.92 to +5.20) | +4.40 (+3.40 to +5.40) | +7.54 (+6.15 to +8.97) |
