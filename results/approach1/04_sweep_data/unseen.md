# Transfer to unseen datasets (frozen pipeline)

Questions: {'ai2d': 3088, 'infovqa': 2801}. Ranker variant: scalars_nods (no dataset input). Configurations chosen on in-domain validation: {'multi': 'T1_k0_ground_cot_multi', 'multi_esc': 'T1_k5_esc8b', 'cot8b_nojudge': 'T1_k0_ground_cot_multi_esc8b_cot8b'}. Gate thresholds from in-domain validation.

| Setting | Accuracy | Gain over greedy (95% interval) | Fixed / broken | ai2d | infovqa |
|---|---:|---|---|---|---|
| T1_k0_ground_cot_multi, every question | 68.57% | +3.85 (+3.06 to +4.62) | +399 / -172 | +1.78 (+0.81 to +2.78) | +6.14 (+4.93 to +7.36) |
| T1_k0_ground_cot_multi, gate 30% (corrected 49.0%) | 68.40% | +3.68 (+2.92 to +4.45) | +377 / -160 | +1.75 (+0.81 to +2.72) | +5.82 (+4.68 to +6.96) |
| T1_k0_ground_cot_multi, gate 50% (corrected 70.1%) | 68.59% | +3.87 (+3.07 to +4.64) | +397 / -169 | +1.81 (+0.84 to +2.82) | +6.14 (+4.93 to +7.35) |
| T1_k0_cot_multi, every question | 68.31% | +3.60 (+2.77 to +4.38) | +385 / -173 | +1.75 (+0.78 to +2.75) | +5.64 (+4.39 to +6.89) |
| T1_k0_cot_multi, gate 30% (corrected 49.0%) | 68.13% | +3.41 (+2.63 to +4.18) | +361 / -160 | +1.65 (+0.68 to +2.66) | +5.36 (+4.18 to +6.50) |
| T1_k0_cot_multi, gate 50% (corrected 70.1%) | 68.33% | +3.62 (+2.80 to +4.40) | +383 / -170 | +1.78 (+0.81 to +2.78) | +5.64 (+4.43 to +6.82) |
| T1_k0_everything_multi, every question | 68.16% | +3.45 (+2.65 to +4.21) | +387 / -184 | +1.36 (+0.32 to +2.36) | +5.75 (+4.53 to +6.96) |
| T1_k0_everything_multi, gate 30% (corrected 49.0%) | 67.96% | +3.24 (+2.46 to +3.97) | +365 / -174 | +1.26 (+0.26 to +2.24) | +5.43 (+4.28 to +6.57) |
| T1_k0_everything_multi, gate 50% (corrected 70.1%) | 68.18% | +3.46 (+2.65 to +4.21) | +386 / -182 | +1.36 (+0.32 to +2.36) | +5.78 (+4.57 to +7.00) |
| T1_k5_esc8b, every question | 75.41% | +10.70 (+9.63 to +11.79) | +866 / -236 | +6.67 (+5.31 to +7.97) | +15.14 (+13.53 to +16.82) |
| T1_k5_esc8b, gate 30% (corrected 49.0%) | 74.16% | +9.44 (+8.46 to +10.43) | +752 / -196 | +6.25 (+4.95 to +7.48) | +12.96 (+11.50 to +14.49) |
| T1_k5_esc8b, gate 50% (corrected 70.1%) | 75.36% | +10.65 (+9.58 to +11.70) | +854 / -227 | +6.77 (+5.41 to +8.06) | +14.92 (+13.35 to +16.60) |
| T1_k0_esc8b, every question | 75.55% | +10.83 (+9.78 to +11.89) | +875 / -237 | +6.74 (+5.37 to +8.06) | +15.35 (+13.71 to +17.03) |
| T1_k0_esc8b, gate 30% (corrected 49.0%) | 74.31% | +9.59 (+8.61 to +10.58) | +755 / -190 | +6.31 (+5.05 to +7.55) | +13.21 (+11.71 to +14.74) |
| T1_k0_esc8b, gate 50% (corrected 70.1%) | 75.51% | +10.80 (+9.75 to +11.85) | +861 / -225 | +6.87 (+5.47 to +8.16) | +15.14 (+13.50 to +16.82) |
| T1_k0_ground_cot_multi_esc8b, every question | 75.07% | +10.36 (+9.36 to +11.34) | +774 / -164 | +6.90 (+5.67 to +8.06) | +14.17 (+12.67 to +15.71) |
| T1_k0_ground_cot_multi_esc8b, gate 30% (corrected 49.0%) | 73.83% | +9.12 (+8.18 to +10.02) | +682 / -145 | +5.93 (+4.76 to +7.06) | +12.64 (+11.21 to +14.03) |
| T1_k0_ground_cot_multi_esc8b, gate 50% (corrected 70.1%) | 74.87% | +10.15 (+9.17 to +11.12) | +759 / -161 | +6.80 (+5.60 to +7.97) | +13.85 (+12.35 to +15.39) |
| T1_k0_ground_cot_multi_esc, every question | 75.07% | +10.36 (+9.36 to +11.34) | +774 / -164 | +6.90 (+5.67 to +8.06) | +14.17 (+12.67 to +15.71) |
| T1_k0_ground_cot_multi_esc, gate 30% (corrected 49.0%) | 73.83% | +9.12 (+8.18 to +10.02) | +682 / -145 | +5.93 (+4.76 to +7.06) | +12.64 (+11.21 to +14.03) |
| T1_k0_ground_cot_multi_esc, gate 50% (corrected 70.1%) | 74.87% | +10.15 (+9.17 to +11.12) | +759 / -161 | +6.80 (+5.60 to +7.97) | +13.85 (+12.35 to +15.39) |
| T1_k0_everything_multi_esc, every question | 74.85% | +10.14 (+9.14 to +11.14) | +777 / -180 | +6.70 (+5.47 to +7.90) | +13.92 (+12.39 to +15.42) |
| T1_k0_everything_multi_esc, gate 30% (corrected 49.0%) | 73.63% | +8.91 (+7.95 to +9.83) | +686 / -161 | +5.73 (+4.57 to +6.90) | +12.42 (+10.96 to +13.89) |
| T1_k0_everything_multi_esc, gate 50% (corrected 70.1%) | 74.61% | +9.90 (+8.90 to +10.90) | +760 / -177 | +6.54 (+5.34 to +7.74) | +13.60 (+12.10 to +15.10) |
| T1_k0_ground_cot_multi_esc8b_cot8b (8B reasoning, no judge), every question | 76.99% | +12.28 (+11.22 to +13.28) | +911 / -188 | +7.77 (+6.44 to +9.03) | +17.24 (+15.60 to +18.89) |
| T1_k0_ground_cot_multi_esc8b_cot8b (8B reasoning, no judge), gate 30% (corrected 49.0%) | 75.09% | +10.38 (+9.36 to +11.34) | +779 / -168 | +6.51 (+5.31 to +7.68) | +14.64 (+13.00 to +16.17) |
| T1_k0_ground_cot_multi_esc8b_cot8b (8B reasoning, no judge), gate 50% (corrected 70.1%) | 76.69% | +11.97 (+10.90 to +12.96) | +886 / -181 | +7.58 (+6.31 to +8.84) | +16.82 (+15.14 to +18.42) |
| T1_k0_cot8b (8B reasoning, no judge), every question | 76.70% | +11.99 (+10.87 to +13.04) | +968 / -262 | +7.38 (+6.02 to +8.71) | +17.07 (+15.24 to +18.74) |
| T1_k0_cot8b (8B reasoning, no judge), gate 30% (corrected 49.0%) | 75.14% | +10.43 (+9.39 to +11.41) | +816 / -202 | +6.51 (+5.28 to +7.77) | +14.74 (+13.07 to +16.32) |
| T1_k0_cot8b (8B reasoning, no judge), gate 50% (corrected 70.1%) | 76.48% | +11.77 (+10.68 to +12.82) | +939 / -246 | +7.32 (+5.99 to +8.65) | +16.67 (+14.92 to +18.39) |
| T1_k0_ground_cot_multi_cot8b (8B reasoning, no judge), every question | 76.74% | +12.02 (+10.97 to +13.06) | +944 / -236 | +7.45 (+6.12 to +8.78) | +17.07 (+15.32 to +18.71) |
| T1_k0_ground_cot_multi_cot8b (8B reasoning, no judge), gate 30% (corrected 49.0%) | 74.90% | +10.19 (+9.15 to +11.16) | +798 / -198 | +6.19 (+4.95 to +7.45) | +14.60 (+12.89 to +16.10) |
| T1_k0_ground_cot_multi_cot8b (8B reasoning, no judge), gate 50% (corrected 70.1%) | 76.45% | +11.73 (+10.68 to +12.75) | +918 / -227 | +7.25 (+5.96 to +8.55) | +16.67 (+14.92 to +18.28) |
| 8b on gated questions, gate 30% (corrected 49.0%) | 74.60% | +9.88 (+8.86 to +10.92) | +789 / -207 | +6.96 (+5.67 to +8.23) | +13.10 (+11.64 to +14.64) |
| 8b on gated questions, gate 50% (corrected 70.1%) | 76.18% | +11.46 (+10.37 to +12.58) | +933 / -258 | +7.97 (+6.57 to +9.29) | +15.32 (+13.67 to +16.99) |
| idefics3_8b on gated questions, gate 30% (corrected 49.0%) | 58.62% | -6.10 (-7.10 to -5.08) | +285 / -644 | +0.23 (-1.20 to +1.55) | -13.07 (-14.57 to -11.67) |
| idefics3_8b on gated questions, gate 50% (corrected 70.1%) | 54.27% | -10.44 (-11.68 to -9.27) | +357 / -972 | -0.65 (-2.27 to +0.84) | -21.24 (-22.99 to -19.60) |
| internvl3_8b on gated questions, gate 30% (corrected 49.0%) | 73.37% | +8.66 (+7.59 to +9.70) | +772 / -262 | +6.83 (+5.51 to +8.19) | +10.67 (+9.03 to +12.25) |
| internvl3_8b on gated questions, gate 50% (corrected 70.1%) | 74.85% | +10.14 (+8.95 to +11.29) | +928 / -331 | +7.71 (+6.25 to +9.23) | +12.82 (+11.07 to +14.57) |
| baseline: greedy | 64.71% | +0.00 (+0.00 to +0.00) | +0 / -0 | +0.00 (+0.00 to +0.00) | +0.00 (+0.00 to +0.00) |
| baseline: self_consistency_5 | 64.48% | -0.24 (-0.61 to +0.14) | +57 / -71 | -0.58 (-1.10 to -0.10) | +0.14 (-0.43 to +0.71) |
| baseline: self_consistency_10 | 64.70% | -0.02 (-0.37 to +0.36) | +59 / -60 | -0.29 (-0.74 to +0.19) | +0.29 (-0.32 to +0.89) |
| baseline: reasoning_vote | 67.18% | +2.46 (+1.46 to +3.40) | +514 / -369 | +1.13 (-0.23 to +2.43) | +3.93 (+2.50 to +5.32) |
| baseline: reasoning_greedy_vote | 68.14% | +3.43 (+2.67 to +4.19) | +377 / -175 | +2.46 (+1.46 to +3.47) | +4.50 (+3.32 to +5.71) |
| baseline: set_vote | 69.23% | +4.52 (+3.77 to +5.30) | +404 / -138 | +2.43 (+1.46 to +3.37) | +6.82 (+5.53 to +8.07) |
| baseline: set_judge_best | 62.40% | -2.31 (-3.28 to -1.43) | +336 / -472 | -4.15 (-5.47 to -2.85) | -0.29 (-1.64 to +1.07) |
| baseline: set_token_best | 64.66% | -0.05 (-0.34 to +0.24) | +37 / -40 | -0.03 (-0.32 to +0.29) | -0.07 (-0.57 to +0.43) |
| baseline: model_8b | 76.52% | +11.80 (+10.68 to +12.94) | +986 / -291 | +8.26 (+6.74 to +9.68) | +15.71 (+13.99 to +17.42) |
| baseline: model_idefics3_8b | 44.57% | -20.14 (-21.53 to -18.80) | +379 / -1565 | -1.62 (-3.34 to -0.06) | -40.56 (-42.73 to -38.63) |
| baseline: model_internvl3_8b | 74.72% | +10.00 (+8.76 to +11.19) | +973 / -384 | +7.90 (+6.35 to +9.49) | +12.32 (+10.42 to +14.14) |
