# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_drop_judge_models.pt`. Greedy on test: 82.78%. Passes: model passes per question (generation calls plus 1 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only. Ranker inputs zeroed: judge.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 85.75% vs 85.75%, T1_k0_ground_cot 85.81% vs 85.81%, T1_k0_cot 85.42% vs 85.42%, T1_k0_zoom_cot 85.70% vs 85.70%, T1_k0_cot_multi 86.11% vs 86.11%, T1_k0_ground_cot_multi 86.36% vs 86.36%

| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 20.07 | 16.00 | 85.75% | +2.97 (+2.45 to +3.49) | +462 / -179 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 19.4% | 5.16 | 3.92 | 85.26% | +2.48 (+2.05 to +2.92) | +343 / -107 | +1.80 (+1.38 to +2.20) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 29.6% | 7.14 | 5.43 | 85.62% | +2.84 (+2.35 to +3.31) | +411 / -141 | +1.95 (+1.53 to +2.37) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 49.7% | 10.88 | 8.46 | 85.81% | +3.04 (+2.53 to +3.55) | +459 / -170 | +1.68 (+1.29 to +2.06) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 25% / 24.4% | 6.13 | 4.66 | 85.49% | +2.71 (+2.24 to +3.16) | +379 / -121 | +1.88 (+1.47 to +2.28) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 6.76 | 5.00 | 85.81% | +3.04 (+2.54 to +3.55) | +443 / -154 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | after_head | 20% / 19.2% | 3.06 | 1.77 | 85.48% | +2.70 (+2.25 to +3.13) | +357 / -100 | +2.06 (+1.66 to +2.47) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 85.59% | +2.82 (+2.34 to +3.28) | +386 / -118 | +1.93 (+1.53 to +2.35) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 49.7% | 4.05 | 2.99 | 85.84% | +3.06 (+2.56 to +3.56) | +438 / -147 | +1.66 (+1.30 to +2.01) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 30% / 29.6% | 2.90 | 2.18 | 85.59% | +2.82 (+2.34 to +3.28) | +386 / -118 | +1.93 (+1.53 to +2.35) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 3.26 | 2.00 | 85.42% | +2.64 (+2.10 to +3.15) | +439 / -188 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 19.4% | 1.51 | 1.19 | 84.91% | +2.13 (+1.72 to +2.53) | +300 / -97 | +1.52 (+1.13 to +1.89) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 85.21% | +2.43 (+1.93 to +2.87) | +364 / -133 | +1.65 (+1.24 to +2.04) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 49.7% | 2.21 | 1.50 | 85.45% | +2.67 (+2.12 to +3.17) | +430 / -176 | +1.46 (+1.07 to +1.83) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 30% / 29.6% | 1.76 | 1.30 | 85.21% | +2.43 (+1.93 to +2.87) | +364 / -133 | +1.65 (+1.24 to +2.04) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 9.29 | 6.00 | 85.70% | +2.92 (+2.41 to +3.43) | +438 / -160 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 19.4% | 2.75 | 1.97 | 85.11% | +2.33 (+1.91 to +2.74) | +318 / -96 | +1.59 (+1.20 to +1.96) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 85.54% | +2.76 (+2.27 to +3.20) | +385 / -122 | +1.86 (+1.44 to +2.25) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 49.7% | 5.26 | 3.49 | 85.74% | +2.96 (+2.48 to +3.46) | +434 / -152 | +1.61 (+1.23 to +1.95) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 30% / 29.6% | 3.62 | 2.48 | 85.54% | +2.76 (+2.27 to +3.20) | +385 / -122 | +1.86 (+1.44 to +2.25) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 6.55 | 5.00 | 86.11% | +3.33 (+2.83 to +3.84) | +456 / -139 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 19.4% | 2.25 | 1.78 | 85.30% | +2.52 (+2.11 to +2.94) | +323 / -83 | +1.77 (+1.38 to +2.13) |
| best detector at a 30% budget | T1_k0_cot_multi | after_head | 30% / 28.2% | 3.47 | 2.13 | 85.89% | +3.11 (+2.63 to +3.60) | +424 / -128 | +2.08 (+1.68 to +2.49) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 49.7% | 3.96 | 2.99 | 86.11% | +3.33 (+2.84 to +3.81) | +447 / -130 | +1.67 (+1.30 to +2.02) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 35% / 34.9% | 3.14 | 2.39 | 85.97% | +3.19 (+2.72 to +3.66) | +416 / -112 | +1.96 (+1.56 to +2.35) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 10.01 | 8.00 | 86.36% | +3.58 (+3.08 to +4.09) | +472 / -131 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 19.4% | 3.01 | 2.36 | 85.52% | +2.74 (+2.33 to +3.18) | +343 / -82 | +1.93 (+1.54 to +2.31) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 86.01% | +3.24 (+2.76 to +3.68) | +412 / -104 | +2.13 (+1.73 to +2.53) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 49.7% | 5.76 | 4.48 | 86.38% | +3.60 (+3.10 to +4.10) | +468 / -125 | +1.79 (+1.44 to +2.13) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 30% / 29.6% | 3.96 | 3.07 | 86.01% | +3.24 (+2.76 to +3.68) | +412 / -104 | +2.13 (+1.73 to +2.53) |
| best under 2 passes per question (any configuration) | T1_k0_cot | token | 40% / 39.8% | 1.99 | 1.40 | 85.51% | +2.73 (+2.22 to +3.23) | +412 / -152 | +1.73 (+1.32 to +2.13) |
| best under 3 passes per question (any configuration) | T1_k0_cot_multi | token | 30% / 29.6% | 2.84 | 2.18 | 85.73% | +2.95 (+2.48 to +3.39) | +389 / -108 | +1.92 (+1.53 to +2.31) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot_multi | token | 40% / 39.8% | 4.89 | 3.79 | 86.36% | +3.58 (+3.09 to +4.06) | +452 / -111 | +2.08 (+1.71 to +2.48) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +5.36 (+4.12 to +6.56) | +2.72 (+1.72 to +3.72) | +1.16 (+0.48 to +1.80) | +2.58 (+1.29 to +3.92) |
| best detector at a 20% budget | T1_k5_everything | +4.68 (+3.60 to +5.80) | +2.44 (+1.64 to +3.32) | +0.64 (+0.16 to +1.12) | +2.08 (+1.09 to +3.12) |
| best detector at a 30% budget | T1_k5_everything | +5.16 (+4.00 to +6.32) | +2.68 (+1.80 to +3.64) | +0.88 (+0.28 to +1.48) | +2.58 (+1.49 to +3.77) |
| best detector at a 50% budget | T1_k5_everything | +5.28 (+4.04 to +6.48) | +2.72 (+1.72 to +3.72) | +1.20 (+0.52 to +1.84) | +2.93 (+1.69 to +4.17) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +5.00 (+3.84 to +6.12) | +2.64 (+1.80 to +3.56) | +0.88 (+0.32 to +1.40) | +2.23 (+1.14 to +3.32) |
| correct every question (no gate) | T1_k0_ground_cot | +6.28 (+5.08 to +7.52) | +2.20 (+1.36 to +3.08) | +1.00 (+0.36 to +1.60) | +2.58 (+1.39 to +3.87) |
| best detector at a 20% budget | T1_k0_ground_cot | +6.00 (+4.84 to +7.16) | +1.56 (+0.88 to +2.28) | +0.56 (+0.16 to +0.96) | +2.68 (+1.64 to +3.77) |
| best detector at a 30% budget | T1_k0_ground_cot | +5.76 (+4.60 to +6.92) | +2.24 (+1.48 to +3.08) | +0.72 (+0.16 to +1.28) | +2.48 (+1.44 to +3.57) |
| best detector at a 50% budget | T1_k0_ground_cot | +6.16 (+4.96 to +7.36) | +2.20 (+1.36 to +3.08) | +1.00 (+0.36 to +1.60) | +2.83 (+1.69 to +4.07) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +5.76 (+4.60 to +6.92) | +2.24 (+1.48 to +3.08) | +0.72 (+0.16 to +1.28) | +2.48 (+1.44 to +3.57) |
| correct every question (no gate) | T1_k0_cot | +6.60 (+5.32 to +7.92) | +0.76 (-0.08 to +1.60) | +0.52 (-0.20 to +1.16) | +2.68 (+1.44 to +3.92) |
| best detector at a 20% budget | T1_k0_cot | +5.16 (+4.12 to +6.24) | +1.08 (+0.40 to +1.80) | +0.36 (-0.12 to +0.84) | +1.88 (+0.99 to +2.83) |
| best detector at a 30% budget | T1_k0_cot | +6.00 (+4.84 to +7.20) | +0.96 (+0.20 to +1.72) | +0.40 (-0.20 to +1.00) | +2.33 (+1.34 to +3.47) |
| best detector at a 50% budget | T1_k0_cot | +6.36 (+5.08 to +7.64) | +0.88 (+0.08 to +1.72) | +0.52 (-0.20 to +1.16) | +2.97 (+1.78 to +4.21) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +6.00 (+4.84 to +7.20) | +0.96 (+0.20 to +1.72) | +0.40 (-0.20 to +1.00) | +2.33 (+1.34 to +3.47) |
| correct every question (no gate) | T1_k0_zoom_cot | +6.08 (+4.84 to +7.32) | +2.16 (+1.28 to +3.00) | +1.08 (+0.44 to +1.72) | +2.23 (+1.04 to +3.47) |
| best detector at a 20% budget | T1_k0_zoom_cot | +4.96 (+3.88 to +6.04) | +1.92 (+1.20 to +2.68) | +0.48 (+0.00 to +0.96) | +1.88 (+0.94 to +2.83) |
| best detector at a 30% budget | T1_k0_zoom_cot | +5.64 (+4.48 to +6.80) | +2.20 (+1.40 to +3.00) | +0.80 (+0.24 to +1.36) | +2.33 (+1.29 to +3.47) |
| best detector at a 50% budget | T1_k0_zoom_cot | +5.96 (+4.72 to +7.20) | +2.16 (+1.28 to +3.00) | +1.08 (+0.44 to +1.72) | +2.58 (+1.44 to +3.82) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +5.64 (+4.48 to +6.80) | +2.20 (+1.40 to +3.00) | +0.80 (+0.24 to +1.36) | +2.33 (+1.29 to +3.47) |
| correct every question (no gate) | T1_k0_cot_multi | +7.00 (+5.76 to +8.28) | +1.80 (+1.04 to +2.60) | +1.12 (+0.52 to +1.72) | +3.42 (+2.18 to +4.81) |
| best detector at a 20% budget | T1_k0_cot_multi | +5.44 (+4.40 to +6.52) | +1.68 (+1.04 to +2.36) | +0.76 (+0.28 to +1.24) | +2.13 (+1.19 to +3.17) |
| best detector at a 30% budget | T1_k0_cot_multi | +6.88 (+5.68 to +8.08) | +1.40 (+0.72 to +2.12) | +0.84 (+0.32 to +1.36) | +3.37 (+2.13 to +4.76) |
| best detector at a 50% budget | T1_k0_cot_multi | +6.84 (+5.60 to +8.12) | +1.84 (+1.08 to +2.64) | +1.12 (+0.52 to +1.72) | +3.57 (+2.33 to +4.91) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +6.76 (+5.56 to +7.96) | +1.84 (+1.12 to +2.60) | +1.00 (+0.40 to +1.60) | +3.17 (+2.03 to +4.41) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +7.00 (+5.76 to +8.24) | +2.72 (+1.88 to +3.60) | +1.24 (+0.60 to +1.84) | +3.32 (+2.08 to +4.66) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +5.52 (+4.44 to +6.60) | +2.44 (+1.68 to +3.20) | +0.68 (+0.24 to +1.12) | +2.23 (+1.24 to +3.32) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +6.44 (+5.28 to +7.56) | +2.60 (+1.80 to +3.44) | +1.04 (+0.48 to +1.60) | +2.78 (+1.64 to +4.02) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +6.92 (+5.68 to +8.16) | +2.72 (+1.88 to +3.60) | +1.24 (+0.60 to +1.84) | +3.52 (+2.28 to +4.81) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +6.44 (+5.28 to +7.56) | +2.60 (+1.80 to +3.44) | +1.04 (+0.48 to +1.60) | +2.78 (+1.64 to +4.02) |
| best under 2 passes per question (any configuration) | T1_k0_cot | +6.40 (+5.16 to +7.64) | +1.00 (+0.20 to +1.84) | +0.52 (-0.16 to +1.12) | +3.07 (+1.98 to +4.26) |
| best under 3 passes per question (any configuration) | T1_k0_cot_multi | +6.44 (+5.28 to +7.60) | +1.76 (+1.08 to +2.48) | +0.92 (+0.32 to +1.48) | +2.63 (+1.54 to +3.87) |
| best under 5 passes per question (any configuration) | T1_k0_ground_cot_multi | +6.88 (+5.68 to +8.08) | +2.72 (+1.92 to +3.56) | +1.20 (+0.56 to +1.80) | +3.52 (+2.38 to +4.81) |
