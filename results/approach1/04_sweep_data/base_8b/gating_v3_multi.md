# Gating the v3 correction step

Thresholds on the detector score come from validation; on test each question is gated on its own score. Ranker: v3 scalars_ds from `candidate_ranker_v3_multi_models.pt`. Greedy on test: 87.87%. Passes: model passes per question (generation calls plus 2 scoring pass(es) per distinct candidate of a corrected question); calls: generation calls only.

Reproduction check (ungated test accuracy against the training run's report): T1_k5_everything 88.18% vs 88.18%, T1_k0_ground_cot 88.28% vs 88.28%, T1_k0_cot 88.04% vs 88.04%, T1_k0_zoom_cot 88.19% vs 88.19%, T1_k0_cot_multi 88.03% vs 88.03%, T1_k0_ground_cot_multi 88.22% vs 88.22%



| Setting | Configuration | Detector | Budget (val) / share corrected (test) | Passes | Calls | Seconds | Test accuracy | Gain (95% interval) | Fixed / broken | Over random gate |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| correct every question (no gate) | T1_k5_everything | token | 100% / 100.0% | 23.62 | 16.00 | 0.00 | 88.18% | +0.30 (+0.12 to +0.49) | +57 / -28 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k5_everything | token | 20% / 20.4% | 6.33 | 4.06 | 0.00 | 88.15% | +0.27 (+0.08 to +0.46) | +54 / -28 | +0.18 (+0.01 to +0.34) |
| best detector at a 30% budget | T1_k5_everything | token | 30% / 30.9% | 8.75 | 5.64 | 0.00 | 88.18% | +0.30 (+0.12 to +0.49) | +57 / -28 | +0.21 (+0.05 to +0.37) |
| best detector at a 50% budget | T1_k5_everything | token | 50% / 52.5% | 13.55 | 8.87 | 0.00 | 88.18% | +0.30 (+0.12 to +0.49) | +57 / -28 | +0.17 (+0.04 to +0.30) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | token | 15% / 15.2% | 5.07 | 3.27 | 0.00 | 88.11% | +0.23 (+0.05 to +0.41) | +48 / -26 | +0.13 (-0.04 to +0.29) |
| correct every question (no gate) | T1_k0_ground_cot | token | 100% / 100.0% | 8.40 | 5.00 | 0.00 | 88.28% | +0.41 (+0.21 to +0.61) | +65 / -26 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot | token | 20% / 20.4% | 2.79 | 1.82 | 0.00 | 88.24% | +0.37 (+0.18 to +0.56) | +59 / -24 | +0.24 (+0.06 to +0.40) |
| best detector at a 30% budget | T1_k0_ground_cot | token | 30% / 30.9% | 3.59 | 2.24 | 0.00 | 88.27% | +0.40 (+0.20 to +0.60) | +64 / -26 | +0.25 (+0.08 to +0.41) |
| best detector at a 50% budget | T1_k0_ground_cot | token | 50% / 52.5% | 5.15 | 3.10 | 0.00 | 88.28% | +0.41 (+0.21 to +0.61) | +65 / -26 | +0.22 (+0.07 to +0.36) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | token | 15% / 15.2% | 2.36 | 1.61 | 0.00 | 88.17% | +0.29 (+0.12 to +0.47) | +51 / -23 | +0.19 (+0.03 to +0.35) |
| correct every question (no gate) | T1_k0_cot | token | 100% / 100.0% | 4.59 | 2.00 | 0.00 | 88.04% | +0.17 (+0.02 to +0.30) | +33 / -17 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot | token | 20% / 20.4% | 1.86 | 1.20 | 0.00 | 88.01% | +0.14 (+0.00 to +0.27) | +30 / -17 | +0.07 (-0.05 to +0.19) |
| best detector at a 30% budget | T1_k0_cot | token | 30% / 30.9% | 2.26 | 1.31 | 0.00 | 88.03% | +0.16 (+0.01 to +0.29) | +32 / -17 | +0.09 (-0.02 to +0.20) |
| best detector at a 50% budget | T1_k0_cot | token | 50% / 52.5% | 3.03 | 1.52 | 0.00 | 88.04% | +0.17 (+0.02 to +0.30) | +33 / -17 | +0.07 (-0.03 to +0.18) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | token | 10% / 10.0% | 1.44 | 1.10 | 0.00 | 87.96% | +0.08 (-0.03 to +0.20) | +21 / -13 | +0.00 (-0.12 to +0.12) |
| correct every question (no gate) | T1_k0_zoom_cot | token | 100% / 100.0% | 12.62 | 6.00 | 0.00 | 88.19% | +0.32 (+0.13 to +0.49) | +55 / -25 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_zoom_cot | token | 20% / 20.4% | 3.70 | 2.02 | 0.00 | 88.17% | +0.29 (+0.12 to +0.46) | +52 / -24 | +0.22 (+0.05 to +0.37) |
| best detector at a 30% budget | T1_k0_zoom_cot | token | 30% / 30.9% | 4.99 | 2.55 | 0.00 | 88.18% | +0.30 (+0.12 to +0.48) | +54 / -25 | +0.22 (+0.07 to +0.36) |
| best detector at a 50% budget | T1_k0_zoom_cot | token | 50% / 52.5% | 7.51 | 3.62 | 0.00 | 88.19% | +0.32 (+0.13 to +0.49) | +55 / -25 | +0.17 (+0.04 to +0.29) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | token | 15% / 15.2% | 3.03 | 1.76 | 0.00 | 88.12% | +0.24 (+0.07 to +0.40) | +44 / -21 | +0.17 (+0.01 to +0.32) |
| correct every question (no gate) | T1_k0_cot_multi | token | 100% / 100.0% | 8.02 | 5.00 | 0.00 | 88.03% | +0.16 (+0.01 to +0.30) | +35 / -20 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_cot_multi | token | 20% / 20.4% | 2.68 | 1.82 | 0.00 | 88.00% | +0.13 (-0.02 to +0.26) | +31 / -19 | +0.06 (-0.06 to +0.19) |
| best detector at a 30% budget | T1_k0_cot_multi | token | 30% / 30.9% | 3.46 | 2.24 | 0.00 | 88.01% | +0.14 (-0.01 to +0.28) | +33 / -20 | +0.07 (-0.05 to +0.19) |
| best detector at a 50% budget | T1_k0_cot_multi | token | 50% / 52.5% | 4.95 | 3.10 | 0.00 | 88.03% | +0.16 (+0.01 to +0.30) | +35 / -20 | +0.06 (-0.05 to +0.17) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | token | 15% / 15.2% | 2.28 | 1.61 | 0.00 | 87.98% | +0.11 (-0.03 to +0.23) | +26 / -16 | +0.03 (-0.09 to +0.15) |
| correct every question (no gate) | T1_k0_ground_cot_multi | token | 100% / 100.0% | 11.74 | 8.00 | 0.00 | 88.22% | +0.35 (+0.15 to +0.54) | +59 / -26 | +0.00 (+0.00 to +0.00) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | token | 20% / 20.4% | 3.57 | 2.43 | 0.00 | 88.20% | +0.33 (+0.14 to +0.50) | +54 / -23 | +0.21 (+0.04 to +0.37) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | token | 30% / 30.9% | 4.74 | 3.16 | 0.00 | 88.21% | +0.34 (+0.15 to +0.53) | +58 / -26 | +0.20 (+0.04 to +0.36) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | token | 50% / 52.5% | 7.00 | 4.67 | 0.00 | 88.22% | +0.35 (+0.15 to +0.54) | +59 / -26 | +0.17 (+0.03 to +0.30) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | token | 15% / 15.2% | 2.96 | 2.06 | 0.00 | 88.16% | +0.28 (+0.11 to +0.46) | +49 / -22 | +0.20 (+0.04 to +0.36) |
| best under 2 passes per question (any configuration) | T1_k0_ground_cot | token | 10% / 10.0% | 1.93 | 1.40 | 0.00 | 88.11% | +0.23 (+0.07 to +0.39) | +40 / -18 | +0.15 (-0.01 to +0.29) |
| best under 3 passes per question (any configuration) | T1_k0_ground_cot_multi | token | 15% / 15.2% | 2.96 | 2.06 | 0.00 | 88.16% | +0.28 (+0.11 to +0.46) | +49 / -22 | +0.20 (+0.04 to +0.36) |
| best under 5 passes per question (any configuration) | T1_k0_zoom_cot | token | 15% / 15.2% | 3.03 | 1.76 | 0.00 | 88.12% | +0.24 (+0.07 to +0.40) | +44 / -21 | +0.17 (+0.01 to +0.32) |

Per dataset (gain over greedy on test):

| Setting | Configuration | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---|---|---|---|---|
| correct every question (no gate) | T1_k5_everything | +0.16 (-0.08 to +0.40) | +0.60 (+0.16 to +1.04) | +0.36 (+0.04 to +0.72) | +0.05 (-0.45 to +0.50) |
| best detector at a 20% budget | T1_k5_everything | +0.16 (-0.08 to +0.40) | +0.60 (+0.16 to +1.04) | +0.24 (-0.04 to +0.56) | +0.05 (-0.45 to +0.50) |
| best detector at a 30% budget | T1_k5_everything | +0.16 (-0.08 to +0.40) | +0.60 (+0.16 to +1.04) | +0.36 (+0.04 to +0.72) | +0.05 (-0.45 to +0.50) |
| best detector at a 50% budget | T1_k5_everything | +0.16 (-0.08 to +0.40) | +0.60 (+0.16 to +1.04) | +0.36 (+0.04 to +0.72) | +0.05 (-0.45 to +0.50) |
| smallest budget keeping 90% of the full validation gain | T1_k5_everything | +0.12 (-0.12 to +0.36) | +0.56 (+0.16 to +1.04) | +0.28 (+0.04 to +0.56) | -0.10 (-0.55 to +0.30) |
| correct every question (no gate) | T1_k0_ground_cot | +0.40 (+0.08 to +0.72) | +0.64 (+0.16 to +1.12) | +0.48 (+0.16 to +0.80) | +0.05 (-0.40 to +0.50) |
| best detector at a 20% budget | T1_k0_ground_cot | +0.32 (+0.08 to +0.60) | +0.68 (+0.20 to +1.16) | +0.36 (+0.08 to +0.64) | +0.05 (-0.40 to +0.50) |
| best detector at a 30% budget | T1_k0_ground_cot | +0.36 (+0.08 to +0.68) | +0.64 (+0.16 to +1.12) | +0.48 (+0.16 to +0.80) | +0.05 (-0.40 to +0.50) |
| best detector at a 50% budget | T1_k0_ground_cot | +0.40 (+0.08 to +0.72) | +0.64 (+0.16 to +1.12) | +0.48 (+0.16 to +0.80) | +0.05 (-0.40 to +0.50) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot | +0.28 (+0.04 to +0.56) | +0.52 (+0.08 to +0.96) | +0.36 (+0.12 to +0.64) | -0.05 (-0.50 to +0.40) |
| correct every question (no gate) | T1_k0_cot | +0.28 (+0.08 to +0.48) | +0.16 (-0.12 to +0.44) | +0.20 (-0.04 to +0.44) | +0.00 (-0.40 to +0.40) |
| best detector at a 20% budget | T1_k0_cot | +0.28 (+0.08 to +0.48) | +0.12 (-0.16 to +0.40) | +0.12 (-0.12 to +0.36) | +0.00 (-0.40 to +0.40) |
| best detector at a 30% budget | T1_k0_cot | +0.28 (+0.08 to +0.48) | +0.16 (-0.12 to +0.44) | +0.16 (-0.08 to +0.40) | +0.00 (-0.40 to +0.40) |
| best detector at a 50% budget | T1_k0_cot | +0.28 (+0.08 to +0.48) | +0.16 (-0.12 to +0.44) | +0.20 (-0.04 to +0.44) | +0.00 (-0.40 to +0.40) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot | +0.24 (+0.08 to +0.44) | +0.04 (-0.20 to +0.28) | +0.12 (+0.00 to +0.28) | -0.10 (-0.50 to +0.25) |
| correct every question (no gate) | T1_k0_zoom_cot | +0.48 (+0.24 to +0.76) | +0.48 (+0.08 to +0.92) | +0.24 (-0.04 to +0.56) | +0.00 (-0.45 to +0.50) |
| best detector at a 20% budget | T1_k0_zoom_cot | +0.44 (+0.20 to +0.72) | +0.44 (+0.04 to +0.84) | +0.24 (-0.04 to +0.56) | +0.00 (-0.45 to +0.50) |
| best detector at a 30% budget | T1_k0_zoom_cot | +0.44 (+0.20 to +0.72) | +0.48 (+0.08 to +0.92) | +0.24 (-0.04 to +0.56) | +0.00 (-0.45 to +0.50) |
| best detector at a 50% budget | T1_k0_zoom_cot | +0.48 (+0.24 to +0.76) | +0.48 (+0.08 to +0.92) | +0.24 (-0.04 to +0.56) | +0.00 (-0.45 to +0.50) |
| smallest budget keeping 90% of the full validation gain | T1_k0_zoom_cot | +0.40 (+0.16 to +0.68) | +0.36 (-0.04 to +0.76) | +0.28 (+0.08 to +0.52) | -0.15 (-0.59 to +0.30) |
| correct every question (no gate) | T1_k0_cot_multi | +0.36 (+0.16 to +0.60) | +0.12 (-0.20 to +0.44) | +0.16 (-0.08 to +0.44) | -0.05 (-0.45 to +0.35) |
| best detector at a 20% budget | T1_k0_cot_multi | +0.32 (+0.12 to +0.56) | +0.08 (-0.24 to +0.40) | +0.12 (-0.12 to +0.36) | -0.05 (-0.45 to +0.35) |
| best detector at a 30% budget | T1_k0_cot_multi | +0.32 (+0.12 to +0.56) | +0.12 (-0.20 to +0.44) | +0.12 (-0.12 to +0.36) | -0.05 (-0.45 to +0.35) |
| best detector at a 50% budget | T1_k0_cot_multi | +0.36 (+0.16 to +0.60) | +0.12 (-0.20 to +0.44) | +0.16 (-0.08 to +0.44) | -0.05 (-0.45 to +0.35) |
| smallest budget keeping 90% of the full validation gain | T1_k0_cot_multi | +0.28 (+0.08 to +0.52) | +0.00 (-0.28 to +0.28) | +0.20 (+0.04 to +0.40) | -0.10 (-0.50 to +0.30) |
| correct every question (no gate) | T1_k0_ground_cot_multi | +0.36 (+0.08 to +0.68) | +0.52 (+0.08 to +0.96) | +0.44 (+0.12 to +0.80) | +0.00 (-0.40 to +0.40) |
| best detector at a 20% budget | T1_k0_ground_cot_multi | +0.32 (+0.08 to +0.60) | +0.56 (+0.12 to +1.04) | +0.36 (+0.08 to +0.64) | +0.00 (-0.40 to +0.40) |
| best detector at a 30% budget | T1_k0_ground_cot_multi | +0.32 (+0.04 to +0.64) | +0.52 (+0.08 to +0.96) | +0.44 (+0.12 to +0.80) | +0.00 (-0.40 to +0.40) |
| best detector at a 50% budget | T1_k0_ground_cot_multi | +0.36 (+0.08 to +0.68) | +0.52 (+0.08 to +0.96) | +0.44 (+0.12 to +0.80) | +0.00 (-0.40 to +0.40) |
| smallest budget keeping 90% of the full validation gain | T1_k0_ground_cot_multi | +0.28 (+0.04 to +0.56) | +0.48 (+0.04 to +0.92) | +0.36 (+0.12 to +0.64) | -0.05 (-0.45 to +0.35) |
| best under 2 passes per question (any configuration) | T1_k0_ground_cot | +0.16 (-0.00 to +0.36) | +0.48 (+0.04 to +0.92) | +0.24 (+0.04 to +0.48) | +0.00 (-0.40 to +0.35) |
| best under 3 passes per question (any configuration) | T1_k0_ground_cot_multi | +0.28 (+0.04 to +0.56) | +0.48 (+0.04 to +0.92) | +0.36 (+0.12 to +0.64) | -0.05 (-0.45 to +0.35) |
| best under 5 passes per question (any configuration) | T1_k0_zoom_cot | +0.40 (+0.16 to +0.68) | +0.36 (-0.04 to +0.76) | +0.28 (+0.08 to +0.52) | -0.15 (-0.59 to +0.30) |
