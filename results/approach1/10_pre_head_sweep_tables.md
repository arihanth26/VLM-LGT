# Pre-generation head: training sweep

Input: the cached question embedding (constant across every row). Pass rate: first 8 samples at T = 1. Winners chosen on validation; numbers are test means over three seeds.

| Row | Config | AUROC (greedy) | AURC ↓ | E-AURC ↓ | Spearman with pass rate | Routing gain at 20% | Mean routing gain |
|---|---|---:|---:|---:|---:|---:|---:|
| exported pre-generation (MLP) | reference | 0.862 | 0.0468 | 0.0304 | 0.584 | +0.15 | +0.25 |
| exported pre-generation (linear) | reference | 0.858 | 0.0483 | 0.0319 | 0.575 | +0.07 | +0.26 |
| exported unified, pre-generation mode | reference | 0.859 | 0.0478 | 0.0314 | 0.576 | +0.14 | +0.25 |
| winner: gate | pass_rate, mlp_256, lr 0.0003, wd 0.01, dropout 0.3, rank 0.5 | 0.867 | 0.0450 | 0.0286 | 0.595 | +0.15 | +0.25 |
| winner: routing | pass_rate, mlp_256, lr 0.0003, wd 0.0001, dropout 0.3, rank 0.5 | 0.867 | 0.0450 | 0.0286 | 0.595 | +0.16 | +0.25 |
| winner: gate_greedy | greedy, mlp_256, lr 0.0003, wd 0.01, dropout 0.3, rank 0.5 | 0.867 | 0.0450 | 0.0286 | 0.595 | +0.14 | +0.24 |
| winner: gate_pass_rate | pass_rate, mlp_256, lr 0.0003, wd 0.01, dropout 0.3, rank 0.5 | 0.867 | 0.0450 | 0.0286 | 0.595 | +0.15 | +0.25 |
| winner: gate_blend | blend, mlp_256, lr 0.0003, wd 0.01, dropout 0.3, rank 0.5 | 0.868 | 0.0448 | 0.0284 | 0.596 | +0.16 | +0.25 |
| winner: gate_any_correct | any_correct, mlp_256, lr 0.0003, wd 0.01, dropout 0.3, rank 0.5 | 0.864 | 0.0456 | 0.0292 | 0.587 | +0.13 | +0.22 |

## Per dataset (test AUROC against greedy correctness / AURC)

| Row | chartqa_full | textvqa | docvqa | scienceqa_img |
|---|---:|---:|---:|---:|
| exported pre-generation (MLP) | 0.893 / 0.0626 | 0.828 / 0.0615 | 0.791 / 0.0487 | 0.901 / 0.0239 |
| exported pre-generation (linear) | 0.893 / 0.0635 | 0.820 / 0.0654 | 0.786 / 0.0500 | 0.895 / 0.0251 |
| exported unified, pre-generation mode | 0.895 / 0.0618 | 0.821 / 0.0641 | 0.784 / 0.0499 | 0.903 / 0.0236 |
| winner: gate | 0.890 / 0.0626 | 0.830 / 0.0595 | 0.798 / 0.0463 | 0.912 / 0.0220 |
| winner: routing | 0.890 / 0.0627 | 0.830 / 0.0595 | 0.798 / 0.0463 | 0.912 / 0.0220 |
| winner: gate_greedy | 0.892 / 0.0622 | 0.830 / 0.0600 | 0.800 / 0.0463 | 0.911 / 0.0222 |
| winner: gate_pass_rate | 0.890 / 0.0626 | 0.830 / 0.0595 | 0.798 / 0.0463 | 0.912 / 0.0220 |
| winner: gate_blend | 0.893 / 0.0617 | 0.831 / 0.0595 | 0.801 / 0.0460 | 0.911 / 0.0222 |
| winner: gate_any_correct | 0.887 / 0.0633 | 0.831 / 0.0599 | 0.796 / 0.0466 | 0.907 / 0.0227 |
