# Q-head comparison

## textvqa

Base model accuracy on the test split: 81.04%

| Metric | Token confidence | Separate pre-gen | Separate verifier | Unified, pre-gen mode | Unified, joint mode |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC ↑ | 0.726 | 0.816 | 0.868 | 0.812 ± 0.002 | 0.867 ± 0.002 |
| Error AUPRC ↑ | 0.449 | 0.527 | 0.644 | 0.514 ± 0.004 | 0.635 ± 0.003 |
| Brier score ↓ | 0.158 | 0.122 | 0.103 | 0.123 ± 0.003 | 0.107 ± 0.001 |
| ECE (10 bins) ↓ | 0.152 | 0.042 | 0.016 | 0.034 ± 0.021 | 0.037 ± 0.013 |
| Pearson r with correctness ↑ | 0.379 | 0.472 | 0.573 | 0.462 ± 0.003 | 0.566 ± 0.004 |

Separate heads picked by validation: pre-gen {'architecture': 'mlp_256', 'learning_rate': 0.0003, 'ranking_weight': 0.0, 'seed': 2}, verifier {'architecture': 'mlp_256', 'learning_rate': 0.001, 'ranking_weight': 0.0, 'seed': 2}.
Unified winner: {'architecture': 'mlp_256', 'learning_rate': 0.001, 'pre_weight': 1.0, 'rank_weight': 0.0, 'pair_weight': 0.0}, selected on mean validation score.

Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.

## docvqa

Base model accuracy on the test split: 87.00%

| Metric | Token confidence | Separate pre-gen | Separate verifier | Unified, pre-gen mode | Unified, joint mode |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC ↑ | 0.685 | 0.776 | 0.788 | 0.769 ± 0.005 | 0.811 ± 0.001 |
| Error AUPRC ↑ | 0.333 | 0.399 | 0.443 | 0.384 ± 0.008 | 0.462 ± 0.002 |
| Brier score ↓ | 0.114 | 0.096 | 0.094 | 0.097 ± 0.001 | 0.090 ± 0.000 |
| ECE (10 bins) ↓ | 0.106 | 0.015 | 0.013 | 0.020 ± 0.003 | 0.014 ± 0.000 |
| Pearson r with correctness ↑ | 0.312 | 0.390 | 0.426 | 0.379 ± 0.008 | 0.453 ± 0.001 |

Separate heads picked by validation: pre-gen {'architecture': 'mlp_256', 'learning_rate': 0.0003, 'ranking_weight': 0.0, 'seed': 1}, verifier {'architecture': 'linear', 'learning_rate': 0.001, 'ranking_weight': 0.0, 'seed': 1}.
Unified winner: {'architecture': 'mlp_512', 'learning_rate': 0.001, 'pre_weight': 1.0, 'rank_weight': 0.0, 'pair_weight': 0.0}, selected on mean validation score.

Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.

## scienceqa_img

Base model accuracy on the test split: 86.71%

| Metric | Token confidence | Separate pre-gen | Separate verifier | Unified, pre-gen mode | Unified, joint mode |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC ↑ | 0.654 | 0.918 | 0.879 | 0.923 ± 0.002 | 0.925 ± 0.001 |
| Error AUPRC ↑ | 0.326 | 0.560 | 0.503 | 0.573 ± 0.012 | 0.577 ± 0.007 |
| Brier score ↓ | 0.116 | 0.078 | 0.084 | 0.073 ± 0.001 | 0.072 ± 0.001 |
| ECE (10 bins) ↓ | 0.108 | 0.036 | 0.023 | 0.019 ± 0.004 | 0.021 ± 0.005 |
| Pearson r with correctness ↑ | 0.321 | 0.590 | 0.527 | 0.605 ± 0.005 | 0.613 ± 0.004 |

Separate heads picked by validation: pre-gen {'architecture': 'mlp_512', 'learning_rate': 0.001, 'ranking_weight': 0.25, 'seed': 2}, verifier {'architecture': 'mlp_256', 'learning_rate': 0.001, 'ranking_weight': 0.0, 'seed': 3}.
Unified winner: {'architecture': 'mlp_512', 'learning_rate': 0.001, 'pre_weight': 1.0, 'rank_weight': 0.0, 'pair_weight': 0.0}, selected on mean validation score.

Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.
