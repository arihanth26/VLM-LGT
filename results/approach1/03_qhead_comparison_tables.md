# Q-head comparison

## chartqa_full

| Metric | Token confidence | Separate pre-gen | Separate verifier | Unified, pre-gen mode | Unified, joint mode |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC ↑ | n/a | 0.895 | 0.892 | 0.894 ± 0.001 | 0.898 ± 0.001 |
| Error AUPRC ↑ | n/a | 0.732 | 0.732 | 0.741 ± 0.001 | 0.751 ± 0.002 |
| Brier score ↓ | n/a | 0.108 | 0.110 | 0.108 ± 0.001 | 0.107 ± 0.001 |
| ECE (10 bins) ↓ | n/a | 0.026 | 0.032 | 0.031 ± 0.005 | 0.038 ± 0.010 |
| Pearson r with correctness ↑ | n/a | 0.641 | 0.634 | 0.645 ± 0.001 | 0.652 ± 0.001 |

Separate heads picked by validation: pre-gen {'architecture': 'linear', 'learning_rate': 0.0003, 'ranking_weight': 0.0, 'seed': 2}, verifier {'architecture': 'mlp_512', 'learning_rate': 0.001, 'ranking_weight': 0.0, 'seed': 2}.
Unified winner: {'architecture': 'linear', 'learning_rate': 0.001, 'pre_weight': 0.5, 'rank_weight': 0.0, 'pair_weight': 0.0}, selected on mean validation score.

Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.

## textvqa

| Metric | Token confidence | Separate pre-gen | Separate verifier | Unified, pre-gen mode | Unified, joint mode |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC ↑ | n/a | 0.816 | 0.868 | 0.812 ± 0.002 | 0.867 ± 0.002 |
| Error AUPRC ↑ | n/a | 0.527 | 0.644 | 0.514 ± 0.004 | 0.635 ± 0.003 |
| Brier score ↓ | n/a | 0.122 | 0.103 | 0.123 ± 0.003 | 0.107 ± 0.001 |
| ECE (10 bins) ↓ | n/a | 0.042 | 0.016 | 0.034 ± 0.021 | 0.037 ± 0.013 |
| Pearson r with correctness ↑ | n/a | 0.472 | 0.573 | 0.462 ± 0.003 | 0.566 ± 0.004 |

Separate heads picked by validation: pre-gen {'architecture': 'mlp_256', 'learning_rate': 0.0003, 'ranking_weight': 0.0, 'seed': 2}, verifier {'architecture': 'mlp_256', 'learning_rate': 0.001, 'ranking_weight': 0.0, 'seed': 2}.
Unified winner: {'architecture': 'mlp_256', 'learning_rate': 0.001, 'pre_weight': 1.0, 'rank_weight': 0.0, 'pair_weight': 0.0}, selected on mean validation score.

Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.

## docvqa

| Metric | Token confidence | Separate pre-gen | Separate verifier | Unified, pre-gen mode | Unified, joint mode |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC ↑ | n/a | 0.776 | 0.788 | 0.769 ± 0.005 | 0.811 ± 0.001 |
| Error AUPRC ↑ | n/a | 0.399 | 0.443 | 0.384 ± 0.008 | 0.462 ± 0.002 |
| Brier score ↓ | n/a | 0.096 | 0.094 | 0.097 ± 0.001 | 0.090 ± 0.000 |
| ECE (10 bins) ↓ | n/a | 0.015 | 0.013 | 0.020 ± 0.003 | 0.014 ± 0.000 |
| Pearson r with correctness ↑ | n/a | 0.390 | 0.426 | 0.379 ± 0.008 | 0.453 ± 0.001 |

Separate heads picked by validation: pre-gen {'architecture': 'mlp_256', 'learning_rate': 0.0003, 'ranking_weight': 0.0, 'seed': 1}, verifier {'architecture': 'linear', 'learning_rate': 0.001, 'ranking_weight': 0.0, 'seed': 1}.
Unified winner: {'architecture': 'mlp_512', 'learning_rate': 0.001, 'pre_weight': 1.0, 'rank_weight': 0.0, 'pair_weight': 0.0}, selected on mean validation score.

Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.

## scienceqa_img

| Metric | Token confidence | Separate pre-gen | Separate verifier | Unified, pre-gen mode | Unified, joint mode |
|---|---:|---:|---:|---:|---:|
| Correctness AUROC ↑ | n/a | 0.918 | 0.879 | 0.923 ± 0.002 | 0.925 ± 0.001 |
| Error AUPRC ↑ | n/a | 0.560 | 0.503 | 0.573 ± 0.012 | 0.577 ± 0.007 |
| Brier score ↓ | n/a | 0.078 | 0.084 | 0.073 ± 0.001 | 0.072 ± 0.001 |
| ECE (10 bins) ↓ | n/a | 0.036 | 0.023 | 0.019 ± 0.004 | 0.021 ± 0.005 |
| Pearson r with correctness ↑ | n/a | 0.590 | 0.527 | 0.605 ± 0.005 | 0.613 ± 0.004 |

Separate heads picked by validation: pre-gen {'architecture': 'mlp_512', 'learning_rate': 0.001, 'ranking_weight': 0.25, 'seed': 2}, verifier {'architecture': 'mlp_256', 'learning_rate': 0.001, 'ranking_weight': 0.0, 'seed': 3}.
Unified winner: {'architecture': 'mlp_512', 'learning_rate': 0.001, 'pre_weight': 1.0, 'rank_weight': 0.0, 'pair_weight': 0.0}, selected on mean validation score.

Separate heads are a single run each. Unified numbers are mean ± std over 3 seeds.
