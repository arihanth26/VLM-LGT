# Trained heads against linear probes, per dataset

Linear probe: the best linear head of the sweep. Trained head: the best head of the same sweep over linear, MLP-256 and MLP-512. Both chosen on validation with the old loss; test numbers are the mean over three seeds. AUROC difference is head minus linear probe. The untrained answer-token probability is the reference.

## Trained on the pooled mixture

| Dataset | Head | Answer-token prob. AUROC | Linear probe AUROC | Trained head AUROC | Difference | Head architecture (parameters) | Linear probe error AUPRC | Trained head error AUPRC | Reading |
|---|---|---|---|---|---|---|---|---|---|
| ChartQA | Pre-generation head | 0.751 | 0.893 ± 0.000 | 0.894 ± 0.001 | +0.002 | mlp_256 (528,897) | 0.729 | 0.742 | head better |
| ChartQA | Verifier head | 0.751 | 0.885 ± 0.000 | 0.895 ± 0.001 | +0.010 | mlp_512 (1,053,697) | 0.703 | 0.740 | head better |
| ChartQA | Unified, pre-generation mode | 0.751 | 0.892 ± 0.000 | 0.893 ± 0.003 | +0.001 | mlp_512 (2,106,881) | 0.735 | 0.734 | within seed spread |
| ChartQA | Unified, joint mode | 0.751 | 0.896 ± 0.000 | 0.901 ± 0.002 | +0.004 | mlp_512 (2,106,881) | 0.744 | 0.752 | head better |
| TextVQA | Pre-generation head | 0.726 | 0.821 ± 0.002 | 0.831 ± 0.001 | +0.010 | mlp_256 (528,897) | 0.545 | 0.556 | head better |
| TextVQA | Verifier head | 0.726 | 0.859 ± 0.003 | 0.869 ± 0.001 | +0.010 | mlp_512 (1,053,697) | 0.636 | 0.653 | head better |
| TextVQA | Unified, pre-generation mode | 0.726 | 0.821 ± 0.000 | 0.823 ± 0.008 | +0.002 | mlp_512 (2,106,881) | 0.545 | 0.537 | within seed spread |
| TextVQA | Unified, joint mode | 0.726 | 0.871 ± 0.000 | 0.874 ± 0.003 | +0.003 | mlp_512 (2,106,881) | 0.653 | 0.657 | within seed spread |
| DocVQA | Pre-generation head | 0.685 | 0.781 ± 0.011 | 0.793 ± 0.004 | +0.013 | mlp_256 (528,897) | 0.402 | 0.418 | head better |
| DocVQA | Verifier head | 0.685 | 0.798 ± 0.002 | 0.807 ± 0.001 | +0.009 | mlp_512 (1,053,697) | 0.449 | 0.459 | head better |
| DocVQA | Unified, pre-generation mode | 0.685 | 0.782 ± 0.002 | 0.788 ± 0.002 | +0.006 | mlp_512 (2,106,881) | 0.402 | 0.409 | head better |
| DocVQA | Unified, joint mode | 0.685 | 0.814 ± 0.002 | 0.816 ± 0.006 | +0.003 | mlp_512 (2,106,881) | 0.466 | 0.480 | within seed spread |
| ScienceQA | Pre-generation head | 0.654 | 0.897 ± 0.000 | 0.903 ± 0.002 | +0.006 | mlp_256 (528,897) | 0.532 | 0.554 | head better |
| ScienceQA | Verifier head | 0.654 | 0.885 ± 0.001 | 0.890 ± 0.001 | +0.005 | mlp_512 (1,053,697) | 0.514 | 0.538 | head better |
| ScienceQA | Unified, pre-generation mode | 0.654 | 0.898 ± 0.000 | 0.906 ± 0.003 | +0.008 | mlp_512 (2,106,881) | 0.535 | 0.559 | head better |
| ScienceQA | Unified, joint mode | 0.654 | 0.904 ± 0.001 | 0.910 ± 0.003 | +0.006 | mlp_512 (2,106,881) | 0.546 | 0.563 | head better |

## Trained on each dataset alone

| Dataset | Head | Answer-token prob. AUROC | Linear probe AUROC | Trained head AUROC | Difference | Head architecture (parameters) | Linear probe error AUPRC | Trained head error AUPRC | Reading |
|---|---|---|---|---|---|---|---|---|---|
| ChartQA | Pre-generation head | 0.751 | 0.893 ± 0.002 | 0.896 ± 0.001 | +0.003 | mlp_512 (1,053,697) | 0.727 | 0.745 | head better |
| ChartQA | Verifier head | 0.751 | 0.886 ± 0.000 | 0.894 ± 0.002 | +0.007 | mlp_256 (528,897) | 0.709 | 0.736 | head better |
| ChartQA | Unified, pre-generation mode | 0.751 | 0.895 ± 0.000 | 0.895 ± 0.000 | +0.000 | linear (12,290) | 0.741 | 0.741 | same model (the best head is linear) |
| ChartQA | Unified, joint mode | 0.751 | 0.898 ± 0.000 | 0.898 ± 0.000 | +0.000 | linear (12,290) | 0.750 | 0.750 | same model (the best head is linear) |
| TextVQA | Pre-generation head | 0.726 | 0.819 ± 0.002 | 0.818 ± 0.003 | -0.001 | mlp_256 (528,897) | 0.525 | 0.527 | within seed spread |
| TextVQA | Verifier head | 0.726 | 0.866 ± 0.002 | 0.869 ± 0.001 | +0.003 | mlp_256 (528,897) | 0.639 | 0.647 | head better |
| TextVQA | Unified, pre-generation mode | 0.726 | 0.820 ± 0.001 | 0.814 ± 0.001 | -0.006 | mlp_512 (2,106,881) | 0.533 | 0.511 | linear probe better |
| TextVQA | Unified, joint mode | 0.726 | 0.870 ± 0.000 | 0.867 ± 0.002 | -0.002 | mlp_512 (2,106,881) | 0.642 | 0.635 | linear probe better |
| DocVQA | Pre-generation head | 0.685 | 0.773 ± 0.007 | 0.781 ± 0.004 | +0.008 | mlp_256 (528,897) | 0.394 | 0.391 | head better |
| DocVQA | Verifier head | 0.685 | 0.788 ± 0.002 | 0.790 ± 0.000 | +0.002 | mlp_512 (1,053,697) | 0.443 | 0.451 | head better |
| DocVQA | Unified, pre-generation mode | 0.685 | 0.766 ± 0.003 | 0.771 ± 0.005 | +0.005 | mlp_256 (1,057,537) | 0.380 | 0.387 | within seed spread |
| DocVQA | Unified, joint mode | 0.685 | 0.806 ± 0.002 | 0.808 ± 0.002 | +0.002 | mlp_256 (1,057,537) | 0.452 | 0.459 | within seed spread |
| ScienceQA | Pre-generation head | 0.654 | 0.901 ± 0.001 | 0.910 ± 0.007 | +0.009 | mlp_256 (528,897) | 0.540 | 0.541 | head better |
| ScienceQA | Verifier head | 0.654 | 0.875 ± 0.006 | 0.878 ± 0.001 | +0.004 | mlp_256 (528,897) | 0.486 | 0.505 | within seed spread |
| ScienceQA | Unified, pre-generation mode | 0.654 | 0.902 ± 0.003 | 0.902 ± 0.006 | +0.000 | mlp_512 (2,106,881) | 0.539 | 0.537 | within seed spread |
| ScienceQA | Unified, joint mode | 0.654 | 0.897 ± 0.003 | 0.906 ± 0.006 | +0.009 | mlp_512 (2,106,881) | 0.540 | 0.545 | head better |
