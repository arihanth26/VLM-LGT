# Trained correctness heads (Approach 1)

Small correctness heads trained on the pooled mixture of ChartQA, TextVQA, DocVQA and ScienceQA. They read cached
last-layer, last-token hidden states of the frozen Qwen3-VL-2B (feature size 2048) and output the probability that
the model's answer is correct. The VLM itself is not part of these files and is never changed.

## The files

`<kind>_<variant>.pt`, nine in total.

- kind: `pre` (reads the question state, scores before the answer exists), `verifier` (reads the question plus answer
  state), `unified` (reads both, joint mode, and can also run with the answer half left out).
- variant: `old` is the old loss. `new_swap` is the new loss with the grounding term built from a swapped image.
  `new_noise` is the new loss with the grounding term built from a noised image.

Each file holds the weights (`state_dict`), the head spec, the temperature of each mode, the best epoch, and the reproduced
and reported test AUROC. `manifest.json` lists the same for all nine.

| File | Architecture | Learning rate | Ground weight | Pair weight | Test AUROC (this head, seed 1) | Sweep mean (3 seeds) |
|---|---|---|---|---|---|---|
| pre_old.pt | mlp_256 | 0.0003 | 0.0 | 0.0 | 0.8645 | 0.8643 |
| verifier_old.pt | mlp_512 | 0.0003 | 0.0 | 0.0 | 0.8732 | 0.8727 |
| unified_old.pt | mlp_512 | 0.0003 | 0.0 | 0.0 | 0.8789 | 0.8813 |
| pre_new_swap.pt | mlp_512 | 0.001 | 0.1 | 0.0 | 0.8623 | 0.8625 |
| verifier_new_swap.pt | mlp_512 | 0.0003 | 0.1 | 0.0 | 0.8711 | 0.8714 |
| unified_new_swap.pt | mlp_512 | 0.0003 | 0.1 | 0.0 | 0.8821 | 0.8822 |
| pre_new_noise.pt | mlp_512 | 0.001 | 0.1 | 0.0 | 0.8615 | 0.8625 |
| verifier_new_noise.pt | mlp_256 | 0.001 | 0.1 | 0.0 | 0.8706 | 0.8731 |
| unified_new_noise.pt | mlp_512 | 0.0003 | 0.1 | 0.0 | 0.8860 | 0.8837 |

Each head is one seed (seed 1) retrained from the winning configuration of the sweep, since the sweeps keep weights only
in memory. The reproduced AUROC is within about 0.003 of the sweep mean, which is the seed spread (the largest gap is the
unified old head, 0.8789 against 0.8813). Reported numbers in the results reports are the three-seed means.

## Using a head

```python
from pathlib import Path
from experiments.approach1_latent_self_correction.export_heads import load_head, score

head, spec, temperatures = load_head(Path("trained_heads/approach1/unified_new_noise.pt"))
# prompt, generated: tensors of shape (n, feature_size) from extract_q_features.py
probabilities = score(head, spec, temperatures, prompt, generated)   # {"joint": ..., "pre_generation": ...}
before_answer = score(head, spec, temperatures, prompt)             # unified head without the answer state
```

- `pre` takes only `prompt`. `verifier` takes `prompt` and `generated` and uses `generated`. `unified` takes both, or only
  `prompt` for pre-generation scoring.
- `prompt` is the state after the question with the generation prompt. `generated` is the state after the question and the
  model's answer. Both come from `experiments/approach1_latent_self_correction/extract_q_features.py`.
- Probabilities are temperature scaled, using the temperature stored for each mode.

## Rebuilding them

```
sbatch slurm/export_heads.sbatch
```

This needs the mixture and noise-mixture sweeps and their cached features on the cluster. Heads for the per-dataset
experiments (each dataset alone, swap and noise) and the sequential-training runs are not saved. They can be exported the same way.
