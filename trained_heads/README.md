# Trained correctness heads (Approach 1)

Every correctness head trained in the Approach 1 experiments: 540 files, 1.5 GB. They read cached last-layer, last-token
hidden states of the frozen Qwen3-VL-2B (feature size 2048) and output the probability that the model's answer is correct.
The VLM itself is not in these files and was never changed. To use a head, run the VLM to get its answer, extract the states
with `experiments/approach1_latent_self_correction/extract_q_features.py`, and pass them to the head.

The sweeps kept weights only in memory, so these heads were retrained from the winning configuration of each sweep, with the
same data, seeds and early stopping, and checked against the sweep results (see "How they were checked").

## What is here

### winners/ (468 heads, three seeds each)

`winners/<tag>/<kind>_<group>_seed<N>.pt`, with a `manifest.json` per tag.

- **kind:** `pre` (reads the question state, scores before the answer exists), `verifier` (reads the question plus answer
  state), `unified` (reads both, joint mode, and can also run with the answer half left out).
- **group:** the winning configuration of the sweep for that group. `old` is the best old-loss head and `new` the best
  new-loss head. `old_linear` and `new_linear` are the best linear probes. `new_full` is the best new loss with both contrastive
  terms on. `cell_plain`, `cell_contrast`, `cell_ground` and `cell_full` are the four cells of the 2 x 2 ablation. The verifier and
  unified kinds have all nine groups and `pre` has the first four. When several groups share a configuration, the file
  is named after the first group and the manifest lists every group it serves.
- **tag:** which data and which grounding corruption the head was trained with.

| Folder | Trained on | Configurations | Files | Largest gap to sweep mean |
|---|---|---|---|---|
| `winners/mixture/` | Pooled mixture, swap | 16 | 48 | 0.0006 |
| `winners/mixture_noise/` | Pooled mixture, new loss with noise | 16 | 48 | 0.0000 |
| `winners/chartqa_swap/` | ChartQA alone, swap | 14 | 42 | 0.0047 |
| `winners/chartqa_noise/` | ChartQA alone, noise | 14 | 42 | 0.0001 |
| `winners/textvqa_swap/` | TextVQA alone, swap | 16 | 48 | 0.0015 |
| `winners/textvqa_noise/` | TextVQA alone, noise | 16 | 48 | 0.0000 |
| `winners/docvqa_swap/` | DocVQA alone, swap | 16 | 48 | 0.0000 |
| `winners/docvqa_noise/` | DocVQA alone, noise | 16 | 48 | 0.0000 |
| `winners/scienceqa_swap/` | ScienceQA alone, swap | 16 | 48 | 0.0023 |
| `winners/scienceqa_noise/` | ScienceQA alone, noise | 16 | 48 | 0.0000 |

Every folder holds its own `old` heads. For the same data, the swap and noise versions of the old loss are the same configuration
and seeds, because the old loss has no grounding term, so their numbers match.

### sequential/ (72 heads, seed 1)

`sequential/<tag>/<kind>_<variant>_<order>_seed1.pt`, with `manifest_<kind>.json` per kind. Each is the head after the last stage
of sequential training (train on one dataset, then the next), so it is the head whose earlier-dataset memory was measured.

- **variant:** `old_replay0`, `old_replay0.1`, `new_replay0`, `new_replay0.1` (old or new loss, without or with a 10% replay buffer).
- **order:** `forward` (ChartQA, TextVQA, DocVQA, ScienceQA), `reverse`, or `shuffled`.
- **tag:** `mixture` (swap) or `mixture_noise`.

The manifest lists each head's final test AUROC on every dataset. Only seed 1 is saved here. The reports average three seeds.

## How they were checked

- All 540 files load and score random inputs with `load_head` and `score`, with probabilities between 0 and 1.
- For each winning configuration, the test AUROC averaged over the three saved seeds is compared with the three-seed mean in the sweep
  results. The last column of the table is the largest gap in that folder. The pooled and noise folders reproduce to under 0.001. The
  largest gap anywhere is 0.0047 (ChartQA with swap, verifier, new loss with both terms), which is about the size of the seed spread.

## Using a head

```python
from pathlib import Path
from experiments.approach1_latent_self_correction.export_heads import load_head, score

head, spec, temperatures = load_head(Path("trained_heads/approach1/winners/mixture_noise/unified_new_seed1.pt"))
# prompt, generated: tensors of shape (n, 2048) from extract_q_features.py
probabilities = score(head, spec, temperatures, prompt, generated)   # {"joint": ..., "pre_generation": ...}
before_answer = score(head, spec, temperatures, prompt)             # unified head without the answer state
```

- `pre` takes only `prompt`. `verifier` takes `prompt` and `generated` and uses `generated`. `unified` takes both, or only `prompt`
  for pre-generation scoring.
- `prompt` is the state after the question with the generation prompt. `generated` is the state after the question and the model's answer.
- Probabilities are temperature scaled, using the temperature stored for each mode.
- A saved file also holds the head spec (architecture, learning rate, loss weights), the best epoch, and the test AUROC it reached.

## Rebuilding them

```
bash slurm/run_export_pipeline.sh
```

This needs the sweeps and cached features on the cluster. One job saves the winners of one tag, and one job saves the sequential heads of one tag and head kind.
Not saved: seeds 2 and 3 of the sequential heads, and the isolated per-dataset references inside the sequential runs.
