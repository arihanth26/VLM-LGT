#!/bin/bash
#SBATCH -J vlm-lgt-pool
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/pool-%j.out

# GPU job: one shard of candidate pools (greedy + K samples at one temperature) for one dataset
# split, read from the splits cached by the Q-head pipeline. Scored with the exported unified head
# and linear probe using the cached question embedding, unless LABELS_ONLY is set (train pass rates).
# Resumable. Output: experiments/approach1_latent_self_correction/results/pools/T<temp>/<split>/.
# Usage: sbatch --export=ALL,DATASET=textvqa,SPLIT=val,TEMPERATURE=1.0 slurm/run_candidate_pool.sbatch
# Optional: SAMPLES (10), TOP_P (0.95), SHARD (0), NUM_SHARDS (1), LABELS_ONLY, SMOKE (examples per shard), CHECK_EMBEDDING

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT to train, val, or test}"
: "${TEMPERATURE:?Set TEMPERATURE}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(--dataset "$DATASET" --split "$SPLIT" --temperature "$TEMPERATURE" --samples "${SAMPLES:-10}" --top-p "${TOP_P:-0.95}"
      --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}")
[[ -n "${LABELS_ONLY:-}" ]] && ARGS+=(--labels-only)
[[ -n "${CHECK_EMBEDDING:-}" ]] && ARGS+=(--check-embedding)
if [[ -n "${SMOKE:-}" ]]; then
    ARGS+=(--max-examples "$SMOKE" --output "experiments/approach1_latent_self_correction/results/pools_smoke/T${TEMPERATURE}_${SPLIT}_${DATASET}.jsonl")
fi
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.candidate_pools "${ARGS[@]}"
