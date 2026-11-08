#!/bin/bash
#SBATCH -J vlm-lgt-corr2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/corr2-%j.out

# GPU job: second-round correction candidates (grounded crop, grounded crop with context, step-by-step
# reasoning) and the self-judge feature for every candidate of one shard. With OLD_CORRECTIONS (train),
# also contrastive decoding and corner crops where missing. Resumable.
# Usage: sbatch --export=ALL,DATASET=textvqa,SPLIT=test,SHARD=0,NUM_SHARDS=3 slurm/run_correction_v2.sbatch
# Optional: MAX_EXAMPLES, OLD_CORRECTIONS, OUTPUT

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
ARGS=(--dataset "$DATASET" --split "$SPLIT" --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}")
[[ -n "${MAX_EXAMPLES:-}" ]] && ARGS+=(--max-examples "$MAX_EXAMPLES")
[[ -n "${OLD_CORRECTIONS:-}" ]] && ARGS+=(--old-corrections)
[[ -n "${OUTPUT:-}" ]] && ARGS+=(--output "$OUTPUT")
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.correction_v2 "${ARGS[@]}"
