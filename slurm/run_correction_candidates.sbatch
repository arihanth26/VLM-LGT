#!/bin/bash
#SBATCH -J vlm-lgt-corrcand
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/corrcand-%j.out

# GPU job: correction candidates (visual contrastive decoding, zoomed crops) and the encoded state of
# every distinct candidate of one shard, for the learned candidate ranker. Needs the candidate pools.
# Resumable. Output: experiments/approach1_latent_self_correction/results/correction/<split>/.
# Usage: sbatch --export=ALL,DATASET=textvqa,SPLIT=val slurm/run_correction_candidates.sbatch
# Optional: SHARD (0), NUM_SHARDS (1), MAX_EXAMPLES, NO_CORRECTIONS, CORRECTIONS_LIMIT, OUTPUT

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
[[ -n "${NO_CORRECTIONS:-}" ]] && ARGS+=(--no-corrections)
[[ -n "${CORRECTIONS_LIMIT:-}" ]] && ARGS+=(--corrections-limit "$CORRECTIONS_LIMIT")
[[ -n "${OUTPUT:-}" ]] && ARGS+=(--output "$OUTPUT")
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.correction_candidates "${ARGS[@]}"
