#!/bin/bash
#SBATCH -J vlm-lgt-extend
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/extend-%j.out

# GPU job: adds EXTRA sampled answers (default 6, so pools reach 16) to one dataset split's pool at one
# temperature, and encodes the new answers for the candidate ranker. Resumable.
# Usage: sbatch --export=ALL,DATASET=textvqa,SPLIT=test,TEMPERATURE=1 slurm/run_extend_pools.sbatch
# Optional: EXTRA (6), SHARD (0), NUM_SHARDS (1), MAX_EXAMPLES, OUTPUT

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT to val or test}"
: "${TEMPERATURE:?Set TEMPERATURE to 1 or 1.3}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
ARGS=(--dataset "$DATASET" --split "$SPLIT" --temperature "$TEMPERATURE" --extra "${EXTRA:-6}" --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}")
[[ -n "${MAX_EXAMPLES:-}" ]] && ARGS+=(--max-examples "$MAX_EXAMPLES")
[[ -n "${OUTPUT:-}" ]] && ARGS+=(--output "$OUTPUT")
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.extend_pools "${ARGS[@]}"
