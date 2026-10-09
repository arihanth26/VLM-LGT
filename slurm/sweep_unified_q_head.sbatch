#!/bin/bash
#SBATCH -J vlm-lgt-unified-sweep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/unified-sweep-%j.out

# Sweeps the unified Q-head on cached features for one dataset.
# Usage: sbatch --export=ALL,DATASET=textvqa slurm/sweep_unified_q_head.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
DATASET="${DATASET:-chartqa_full}"
SIZE="${MODEL_SIZE:-2b}"
BASE=data/approach1_latent_self_correction
mkdir -p "$BASE/q_models/$DATASET"
srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.sweep_unified_q_head \
    --train "$BASE/q_features/${DATASET}_train_${SIZE}.pt" \
    --validation "$BASE/q_features/${DATASET}_val_${SIZE}.pt" \
    --test "$BASE/q_features/${DATASET}_test_${SIZE}.pt" \
    --output "$BASE/q_models/$DATASET/unified_sweep_results.json" \
    --select "${SELECT:-mean}"
