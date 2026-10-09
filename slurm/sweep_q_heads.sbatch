#!/bin/bash
#SBATCH -J vlm-lgt-qsweep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/qsweep-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
DATASET="${DATASET:-chartqa_full}"
SIZE="${MODEL_SIZE:-2b}"
BASE=experiments/approach1_latent_self_correction
mkdir -p "$BASE/q_models/$DATASET"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.sweep_q_heads \
    --train "$BASE/q_features/${DATASET}_train_${SIZE}.pt" \
    --validation "$BASE/q_features/${DATASET}_val_${SIZE}.pt" \
    --test "$BASE/q_features/${DATASET}_test_${SIZE}.pt" \
    --output "$BASE/q_models/$DATASET/sweep_results.json" \
    --epochs "${EPOCHS:-20}"
