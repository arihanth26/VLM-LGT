#!/bin/bash
#SBATCH -J vlm-lgt-qheads
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/qheads-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
DATASET="${DATASET:-chartqa_full}"
SIZE="${MODEL_SIZE:-2b}"
BASE=experiments/approach1_latent_self_correction
mkdir -p "$BASE/q_models/$DATASET"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.train_q_heads \
    --train "$BASE/q_features/${DATASET}_train_${SIZE}.pt" \
    --validation "$BASE/q_features/${DATASET}_val_${SIZE}.pt" \
    --test "$BASE/q_features/${DATASET}_test_${SIZE}.pt" \
    --output "$BASE/q_models/$DATASET/comparison.json" \
    --epochs "${EPOCHS:-20}"
