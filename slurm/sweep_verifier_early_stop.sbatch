#!/bin/bash
#SBATCH -J vlm-lgt-vesweep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/vesweep-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
DATASET="${DATASET:-chartqa_full}"
SIZE="${MODEL_SIZE:-2b}"
BASE=experiments/approach1_latent_self_correction
mkdir -p "$BASE/q_models/$DATASET"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.sweep_verifier_early_stop \
    --train "$BASE/q_features/${DATASET}_train_${SIZE}.pt" \
    --validation "$BASE/q_features/${DATASET}_val_${SIZE}.pt" \
    --test "$BASE/q_features/${DATASET}_test_${SIZE}.pt" \
    --output "$BASE/q_models/$DATASET/verifier_early_stop_sweep.json" \
    --max-epochs "${MAX_EPOCHS:-50}" \
    --patience "${PATIENCE:-5}"
