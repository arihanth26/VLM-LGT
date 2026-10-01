#!/bin/bash
#SBATCH -J vlm-lgt-qheads
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/qheads-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.train_q_heads \
    --train data/approach1_latent_self_correction/q_features/chartqa_full_train_2b.pt \
    --validation data/approach1_latent_self_correction/q_features/chartqa_full_val_2b.pt \
    --test data/approach1_latent_self_correction/q_features/chartqa_full_test_2b.pt \
    --output data/approach1_latent_self_correction/q_models/comparison.json \
    --epochs "${EPOCHS:-20}"
