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

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.sweep_q_heads \
    --train data/approach1_latent_self_correction/q_features/chartqa_full_train_2b.pt \
    --validation data/approach1_latent_self_correction/q_features/chartqa_full_val_2b.pt \
    --test data/approach1_latent_self_correction/q_features/chartqa_full_test_2b.pt \
    --output data/approach1_latent_self_correction/q_models/sweep_results.json \
    --epochs "${EPOCHS:-20}"
