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
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.sweep_verifier_early_stop \
    --train experiments/approach1_latent_self_correction/q_features/chartqa_full_train_2b.pt \
    --validation experiments/approach1_latent_self_correction/q_features/chartqa_full_val_2b.pt \
    --test experiments/approach1_latent_self_correction/q_features/chartqa_full_test_2b.pt \
    --output experiments/approach1_latent_self_correction/q_models/verifier_early_stop_sweep.json \
    --max-epochs "${MAX_EPOCHS:-50}" \
    --patience "${PATIENCE:-5}"
