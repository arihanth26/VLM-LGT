#!/bin/bash
#SBATCH -J vlm-lgt-build-splits
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/build-splits-%j.out

# CPU job: downloads one Q-head dataset and writes its train/val/test splits.
# Usage: sbatch --export=ALL,DATASET=textvqa slurm/build_qhead_splits.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

: "${DATASET:?Set DATASET to textvqa, docvqa, or scienceqa_img}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.build_qhead_splits \
    --dataset "$DATASET" \
    --train-size "${TRAIN_SIZE:-8000}" \
    --val-size "${VAL_SIZE:-1500}" \
    --test-size "${TEST_SIZE:-2500}" \
    --max-side "${MAX_SIDE:-1536}"
