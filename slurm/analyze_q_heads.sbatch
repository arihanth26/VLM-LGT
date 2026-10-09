#!/bin/bash
#SBATCH -J vlm-lgt-analyze
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/analyze-%j.out

# CPU job: scores the test split with every trained head and writes the analysis
# file (curves, reliability, categories, case studies, bootstrap intervals).
# Usage: sbatch --export=ALL,DATASET=textvqa slurm/analyze_q_heads.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

: "${DATASET:?Set DATASET to textvqa, docvqa, scienceqa_img, or chartqa_full}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.analyze_q_heads \
    --dataset "$DATASET" --size "${MODEL_SIZE:-2b}"
