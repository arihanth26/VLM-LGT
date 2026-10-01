#!/bin/bash
#SBATCH -J vlm-lgt-qfeatures
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/qfeatures-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
: "${SPLIT:?Set SPLIT to train, val, or test}"
: "${BASELINE:?Set BASELINE to the matching baseline JSONL path}"

ARGS=(
    --dataset chartqa_full
    --split "$SPLIT"
    --baseline "$BASELINE"
    --size "${MODEL_SIZE:-2b}"
)
if [[ "$SPLIT" != "test" ]]; then
    ARGS+=(--include-references)
fi

srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.extract_q_features "${ARGS[@]}"
