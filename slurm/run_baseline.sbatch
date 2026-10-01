#!/bin/bash
#SBATCH -J vlm-lgt-baseline
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH -o logs/baseline-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load anaconda3

if [[ -z "${DATASET:-}" ]]; then
    echo "Set DATASET to a key from the Approach 1 dataset config."
    exit 2
fi

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Run: sbatch slurm/setup_environment.sbatch"
    exit 2
fi

ARGS=(
    --dataset "$DATASET"
    --group "${DATA_GROUP:-ood}"
    --split "${SPLIT:-test}"
    --size "${MODEL_SIZE:-2b}"
    --max-new-tokens "${MAX_NEW_TOKENS:-512}"
)
if [[ -n "${LIMIT:-}" ]]; then
    ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${OUTPUT:-}" ]]; then
    ARGS+=(--output "$OUTPUT")
fi

srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.baseline "${ARGS[@]}"
