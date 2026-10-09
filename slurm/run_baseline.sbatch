#!/bin/bash
#SBATCH -J vlm-lgt-baseline
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/baseline-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

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
