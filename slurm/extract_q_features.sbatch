#!/bin/bash
#SBATCH -J vlm-lgt-qfeatures
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/qfeatures-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
: "${SPLIT:?Set SPLIT to train, val, or test}"
: "${BASELINE:?Set BASELINE to the matching baseline JSONL path}"

ARGS=(
    --dataset "${DATASET:-chartqa_full}"
    --split "$SPLIT"
    --baseline "$BASELINE"
    --size "${MODEL_SIZE:-2b}"
)
if [[ "$SPLIT" != "test" ]]; then
    ARGS+=(--include-references)
fi

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.extract_q_features "${ARGS[@]}"
