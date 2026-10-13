#!/bin/bash
#SBATCH -J vlm-lgt-corrupt
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/corrupt-%j.out

# GPU job: caches Qwen3-VL states for the same question and answer under a corrupted
# image, plus the model's own answer confidence on the clean and corrupted image.
# Needs the clean features and baseline for the same dataset split.
# Usage: sbatch --export=ALL,DATASET=textvqa,SPLIT=train slurm/extract_corrupt_features.sbatch
# Optional: MODEL_SIZE (2b), CORRUPTION (swap or shuffle), LIMIT (smoke test), BASELINE, OUTPUT

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

: "${DATASET:?Set DATASET to textvqa, docvqa, scienceqa_img, or chartqa_full}"
: "${SPLIT:?Set SPLIT to train, val, or test}"
SIZE="${MODEL_SIZE:-2b}"
BASE=experiments/approach1_latent_self_correction
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(
    --dataset "$DATASET"
    --split "$SPLIT"
    --baseline "${BASELINE:-$BASE/results/${DATASET}_${SPLIT}_${SIZE}_baseline.jsonl}"
    --size "$SIZE"
    --mode "${CORRUPTION:-swap}"
)
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "$LIMIT")
[[ -n "${OUTPUT:-}" ]] && ARGS+=(--output "$OUTPUT")

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.extract_corrupt_features "${ARGS[@]}"
