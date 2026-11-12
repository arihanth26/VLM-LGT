#!/bin/bash
#SBATCH -J vlm-lgt-cot3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/cot3-%j.out

# GPU job: several sampled step-by-step reasoning answers per question (correction_v3_cot.py), with their
# encodings and judge scores, for one shard of the questions correction_v2 covered. Resumable.
# Usage: sbatch --export=ALL,DATASET=chartqa_full,SPLIT=val,SHARD=0,NUM_SHARDS=6 slurm/run_correction_v3_cot.sbatch
# Optional: MAX_EXAMPLES, SAMPLES, TEMPERATURE, OUTPUT

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
ARGS=(--dataset "$DATASET" --split "$SPLIT" --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}")
[[ -n "${MAX_EXAMPLES:-}" ]] && ARGS+=(--max-examples "$MAX_EXAMPLES")
[[ -n "${SAMPLES:-}" ]] && ARGS+=(--samples "$SAMPLES")
[[ -n "${TEMPERATURE:-}" ]] && ARGS+=(--temperature "$TEMPERATURE")
[[ -n "${OUTPUT:-}" ]] && ARGS+=(--output "$OUTPUT")
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.correction_v3_cot "${ARGS[@]}"
