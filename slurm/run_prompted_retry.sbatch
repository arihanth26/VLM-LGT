#!/bin/bash
#SBATCH -J vlm-lgt-retry
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/prompted-retry-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
# Keep downloaded evaluation datasets and transformed Arrow files off the shared
# repository filesystem. Slurm removes this node-local directory with the job.
ONLINE_CACHE="${TMPDIR:-/tmp}/vlm-lgt-${SLURM_JOB_ID}"
mkdir -p "$ONLINE_CACHE"
trap 'rm -rf "$ONLINE_CACHE"' EXIT
export HF_DATASETS_CACHE="$ONLINE_CACHE/datasets"
export HF_HUB_CACHE="$ONLINE_CACHE/hub"
export TRANSFORMERS_CACHE="$ONLINE_CACHE/models"
export XDG_CACHE_HOME="$ONLINE_CACHE/xdg"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(
    --threshold "${THRESHOLD:-0.5}"
    --review-fraction "${REVIEW_FRACTION:-0.2}"
    --random-seed "${RANDOM_SEED:-0}"
    --max-new-tokens "${MAX_NEW_TOKENS:-64}"
)
if [[ -n "${LIMIT:-}" ]]; then
    ARGS+=(--limit "$LIMIT")
    if [[ -z "${OUTPUT:-}" ]]; then
        ARGS+=(--output "experiments/approach1_latent_self_correction/results/mixture_test_2b_prompted_retry_reranked_limit${LIMIT}.jsonl")
    fi
fi
if [[ -n "${OUTPUT:-}" ]]; then
    ARGS+=(--output "$OUTPUT")
fi

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.online_prompted_retry "${ARGS[@]}"
