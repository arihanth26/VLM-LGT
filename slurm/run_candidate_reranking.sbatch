#!/bin/bash
#SBATCH -J vlm-lgt-best5
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/candidate-reranking-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

ONLINE_CACHE="${TMPDIR:-/tmp}/vlm-lgt-candidates-${SLURM_JOB_ID}"
mkdir -p "$ONLINE_CACHE"
trap 'rm -rf "$ONLINE_CACHE"' EXIT
export HF_DATASETS_CACHE="$ONLINE_CACHE/datasets"
export HF_HUB_CACHE="$ONLINE_CACHE/hub"
export TRANSFORMERS_CACHE="$ONLINE_CACHE/models"
export XDG_CACHE_HOME="$ONLINE_CACHE/xdg"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(
    --candidates "${CANDIDATES:-5}"
    --temperature "${TEMPERATURE:-0.9}"
    --top-p "${TOP_P:-0.95}"
    --seed "${SEED:-17}"
    --max-new-tokens "${MAX_NEW_TOKENS:-64}"
)
if [[ -n "${LIMIT:-}" ]]; then
    ARGS+=(--limit "$LIMIT")
    if [[ -z "${OUTPUT:-}" ]]; then
        ARGS+=(--output "experiments/approach1_latent_self_correction/results/mixture_test_2b_best_of_five_limit${LIMIT}.jsonl")
    fi
fi
if [[ -n "${OUTPUT:-}" ]]; then
    ARGS+=(--output "$OUTPUT")
fi

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.candidate_reranking "${ARGS[@]}"
