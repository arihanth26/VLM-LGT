#!/bin/bash
#SBATCH -J vlm-lgt-deepen
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/deepen-selected-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

ONLINE_CACHE="${TMPDIR:-/tmp}/vlm-lgt-deepen-${SLURM_JOB_ID}"
mkdir -p "$ONLINE_CACHE"
trap 'rm -rf "$ONLINE_CACHE"' EXIT
export HF_DATASETS_CACHE="$ONLINE_CACHE/datasets"
export HF_HUB_CACHE="$ONLINE_CACHE/hub"
export TRANSFORMERS_CACHE="$ONLINE_CACHE/models"
export XDG_CACHE_HOME="$ONLINE_CACHE/xdg"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(--max-new-tokens "${MAX_NEW_TOKENS:-64}")
if [[ -n "${INPUT:-}" ]]; then
    ARGS+=(--input "$INPUT")
fi
if [[ -n "${OUTPUT:-}" ]]; then
    ARGS+=(--output "$OUTPUT")
fi

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.deepen_selected_candidate "${ARGS[@]}"
