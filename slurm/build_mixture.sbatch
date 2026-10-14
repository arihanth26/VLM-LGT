#!/bin/bash
#SBATCH -J vlm-lgt-mixture
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/mixture-%j.out

# CPU job: pools the cached features of several datasets into one mixture, per split.
# Usage: sbatch slurm/build_mixture.sbatch
# Optional: DATASETS (space separated), MODEL_SIZE (2b), CAP_TRAIN (max train rows per dataset), TAG,
# CORRUPTION (swap), EXTRA_TEST (space separated corruptions to pool as test-only files), EXTRA_ONLY=1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(--size "${MODEL_SIZE:-2b}" --tag "${TAG:-mixture}")
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a NAMES <<< "$DATASETS"
    ARGS+=(--datasets "${NAMES[@]}")
fi
[[ -n "${CAP_TRAIN:-}" ]] && ARGS+=(--cap-train "$CAP_TRAIN")
[[ -n "${CORRUPTION:-}" ]] && ARGS+=(--corruption "$CORRUPTION")
if [[ -n "${EXTRA_TEST:-}" ]]; then
    read -r -a EXTRA_MODES <<< "$EXTRA_TEST"
    ARGS+=(--extra-test "${EXTRA_MODES[@]}")
fi
[[ -n "${EXTRA_ONLY:-}" ]] && ARGS+=(--extra-only)

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.build_mixture "${ARGS[@]}"
