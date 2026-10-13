#!/bin/bash
#SBATCH -J vlm-lgt-mixsweep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/mixsweep-%j.out

# GPU job: sweeps one head kind on the dataset mixture under the old loss (ground
# weight 0) and the new loss (ground weight above 0), three seeds per configuration.
# Usage: sbatch --export=ALL,KIND=unified slurm/sweep_mixture.sbatch
# KIND is pre, verifier, or unified. Optional: MODEL_SIZE (2b), TAG, QUICK=1 (smoke test)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

: "${KIND:?Set KIND to pre, verifier, or unified}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(--kind "$KIND" --size "${MODEL_SIZE:-2b}" --tag "${TAG:-mixture}")
[[ -n "${QUICK:-}" ]] && ARGS+=(--quick)

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.sweep_mixture "${ARGS[@]}"
