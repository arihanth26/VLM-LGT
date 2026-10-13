#!/bin/bash
#SBATCH -J vlm-lgt-continual
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/continual-%j.out

# GPU job: trains one head kind through the datasets one after another (several orders,
# old and new loss, with and without replay) and scores every dataset after every stage,
# to measure how much earlier datasets are forgotten. Uses the head configuration of the
# mixture-sweep winners, so run it after slurm/sweep_mixture.sbatch for the same kind.
# Usage: sbatch --export=ALL,KIND=unified slurm/sweep_continual.sbatch
# KIND is pre, verifier, or unified. Optional: MODEL_SIZE (2b), TAG, REPLAY (space separated, default "0 0.1")

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

: "${KIND:?Set KIND to pre, verifier, or unified}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

read -r -a REPLAY_VALUES <<< "${REPLAY:-0 0.1}"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.sweep_continual \
    --kind "$KIND" --size "${MODEL_SIZE:-2b}" --tag "${TAG:-mixture}" --replay "${REPLAY_VALUES[@]}"
