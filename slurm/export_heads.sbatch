#!/bin/bash
#SBATCH -J vlm-lgt-export
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/export-%j.out

# GPU job: retrains and saves the trained heads of one tag to trained_heads/approach1/.
# MODE=winners saves every winning configuration of the sweep, on all three seeds.
# MODE=sequential saves the final head of each sequential-training variant and order for one KIND.
# Run it after the sweeps for the tag. Usage:
#   sbatch --export=ALL,MODE=winners,TAG=mixture slurm/export_heads.sbatch
#   sbatch --export=ALL,MODE=sequential,TAG=mixture_noise,KIND=unified slurm/export_heads.sbatch
# Optional: MODEL_SIZE (2b)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

: "${MODE:?Set MODE to winners or sequential}"
: "${TAG:?Set TAG, for example mixture or textvqa_noise}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

ARGS=(--mode "$MODE" --tag "$TAG" --size "${MODEL_SIZE:-2b}")
[[ -n "${KIND:-}" ]] && ARGS+=(--kind "$KIND")

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.export_heads "${ARGS[@]}"
