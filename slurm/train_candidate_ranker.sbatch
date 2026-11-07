#!/bin/bash
#SBATCH -J vlm-lgt-ranker
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/candidate-ranker-%j.out

# GPU job: trains and evaluates the within-question candidate ranker on the encoded candidates of
# run_correction_candidates.sbatch. Writes results/candidate_ranker.json and .md.
# Usage: sbatch slurm/train_candidate_ranker.sbatch   Optional: QUICK, EPOCHS

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
ARGS=()
[[ -n "${QUICK:-}" ]] && ARGS+=(--quick)
[[ -n "${EPOCHS:-}" ]] && ARGS+=(--epochs "$EPOCHS")
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.train_candidate_ranker "${ARGS[@]}"
