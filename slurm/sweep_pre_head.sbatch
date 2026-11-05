#!/bin/bash
#SBATCH -J vlm-lgt-prehead
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/pre-head-sweep-%j.out

# GPU job: sweeps the training of the pre-generation head (targets, ranking term, architecture,
# learning rate, weight decay, dropout) on the constant cached question embedding.
# Needs the train label pools and the val/test pools at T=1.0, and results/candidate_search.json.
# Usage: sbatch slurm/sweep_pre_head.sbatch
# Optional: QUICK, LABEL_TEMPERATURE (1.0), LABEL_SAMPLES (8)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
ARGS=(--label-temperature "${LABEL_TEMPERATURE:-1.0}" --label-samples "${LABEL_SAMPLES:-8}")
[[ -n "${QUICK:-}" ]] && ARGS+=(--quick)
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.sweep_pre_head "${ARGS[@]}"
