#!/bin/bash
#SBATCH -J vlm-lgt-ranker2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o logs/candidate-ranker-v2-%j.out

# GPU job: trains the second candidate ranker (grounded crops, reasoning, self-judge feature, more
# training questions) and compares it with the first on the same candidate sets.
# Writes results/candidate_ranker_v2.json and .md. Usage: sbatch slurm/train_ranker_v2.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.train_ranker_v2
