#!/bin/bash
#SBATCH -J vlm-lgt-ranker3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o logs/candidate-ranker-v3-%j.out

# GPU job: trains the third candidate ranker (reasoning + crops in one ranker, source and dataset
# features, per-dataset router) and compares it with v1 and v2 on the same candidate sets.
# Writes results/candidate_ranker_v3.json and .md (with RANKER_ARGS="--with-cot-samples": candidate_ranker_v3_multi.*).
# Usage: sbatch slurm/train_ranker_v3.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.train_ranker_v3 ${RANKER_ARGS:-}
