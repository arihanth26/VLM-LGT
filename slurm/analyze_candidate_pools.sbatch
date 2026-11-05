#!/bin/bash
#SBATCH -J vlm-lgt-search
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/candidate-search-%j.out

# GPU-node job (the project runs everything on ICE GPU nodes): evaluates every temperature, pool size and selector of the candidate pools, chooses one
# on validation and reports it on test. Writes results/candidate_search.json and .md.
# Usage: sbatch slurm/analyze_candidate_pools.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_candidate_pools
