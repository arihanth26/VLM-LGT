#!/bin/bash
#SBATCH -J vlm-lgt-cascade
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/cascade-%j.out

# GPU-node job (the project runs everything on ICE GPU nodes): the 2B -> 4B -> 8B cascade against
# routing straight to 8B, over the escalation answers and the 2B pools; runs its unit tests first.
# Writes results/cascade.json and .md.
# Usage: sbatch slurm/analyze_cascade.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m tests.run_tests tests.test_escalation tests.test_cascade
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_cascade
