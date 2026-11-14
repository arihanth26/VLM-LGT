#!/bin/bash
#SBATCH -J vlm-lgt-esc-analysis
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/escalation-analysis-%j.out

# GPU-node job (the project runs everything on ICE GPU nodes): the routing study of report 13 over the
# escalation answers and the 2B pools; also runs the escalation unit tests first.
# Writes results/escalation.json and .md.
# Usage: sbatch slurm/analyze_escalation.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m tests.run_tests tests.test_escalation
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_escalation
