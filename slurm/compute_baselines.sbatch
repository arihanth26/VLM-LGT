#!/bin/bash
#SBATCH -J vlm-lgt-baselines
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o logs/baselines-%j.out

# GPU-node job (the project runs everything on ICE GPU nodes): training-free baselines on the correction
# candidates (self-consistency, reasoning votes, judge and token best, larger models), then the
# escalation routing study again so it includes every larger model answered so far. Tests run first.
# Writes results/baselines.json/.md and results/escalation.json/.md.
# Usage: sbatch slurm/compute_baselines.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m tests.run_tests tests.test_baselines
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.compute_baselines
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_escalation
