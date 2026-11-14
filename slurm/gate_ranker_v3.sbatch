#!/bin/bash
#SBATCH -J vlm-lgt-gate-v3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o logs/gate-v3-%j.out

# GPU job: gating for the v3 correction step (validation thresholds, per-question decisions, measured
# cost), then the escalation routing study again with the same validation thresholds. Runs the unit
# tests first. Needs results/candidate_ranker_v3_models.pt (slurm/train_ranker_v3.sbatch).
# Writes results/gating_v3.json/.md and results/escalation.json/.md.
# Usage: sbatch slurm/gate_ranker_v3.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m tests.run_tests tests.test_gating tests.test_escalation tests.test_ranker_v3
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.gate_ranker_v3
if compgen -G "experiments/approach1_latent_self_correction/results/escalation/model_*" > /dev/null; then
    srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_escalation
fi
