#!/bin/bash
#SBATCH -J vlm-lgt-gate-v3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o logs/gate-v3-%j.out

# GPU job: gating for the v3 correction step (validation thresholds, per-question decisions, measured
# cost), with the rankers exactly as trained, then the escalation routing study with the same
# validation thresholds. Runs the unit tests first. Each gating run first checks that its ungated rows
# reproduce the training run's reported accuracy and stops otherwise.
#   1. v3 models (candidate_ranker_v3_models.pt, report candidate_ranker_v3.json)       -> gating_v3.*
#   2. v3 with sampled reasoning (candidate_ranker_v3_multi_models.pt, correction_v3)    -> gating_v3_multi.*
# Needs both model files from slurm/train_ranker_v3.sbatch (the second with RANKER_ARGS=--with-cot-samples).
# With GATE_ARGS set, runs only that one gating, e.g.
#   GATE_ARGS="--with-escalation --tag _multi_esc --models .../candidate_ranker_v3_multi_esc_models.pt --reported .../candidate_ranker_v3_multi_esc.json"
# Usage: sbatch slurm/gate_ranker_v3.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m tests.run_tests tests.test_gating tests.test_escalation tests.test_ranker_v3 tests.test_escalation_candidates tests.test_cascade
RESULTS=experiments/approach1_latent_self_correction/results
if [[ -n "${GATE_ARGS:-}" ]]; then
    srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.gate_ranker_v3 $GATE_ARGS
    exit 0
fi
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.gate_ranker_v3 \
    --models "$RESULTS/candidate_ranker_v3_models.pt" --reported "$RESULTS/candidate_ranker_v3.json"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.gate_ranker_v3 --with-cot-samples --tag _multi \
    --models "$RESULTS/candidate_ranker_v3_multi_models.pt" --reported "$RESULTS/candidate_ranker_v3_multi.json"
if compgen -G "experiments/approach1_latent_self_correction/results/escalation/model_*" > /dev/null; then
    srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_escalation
fi
