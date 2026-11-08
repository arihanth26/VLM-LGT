#!/bin/bash
#SBATCH -J vlm-lgt-rankan
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o logs/ranker-analysis-%j.out

# GPU job: analysis of the trained candidate ranker for report 11 (k curve to 16 samples,
# within-question ranking, attribution, gating, cases). Writes results/ranker_analysis.json.
# Usage: sbatch slurm/analyze_ranker.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_ranker
