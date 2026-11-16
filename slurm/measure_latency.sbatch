#!/bin/bash
#SBATCH -J vlm-lgt-latency
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H100"
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/latency-%j.out

# GPU job: wall-clock seconds of every pipeline component (measure_latency.py), on one H100 so every
# component is timed on the same hardware. Writes results/latency.json.
# Usage: sbatch slurm/measure_latency.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.measure_latency
