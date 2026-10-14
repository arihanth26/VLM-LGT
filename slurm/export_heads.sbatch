#!/bin/bash
#SBATCH -J vlm-lgt-export
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/export-%j.out

# GPU job: retrains the winning heads of the pooled-mixture sweeps and saves their weights to
# trained_heads/approach1/, with a manifest of reproduced against reported test AUROC.
# Run it after the mixture and mixture_noise sweeps. Usage: sbatch slurm/export_heads.sbatch
# Optional: MODEL_SIZE (2b)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }

srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.export_heads --size "${MODEL_SIZE:-2b}"
