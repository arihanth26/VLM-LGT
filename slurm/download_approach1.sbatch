#!/bin/bash
#SBATCH -J vlm-lgt-data
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/data-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Run: sbatch slurm/setup_environment.sbatch"
    exit 2
fi

ARGS=(--groups "${DATA_GROUP:-training}")
if [[ -n "${DATASET:-}" ]]; then
    ARGS+=(--datasets "$DATASET")
fi

srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.download "${ARGS[@]}"
