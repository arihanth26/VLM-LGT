#!/bin/bash
#SBATCH -J vlm-lgt-mixanalyze
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/mixanalyze-%j.out

# CPU job: token-confidence baselines, curves for the winning heads, and the
# clean-versus-corrupted gallery for the mixture experiments.
# Usage: sbatch slurm/analyze_mixture.sbatch
# Optional: MODEL_SIZE (2b), TAG

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

# Keep Hugging Face downloads in scratch, home is only 30 GB.
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.analyze_mixture \
    --size "${MODEL_SIZE:-2b}" --tag "${TAG:-mixture}"
