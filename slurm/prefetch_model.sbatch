#!/bin/bash
#SBATCH -J vlm-lgt-prefetch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/prefetch-%j.out

# Downloads the Qwen3-VL weights once, so parallel jobs do not race to fetch them.
# Usage: sbatch slurm/prefetch_model.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
"$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-VL-${MODEL_SIZE_UPPER:-2B}-Instruct'))"
