#!/bin/bash
#SBATCH -J vlm-lgt-prefetch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/prefetch-%j.out

# Downloads the Qwen3-VL weights once, so parallel jobs do not race to fetch them.
# Usage: sbatch slurm/prefetch_model.sbatch   (MODEL_SIZE=4b for the larger model)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
SIZE="${MODEL_SIZE:-2b}"
"$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-VL-${SIZE^^}-Instruct'))"
