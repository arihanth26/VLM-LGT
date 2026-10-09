#!/bin/bash
#SBATCH -J vlm-lgt-setup
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/setup-%j.out

# Create the repository-local Python environment on a CPU compute node.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load anaconda3
export PIP_CACHE_DIR="$SLURM_SUBMIT_DIR/.pip_cache"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Pin the official CUDA 12.8 wheels. Unqualified PyPI installs currently
# select CUDA 13 wheels, which are newer than the ICE GPU driver.
python -m pip install --upgrade --force-reinstall \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

echo "Environment ready at $SLURM_SUBMIT_DIR/.venv"
