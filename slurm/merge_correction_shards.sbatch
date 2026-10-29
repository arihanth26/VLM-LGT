#!/bin/bash
#SBATCH -J vlm-lgt-merge-correct
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/merge-correct-%j.out

# CPU job: merges the shard outputs of the full-split correction experiments into one result per
# experiment (records, summary JSON, best-of-five HTML). Run after every shard has finished.
# Usage: sbatch slurm/merge_correction_shards.sbatch
# Optional: EXPERIMENTS (space separated, default "best5 retry deep")

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
read -r -a NAMES <<< "${EXPERIMENTS:-best5 retry deep}"
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.merge_correction_shards --experiments "${NAMES[@]}"
