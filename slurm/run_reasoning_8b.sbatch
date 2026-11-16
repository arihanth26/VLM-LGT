#!/bin/bash
#SBATCH -J vlm-lgt-cot8b
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=96G
#SBATCH -t 10:00:00
#SBATCH -o logs/reasoning-8b-%j.out

# GPU job: the 8B step-by-step reasoning answer of one dataset split (escalation.py --fix reasoning; the
# ranker's train subset with SPLIT=train), then the same answers encoded by the 2B model as ranker
# candidates "cot_8b" (escalation_candidates.py --sources cot_8b --out-dir correction_v5). Both resumable.
# Usage: sbatch --export=ALL,DATASET=chartqa_full,SPLIT=test slurm/run_reasoning_8b.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
M=experiments.approach1_latent_self_correction
SUB=(); [[ "$SPLIT" == "train" || "$SPLIT" == "unseen" ]] && SUB=(--subset)
srun "$PYTHON_BIN" -m $M.escalation --fix reasoning --size 8b --dataset "$DATASET" --split "$SPLIT" "${SUB[@]}"
srun "$PYTHON_BIN" -m $M.escalation_candidates --dataset "$DATASET" --split "$SPLIT" --sources cot_8b --out-dir correction_v5
