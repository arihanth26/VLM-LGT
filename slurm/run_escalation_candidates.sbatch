#!/bin/bash
#SBATCH -J vlm-lgt-esc-cand
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/escalation-candidates-%j.out

# GPU job: encodes the 8B and upscaled answers of one dataset split as v3 ranker candidates (2B answer
# state, confidence, head and probe scores, judge). Resumable.
# Output: experiments/approach1_latent_self_correction/results/correction_v4/<split>/.
# Usage: sbatch --export=ALL,DATASET=textvqa,SPLIT=val slurm/run_escalation_candidates.sbatch
# Optional: SHARD (0), NUM_SHARDS (1), MAX_EXAMPLES

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
ARGS=(--dataset "$DATASET" --split "$SPLIT" --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}")
[[ -n "${MAX_EXAMPLES:-}" ]] && ARGS+=(--max-examples "$MAX_EXAMPLES")
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.escalation_candidates "${ARGS[@]}"
