#!/bin/bash
#SBATCH -J vlm-lgt-escalate
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/escalation-%j.out

# GPU job: one shard of second-opinion answers for the escalation study (report 13).
#   FIX=model    greedy answer of a larger Qwen3-VL (SIZE, default 8b)
#   FIX=upscale  greedy 2B answer on the image enlarged by each of SCALES (default "1.5 2")
# Resumable. Output: experiments/approach1_latent_self_correction/results/escalation/<fix>/<split>/.
# Usage: sbatch --export=ALL,FIX=model,DATASET=textvqa,SPLIT=val slurm/run_escalation.sbatch
# Optional: SIZE, SCALES, SHARD (0), NUM_SHARDS (1), SMOKE (examples per shard)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

: "${FIX:?Set FIX to model or upscale}"
: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT to val or test}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
read -r -a SCALE_LIST <<< "${SCALES:-1.5 2}"
ARGS=(--fix "$FIX" --size "${SIZE:-8b}" --scales "${SCALE_LIST[@]}" --dataset "$DATASET" --split "$SPLIT"
      --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}")
if [[ -n "${SMOKE:-}" ]]; then
    ARGS+=(--max-examples "$SMOKE" --output "experiments/approach1_latent_self_correction/results/escalation_smoke/${FIX}_${SPLIT}_${DATASET}.jsonl")
fi
srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.escalation "${ARGS[@]}"
