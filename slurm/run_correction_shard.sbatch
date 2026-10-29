#!/bin/bash
#SBATCH -J vlm-lgt-correct
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/correct-%j.out

# GPU job: runs one shard of one correction experiment on the full test split of one dataset,
# using the test split cached by the Q-head pipeline (no downloads) and the exported unified head.
# EXPERIMENT is best5 (best-of-five selection), retry (prompted retry with scored reranking), or
# deep (deeper pass after best-of-five, which needs the best5 shard of the same block as INPUT).
# Output goes to experiments/approach1_latent_self_correction/results/full/<experiment>/ and is
# merged by merge_correction_shards.py after every shard has finished. Resumable: rerun a shard
# and it skips the examples it already wrote.
# Usage: sbatch --export=ALL,EXPERIMENT=best5,DATASET=textvqa,SHARD=0,NUM_SHARDS=4 slurm/run_correction_shard.sbatch
# Optional: MAX_NEW_TOKENS (64), SMOKE (cap on examples per shard, for a timing test), HEAD, INPUT

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3

: "${EXPERIMENT:?Set EXPERIMENT to best5, retry, or deep}"
: "${DATASET:?Set DATASET, for example textvqa}"
: "${SHARD:?Set SHARD}"
: "${NUM_SHARDS:?Set NUM_SHARDS}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"

BASE=experiments/approach1_latent_self_correction
OUT_DIR="$BASE/results/full/${SMOKE:+smoke_}$EXPERIMENT"
mkdir -p "$OUT_DIR"
OUTPUT="$OUT_DIR/${DATASET}_shard${SHARD}of${NUM_SHARDS}.jsonl"
HEAD_ARGS=()
[[ -n "${HEAD:-}" ]] && HEAD_ARGS=(--head "$HEAD")

case "$EXPERIMENT" in
    best5)
        ARGS=(--datasets "$DATASET" --shard "$SHARD" --num-shards "$NUM_SHARDS" --cached-splits --no-summary --output "$OUTPUT" --max-new-tokens "${MAX_NEW_TOKENS:-64}")
        [[ -n "${SMOKE:-}" ]] && ARGS+=(--max-examples "$SMOKE")
        srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.candidate_reranking "${ARGS[@]}" "${HEAD_ARGS[@]}"
        ;;
    retry)
        ARGS=(--datasets "$DATASET" --shard "$SHARD" --num-shards "$NUM_SHARDS" --cached-splits --no-summary --output "$OUTPUT" --max-new-tokens "${MAX_NEW_TOKENS:-64}")
        [[ -n "${SMOKE:-}" ]] && ARGS+=(--max-examples "$SMOKE")
        srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.online_prompted_retry "${ARGS[@]}" "${HEAD_ARGS[@]}"
        ;;
    deep)
        INPUT="${INPUT:-$BASE/results/full/${SMOKE:+smoke_}best5/${DATASET}_shard${SHARD}of${NUM_SHARDS}.jsonl}"
        [[ -f "$INPUT" ]] || { echo "Missing best5 shard: $INPUT"; exit 2; }
        srun "$PYTHON_BIN" -m experiments.approach1_latent_self_correction.deepen_selected_candidate \
            --input "$INPUT" --output "$OUTPUT" --cached-splits --no-summary --max-new-tokens "${MAX_NEW_TOKENS:-64}" "${HEAD_ARGS[@]}"
        ;;
    *) echo "Unknown EXPERIMENT '$EXPERIMENT'"; exit 2 ;;
esac
