#!/bin/bash
# Submits the escalation study (report 13) as one dependency chain:
#   prefetch the larger model's weights (once, before any GPU job loads it)
#   -> model fix: Qwen3-VL-<SIZE> greedy answers, val and test, every dataset (GPU, sharded)
#   upscale fix: 2B greedy answers on images enlarged 1.5x and 2x (GPU, sharded, no prefetch needed)
#   -> routing analysis over both (GPU node): results/escalation.md
# Needs the T=1 candidate pools (2B greedy answer and detector scores) of run_search_pipeline.sh.
# Run from the repository root on a login node.
# Usage: [SIZE=8b] [NUM_SHARDS=2] [SMOKE=20] bash slurm/run_escalation_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
DATASETS=(chartqa_full textvqa docvqa scienceqa_img)
SIZE="${SIZE:-8b}"
SHARDS="${NUM_SHARDS:-2}"
POOLS=experiments/approach1_latent_self_correction/results/pools/T1
for SPLIT in val test; do
    [[ $(ls "$POOLS/$SPLIT"/*.jsonl 2>/dev/null | wc -l) -eq 4 ]] || { echo "missing T=1 pools for $SPLIT"; exit 2; }
done
mkdir -p logs
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
EXPORT=ALL
[[ -n "${SMOKE:-}" ]] && { EXPORT="ALL,SMOKE=$SMOKE"; SHARDS=1; }

PREFETCH=$("${SUBMIT[@]}" --export=ALL,MODEL_SIZE="$SIZE" slurm/prefetch_model.sbatch)
echo "prefetch $SIZE: $PREFETCH"
JOBS=()
for SPLIT in val test; do
    for DATASET in "${DATASETS[@]}"; do
        for ((SHARD = 0; SHARD < SHARDS; SHARD++)); do
            JOBS+=("$("${SUBMIT[@]}" --dependency=afterok:"$PREFETCH" -J "esc-$SIZE-$SPLIT-$DATASET-$SHARD" \
                --export="$EXPORT",FIX=model,SIZE="$SIZE",DATASET="$DATASET",SPLIT="$SPLIT",SHARD="$SHARD",NUM_SHARDS="$SHARDS" \
                slurm/run_escalation.sbatch)")
            JOBS+=("$("${SUBMIT[@]}" -J "esc-up-$SPLIT-$DATASET-$SHARD" \
                --export="$EXPORT",FIX=upscale,DATASET="$DATASET",SPLIT="$SPLIT",SHARD="$SHARD",NUM_SHARDS="$SHARDS" \
                slurm/run_escalation.sbatch)")
        done
    done
done
echo "answer jobs (${#JOBS[@]}): ${JOBS[*]}"
if [[ -z "${SMOKE:-}" ]]; then
    ANALYSIS=$("${SUBMIT[@]}" --dependency=afterok:"$(IFS=:; echo "${JOBS[*]}")" slurm/analyze_escalation.sbatch)
    echo "routing analysis: $ANALYSIS"
    echo "Result: experiments/approach1_latent_self_correction/results/escalation.md"
fi
