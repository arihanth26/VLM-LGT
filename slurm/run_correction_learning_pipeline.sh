#!/bin/bash
# Submits the learned correction step as one dependency chain:
#   val and test: correction candidates (contrastive decoding, zoomed crops) and the encoded state of
#                 every distinct candidate (GPU, sharded per dataset)
#   train:        the same, with corrections only for the first TRAIN_CORRECTED questions of each shard
#                 (a spread subset) and states only for the rest (GPU, sharded); 45 jobs in all,
#                 under the 50-job limit
#   -> the within-question candidate ranker sweep, chosen on validation, read once on test (GPU)
# Needs the candidate pools of run_search_pipeline.sh. Run from the repository root on a login node.
# Usage: bash slurm/run_correction_learning_pipeline.sh
# Optional: EVAL_SHARDS (3), TRAIN_SHARDS (4), TRAIN_CORRECTED (400), CHARTQA_TRAIN_SHARDS (8)

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
DATASETS=(chartqa_full textvqa docvqa scienceqa_img)
EVAL_SHARDS="${EVAL_SHARDS:-3}"
TRAIN_SHARDS="${TRAIN_SHARDS:-4}"
CHARTQA_TRAIN_SHARDS="${CHARTQA_TRAIN_SHARDS:-8}"
TRAIN_CORRECTED="${TRAIN_CORRECTED:-400}"
mkdir -p logs
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
JOBS=()
for DATASET in "${DATASETS[@]}"; do
    for SPLIT in val test; do
        for ((SHARD = 0; SHARD < EVAL_SHARDS; SHARD++)); do
            JOBS+=("$("${SUBMIT[@]}" -J "corr-$SPLIT-$DATASET-$SHARD" \
                --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT",SHARD="$SHARD",NUM_SHARDS="$EVAL_SHARDS" slurm/run_correction_candidates.sbatch)")
        done
    done
    SHARDS="$TRAIN_SHARDS"
    [[ "$DATASET" == "chartqa_full" ]] && SHARDS="$CHARTQA_TRAIN_SHARDS"
    for ((SHARD = 0; SHARD < SHARDS; SHARD++)); do
        JOBS+=("$("${SUBMIT[@]}" -J "corr-train-$DATASET-$SHARD" \
            --export=ALL,DATASET="$DATASET",SPLIT=train,SHARD="$SHARD",NUM_SHARDS="$SHARDS",CORRECTIONS_LIMIT="$TRAIN_CORRECTED" \
            slurm/run_correction_candidates.sbatch)")
    done
    echo "$DATASET: submitted"
done
DEPS=$(IFS=:; echo "${JOBS[*]}")
RANKER=$("${SUBMIT[@]}" --dependency=afterok:"$DEPS" slurm/train_candidate_ranker.sbatch)
echo "${#JOBS[@]} candidate jobs; ranker: $RANKER"
echo "Result: experiments/approach1_latent_self_correction/results/candidate_ranker.md"
