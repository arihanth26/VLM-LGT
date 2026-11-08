#!/bin/bash
# Submits the second round of correction experiments in parallel, then the second ranker:
#   val and test: grounded crops, reasoning and the self-judge for every question (GPU, sharded)
#   train: the same plus contrastive decoding and corner crops for TRAIN_PER_DATASET questions per
#          dataset, spread over TRAIN_SHARDS shards (GPU)
#   -> train_ranker_v2 (GPU), chosen on validation, read once on test
# Needs run_correction_learning_pipeline.sh finished. Run from the repository root on a login node.
# Usage: bash slurm/run_correction_v2_pipeline.sh
# Optional: TEST_SHARDS (3), VAL_SHARDS (2), TRAIN_SHARDS (2), TRAIN_PER_DATASET (3000)

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
TEST_SHARDS="${TEST_SHARDS:-3}"
VAL_SHARDS="${VAL_SHARDS:-2}"
TRAIN_SHARDS="${TRAIN_SHARDS:-2}"
PER_SHARD=$(( ${TRAIN_PER_DATASET:-3000} / TRAIN_SHARDS ))
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
JOBS=()
for DATASET in chartqa_full textvqa docvqa scienceqa_img; do
    for ((SHARD = 0; SHARD < TEST_SHARDS; SHARD++)); do
        JOBS+=("$("${SUBMIT[@]}" -J "c2-test-$DATASET-$SHARD" --export=ALL,DATASET="$DATASET",SPLIT=test,SHARD="$SHARD",NUM_SHARDS="$TEST_SHARDS" slurm/run_correction_v2.sbatch)")
    done
    for ((SHARD = 0; SHARD < VAL_SHARDS; SHARD++)); do
        JOBS+=("$("${SUBMIT[@]}" -J "c2-val-$DATASET-$SHARD" --export=ALL,DATASET="$DATASET",SPLIT=val,SHARD="$SHARD",NUM_SHARDS="$VAL_SHARDS" slurm/run_correction_v2.sbatch)")
    done
    for ((SHARD = 0; SHARD < TRAIN_SHARDS; SHARD++)); do
        JOBS+=("$("${SUBMIT[@]}" -J "c2-train-$DATASET-$SHARD" --export=ALL,DATASET="$DATASET",SPLIT=train,SHARD="$SHARD",NUM_SHARDS="$TRAIN_SHARDS",MAX_EXAMPLES="$PER_SHARD",OLD_CORRECTIONS=1 slurm/run_correction_v2.sbatch)")
    done
    echo "$DATASET: submitted"
done
DEPS=$(IFS=:; echo "${JOBS[*]}")
RANKER=$("${SUBMIT[@]}" --dependency=afterok:"$DEPS" slurm/train_ranker_v2.sbatch)
echo "${#JOBS[@]} candidate jobs; ranker v2: $RANKER"
