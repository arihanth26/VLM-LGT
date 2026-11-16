#!/bin/bash
# Submits the sampled-reasoning round (correction_v3_cot.py) for every question correction_v2 covered
# (train subset, val, test), sharded, then retrains ranker v3 with the samples once every shard succeeds.
# Needs results/correction_v2 in place. Run from the repository root on a login node.
# Usage: bash slurm/run_correction_v3_pipeline.sh
# Optional: SHARDS (6 per dataset and split), PARTITION (ice-gpu), SAMPLES, TEMPERATURE

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
SHARDS="${SHARDS:-6}"
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes -p "${PARTITION:-ice-gpu}")
JOBS=()
for DATASET in chartqa_full textvqa docvqa scienceqa_img; do
    for SPLIT in test val train; do
        for ((SHARD = 0; SHARD < SHARDS; SHARD++)); do
            JOBS+=("$("${SUBMIT[@]}" -J "cot3-$SPLIT-$DATASET-$SHARD" --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT",SHARD="$SHARD",NUM_SHARDS="$SHARDS" slurm/run_correction_v3_cot.sbatch)")
        done
    done
    echo "$DATASET: submitted"
done
DEPS=$(IFS=:; echo "${JOBS[*]}")
RANKER=$("${SUBMIT[@]}" --dependency=afterok:"$DEPS" --export=ALL,RANKER_ARGS="--with-cot-samples --variants full_ds scalars_ds full_nods scalars_nods" slurm/train_ranker_v3.sbatch)
echo "${#JOBS[@]} reasoning jobs; ranker v3 with samples: $RANKER"
