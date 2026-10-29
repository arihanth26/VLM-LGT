#!/bin/bash
# Submits the correction experiments on the full test splits of all four datasets (9,517 questions)
# as one dependency chain:
#   best-of-five selection (GPU, SHARDS jobs per dataset, in parallel)
#   prompted retry with scored reranking (GPU, SHARDS jobs per dataset, in parallel)
#   deeper pass after best-of-five (GPU, each shard starts when the best-of-five shard it reads is done)
#   -> merge every shard into one result per experiment (CPU).
# Uses the test splits cached by the Q-head pipeline, so it needs run_qhead_pipeline.sh finished for
# every dataset, and the exported pooled-mixture unified head in trained_heads/. Measured speed on one
# L40S is 1 to 2.5 seconds per question per experiment, so two shards per dataset finish in under an hour.
# Run from the repository root on a login node.
# Usage: [SHARDS=2] [EXCLUDE_NODES=atl1-1-03-004-21-0] bash slurm/run_correction_full_pipeline.sh
# To rerun only what failed, list finished or running work in SKIP (DATASET:SHARD:EXPERIMENT ...) and the running job ids in EXTRA_DEPS (a:b:c) so the merge waits for them.
# Optional: SHARD_TIME (02:00:00 for best-of-five and retry), DEEP_TIME (01:30:00), MAX_NEW_TOKENS (64), DATASETS, PARTITION (coc-gpu)

set -euo pipefail
SHARDS="${SHARDS:-2}"
SHARD_TIME="${SHARD_TIME:-02:00:00}"
DEEP_TIME="${DEEP_TIME:-01:30:00}"
HEAD_FILE=trained_heads/approach1/winners/mixture/unified_new_seed1.pt
read -r -a DATASETS <<< "${DATASETS:-chartqa_full textvqa docvqa scienceqa_img}"
mkdir -p logs

missing=0
[[ -f "$HEAD_FILE" ]] || { echo "missing: $HEAD_FILE"; missing=1; }
for DATASET in "${DATASETS[@]}"; do
    [[ -d "data/approach1_latent_self_correction/raw/$DATASET" ]] || { echo "missing cached splits: $DATASET"; missing=1; }
done
[[ $missing -eq 0 ]] || { echo "Run slurm/run_qhead_pipeline.sh and slurm/export_heads.sbatch first."; exit 2; }

SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
# On ICE the scheduler replaces a node exclusion on GPU jobs unless the partition is named, so name it.
GPU_SUBMIT=("${SUBMIT[@]}")
[[ -n "${EXCLUDE_NODES:-}" ]] && GPU_SUBMIT+=(-p "${PARTITION:-coc-gpu}" --exclude="$EXCLUDE_NODES")
EXPORTS="ALL,NUM_SHARDS=$SHARDS,MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}"

# skipped DATASET:SHARD:EXPERIMENT combinations (for rerunning only what failed) are listed in SKIP.
skipped() { [[ " ${SKIP:-} " == *" $1:$2:$3 "* ]]; }

ALL_JOBS=()
[[ -n "${EXTRA_DEPS:-}" ]] && IFS=':' read -r -a ALL_JOBS <<< "$EXTRA_DEPS"
for DATASET in "${DATASETS[@]}"; do
    for ((SHARD = 0; SHARD < SHARDS; SHARD++)); do
        COMMON="$EXPORTS,DATASET=$DATASET,SHARD=$SHARD"
        BEST="" RETRY="" DEEP=""
        if ! skipped "$DATASET" "$SHARD" best5; then
            BEST=$("${GPU_SUBMIT[@]}" -t "$SHARD_TIME" --export="$COMMON,EXPERIMENT=best5" slurm/run_correction_shard.sbatch)
            ALL_JOBS+=("$BEST")
        fi
        if ! skipped "$DATASET" "$SHARD" retry; then
            RETRY=$("${GPU_SUBMIT[@]}" -t "$SHARD_TIME" --export="$COMMON,EXPERIMENT=retry" slurm/run_correction_shard.sbatch)
            ALL_JOBS+=("$RETRY")
        fi
        if ! skipped "$DATASET" "$SHARD" deep; then
            DEPENDENCY=()
            [[ -n "$BEST" ]] && DEPENDENCY=(--dependency=afterok:"$BEST")
            DEEP=$("${GPU_SUBMIT[@]}" -t "$DEEP_TIME" ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} --export="$COMMON,EXPERIMENT=deep" slurm/run_correction_shard.sbatch)
            ALL_JOBS+=("$DEEP")
        fi
        echo "$DATASET shard $SHARD: best5=${BEST:-skipped} retry=${RETRY:-skipped} deep=${DEEP:-skipped}"
    done
done

DEPS=$(IFS=:; echo "${ALL_JOBS[*]}")
MERGE=$("${SUBMIT[@]}" --dependency=afterok:"$DEPS" slurm/merge_correction_shards.sbatch)
echo "merge: $MERGE"
echo "When it finishes: build the complete report with build_approach1_report (needs the merged *_full files in results/)."
