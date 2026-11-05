#!/bin/bash
# Submits the candidate-search and pre-generation-head study as one dependency chain:
#   scored pools, greedy + 10 samples, for val and test at each temperature (GPU, one job each)
#   train pass-rate labels, 8 samples at T=1.0, correctness only (GPU, sharded)
#   -> candidate_search: grid over temperature, k and selector, chosen on val (GPU node)
#   -> pre-generation head sweep on the constant question embedding (GPU)
# Needs the cached splits and q_features of run_qhead_pipeline.sh and the exported mixture heads.
# Run from the repository root on a login node.
# Usage: [TEMPERATURES="0.7 1.0 1.3"] bash slurm/run_search_pipeline.sh
# Optional: TRAIN_SHARDS (2), TRAIN_SHARDS_CHARTQA (6), POOL_TIME (03:00:00), SKIP_POOLS, SKIP_LABELS, EXTRA_DEPS (a:b)

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
read -r -a TEMPERATURES <<< "${TEMPERATURES:-0.7 1.0 1.3}"
DATASETS=(chartqa_full textvqa docvqa scienceqa_img)
FEATURES=experiments/approach1_latent_self_correction/q_features
mkdir -p logs

missing=0
for DATASET in "${DATASETS[@]}"; do
    [[ -d "data/approach1_latent_self_correction/raw/$DATASET" ]] || { echo "missing cached splits: $DATASET"; missing=1; }
    for SPLIT in train val test; do
        [[ -f "$FEATURES/${DATASET}_${SPLIT}_2b.pt" ]] || { echo "missing question embeddings: $FEATURES/${DATASET}_${SPLIT}_2b.pt"; missing=1; }
    done
done
[[ $missing -eq 0 ]] || exit 2

SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
POOL_JOBS=()
LABEL_JOBS=()
[[ -n "${EXTRA_DEPS:-}" ]] && IFS=':' read -r -a POOL_JOBS <<< "$EXTRA_DEPS"

if [[ -z "${SKIP_POOLS:-}" ]]; then
    for TEMPERATURE in "${TEMPERATURES[@]}"; do
        for SPLIT in val test; do
            for DATASET in "${DATASETS[@]}"; do
                POOL_JOBS+=("$("${SUBMIT[@]}" -t "${POOL_TIME:-03:00:00}" -J "pool-$TEMPERATURE-$SPLIT-$DATASET" \
                    --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT",TEMPERATURE="$TEMPERATURE" slurm/run_candidate_pool.sbatch)")
            done
        done
        echo "T=$TEMPERATURE: scored pools submitted"
    done
fi

if [[ -z "${SKIP_LABELS:-}" ]]; then
    for DATASET in "${DATASETS[@]}"; do
        # About 1 s per question on one GPU: 8,000 train questions take about 2.2 h, ChartQA has 28,299.
        SHARDS="${TRAIN_SHARDS:-2}"
        [[ "$DATASET" == "chartqa_full" ]] && SHARDS="${TRAIN_SHARDS_CHARTQA:-6}"
        for ((SHARD = 0; SHARD < SHARDS; SHARD++)); do
            LABEL_JOBS+=("$("${SUBMIT[@]}" -t 03:00:00 -J "labels-$DATASET-$SHARD" \
                --export=ALL,DATASET="$DATASET",SPLIT=train,TEMPERATURE=1.0,SAMPLES=8,LABELS_ONLY=1,SHARD="$SHARD",NUM_SHARDS="$SHARDS" \
                slurm/run_candidate_pool.sbatch)")
        done
    done
    echo "train labels: ${LABEL_JOBS[*]}"
fi

DEPS=$(IFS=:; echo "${POOL_JOBS[*]}")
SEARCH=$("${SUBMIT[@]}" ${DEPS:+--dependency=afterok:$DEPS} slurm/analyze_candidate_pools.sbatch)
echo "candidate search analysis: $SEARCH"
ALL=$(IFS=:; echo "${LABEL_JOBS[*]}${LABEL_JOBS[*]:+:}$SEARCH")
SWEEP=$("${SUBMIT[@]}" --dependency=afterok:"$ALL" slurm/sweep_pre_head.sbatch)
echo "pre-generation head sweep: $SWEEP"
echo "Results: experiments/approach1_latent_self_correction/results/candidate_search.md and pre_head_sweep.md"
