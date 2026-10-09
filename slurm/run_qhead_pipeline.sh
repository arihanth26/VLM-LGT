#!/bin/bash
# Submits the whole Q-head pipeline for one dataset as a dependency chain:
# build splits, baseline and feature extraction for train/val/test, then both
# sweeps. Run from the repository root on a login node.
# Usage: bash slurm/run_qhead_pipeline.sh textvqa [2b]

set -euo pipefail
DATASET="${1:?Usage: bash slurm/run_qhead_pipeline.sh <textvqa|docvqa|scienceqa_img|chartqa_full> [2b|4b]}"
SIZE="${2:-2b}"
BASE=data/approach1_latent_self_correction
mkdir -p logs

# Q-head datasets are built from the Hub by the build job. chartqa_full is a plain
# download of the official splits, so its first job is the download job.
if [[ "$DATASET" == "chartqa_full" ]]; then
    BUILD=$(sbatch --parsable --export=ALL,DATASET=chartqa_full,DATA_GROUP=training slurm/download_approach1.sbatch)
    echo "download: $BUILD"
else
    BUILD=$(sbatch --parsable --export=ALL,DATASET="$DATASET" slurm/build_qhead_splits.sbatch)
    echo "build splits: $BUILD"
fi
BUILD_DEP="--dependency=afterok:$BUILD"

GROUP=qhead
[[ "$DATASET" == "chartqa_full" ]] && GROUP=training

FEATURE_JOBS=()
for SPLIT in train val test; do
    BASELINE_FILE="$BASE/results/${DATASET}_${SPLIT}_${SIZE}_baseline.jsonl"
    BASELINE=$(sbatch --parsable $BUILD_DEP \
        --export=ALL,DATASET="$DATASET",DATA_GROUP="$GROUP",SPLIT="$SPLIT",MODEL_SIZE="$SIZE",MAX_NEW_TOKENS=64 \
        slurm/run_baseline.sbatch)
    FEATURES=$(sbatch --parsable --dependency=afterok:"$BASELINE" \
        --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT",BASELINE="$BASELINE_FILE",MODEL_SIZE="$SIZE" \
        slurm/extract_q_features.sbatch)
    FEATURE_JOBS+=("$FEATURES")
    echo "$SPLIT: baseline $BASELINE, features $FEATURES"
done

DEPS=$(IFS=:; echo "${FEATURE_JOBS[*]}")
SEPARATE=$(sbatch --parsable --dependency=afterok:"$DEPS" \
    --export=ALL,DATASET="$DATASET",MODEL_SIZE="$SIZE" slurm/sweep_q_heads.sbatch)
UNIFIED=$(sbatch --parsable --dependency=afterok:"$DEPS" \
    --export=ALL,DATASET="$DATASET",MODEL_SIZE="$SIZE" slurm/sweep_unified_q_head.sbatch)
ANALYSIS=$(sbatch --parsable --dependency=afterok:"$SEPARATE":"$UNIFIED"     --export=ALL,DATASET="$DATASET",MODEL_SIZE="$SIZE" slurm/analyze_q_heads.sbatch)
echo "separate sweep: $SEPARATE, unified sweep: $UNIFIED, analysis: $ANALYSIS"
echo "When all finish: python -m data.approach1_latent_self_correction.compare_q_heads_report"
echo "then build_report and publish_results (see README)."
