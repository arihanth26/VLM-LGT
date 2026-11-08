#!/bin/bash
# Extends the validation and test pools to 16 samples at T = 1.0 and 1.3 (16 GPU jobs), then runs the
# ranker analysis for report 11 (GPU). Needs run_search_pipeline.sh and run_correction_learning_pipeline.sh.
# Usage: bash slurm/run_extend_pipeline.sh   Optional: EXTRA (6)

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
JOBS=()
for TEMPERATURE in 1 1.3; do
    for SPLIT in val test; do
        for DATASET in chartqa_full textvqa docvqa scienceqa_img; do
            JOBS+=("$("${SUBMIT[@]}" -J "extend-$TEMPERATURE-$SPLIT-$DATASET" \
                --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT",TEMPERATURE="$TEMPERATURE",EXTRA="${EXTRA:-6}" slurm/run_extend_pools.sbatch)")
        done
    done
done
DEPS=$(IFS=:; echo "${JOBS[*]}")
ANALYSIS=$("${SUBMIT[@]}" --dependency=afterok:"$DEPS" slurm/analyze_ranker.sbatch)
echo "extension jobs: ${JOBS[*]}"
echo "ranker analysis: $ANALYSIS"
