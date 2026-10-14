#!/bin/bash
# Submits the dataset-mixture experiments as one dependency chain:
#   corrupted-image features for every dataset and split (GPU, in parallel)
#   -> pool everything into the mixture (CPU)
#   -> sweep the three head kinds under the old and the new loss (GPU, in parallel)
#   -> analysis (CPU).
# Needs the clean features and baselines from run_qhead_pipeline.sh for every dataset.
# Run from the repository root on a login node.
# Usage: bash slurm/run_mixture_pipeline.sh [2b|4b]

set -euo pipefail
SIZE="${1:-2b}"
BASE=experiments/approach1_latent_self_correction
DATASETS=(chartqa_full textvqa docvqa scienceqa_img)
SPLITS=(train val test)
mkdir -p logs

# Fail now, with a clear message, if a prerequisite is missing.
missing=0
for DATASET in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        for FILE in "$BASE/q_features/${DATASET}_${SPLIT}_${SIZE}.pt" "$BASE/results/${DATASET}_${SPLIT}_${SIZE}_baseline.jsonl"; do
            if [[ ! -f "$FILE" ]]; then
                echo "missing: $FILE"
                missing=1
            fi
        done
    done
done
if [[ $missing -ne 0 ]]; then
    echo "Run slurm/run_qhead_pipeline.sh for each dataset first."
    exit 2
fi

SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)

CORRUPT_JOBS=()
for DATASET in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        # Only the train splits need hours. A realistic time request keeps the jobs under the
        # per-user cap on requested GPU minutes, so they can all run at the same time.
        LIMIT_TIME="01:00:00"
        [[ "$SPLIT" == "train" ]] && LIMIT_TIME="04:00:00"
        JOB=$("${SUBMIT[@]}" -t "$LIMIT_TIME" --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT",MODEL_SIZE="$SIZE" \
            slurm/extract_corrupt_features.sbatch)
        CORRUPT_JOBS+=("$JOB")
        echo "corrupt features $DATASET $SPLIT: $JOB"
    done
done

DEPS=$(IFS=:; echo "${CORRUPT_JOBS[*]}")
MIXTURE=$("${SUBMIT[@]}" --dependency=afterok:"$DEPS" --export=ALL,MODEL_SIZE="$SIZE" slurm/build_mixture.sbatch)
echo "build mixture: $MIXTURE"

SWEEP_JOBS=()
for KIND in pre verifier unified; do
    JOB=$("${SUBMIT[@]}" -t "${SWEEP_TIME:-05:00:00}" --dependency=afterok:"$MIXTURE" \
        --export=ALL,KIND="$KIND",MODEL_SIZE="$SIZE" slurm/sweep_mixture.sbatch)
    SWEEP_JOBS+=("$JOB")
    echo "sweep $KIND: $JOB"
done

SWEEPS=$(IFS=:; echo "${SWEEP_JOBS[*]}")
ANALYSIS=$("${SUBMIT[@]}" --dependency=afterok:"$SWEEPS" --export=ALL,MODEL_SIZE="$SIZE" slurm/analyze_mixture.sbatch)
echo "analysis: $ANALYSIS"
echo "When all finish: build_mixture_report on a machine with matplotlib (see README)"
echo "then publish_results (see README)."
