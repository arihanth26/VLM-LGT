#!/bin/bash
# Submits the corrupted-image (noise) version of the new loss as one dependency chain:
#   noise features for every dataset and split, plus mask and shuffle features for the test
#   splits so grounding can be checked against other corruptions (GPU, in parallel)
#   -> pooled mixture and per-dataset builds, for noise and for the swap baseline (CPU)
#   -> head sweeps for the three head kinds on every build (GPU, in parallel)
#   -> sequential (forgetting) runs on the noise mixture and the mixture analysis.
# Needs the clean features, baselines and swap features from run_mixture_pipeline.sh.
# If the extraction jobs were already submitted, pass their ids in EXTRACT_JOBS (colon separated)
# and nothing is extracted again.
# Run from the repository root on a login node.
# The cluster allows 50 submitted jobs per user, so if the submission stops at that limit, run it
# again later with SKIP_POOLED=1 and SINGLES="names not yet queued", for example SINGLES="scienceqa".
# Usage: [EXTRACT_JOBS=id:id:...] [SKIP_POOLED=1] [SINGLES="chartqa textvqa"] bash slurm/run_corruption_pipeline.sh [2b|4b]

set -euo pipefail
SIZE="${1:-2b}"
BASE=experiments/approach1_latent_self_correction
DATASETS=(chartqa_full textvqa docvqa scienceqa_img)
SHORT=(chartqa textvqa docvqa scienceqa)
mkdir -p logs

# Fail now, with a clear message, if the clean or swap features are missing.
missing=0
for DATASET in "${DATASETS[@]}"; do
    for SPLIT in train val test; do
        for FILE in "$BASE/q_features/${DATASET}_${SPLIT}_${SIZE}.pt" "$BASE/q_features/${DATASET}_${SPLIT}_${SIZE}_corrupt.pt"; do
            [[ -f "$FILE" ]] || { echo "missing: $FILE"; missing=1; }
        done
    done
done
[[ $missing -eq 0 ]] || { echo "Run slurm/run_mixture_pipeline.sh first."; exit 2; }

SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)

if [[ -n "${SKIP_POOLED:-}" ]]; then
    DEPS=""
elif [[ -n "${EXTRACT_JOBS:-}" ]]; then
    DEPS="$EXTRACT_JOBS"
else
    JOBS=()
    for DATASET in "${DATASETS[@]}"; do
        for SPLIT in train val test; do
            LIMIT_TIME="01:00:00"
            [[ "$SPLIT" == "train" ]] && LIMIT_TIME="04:00:00"
            JOBS+=("$("${SUBMIT[@]}" -t "$LIMIT_TIME" \
                --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT",CORRUPTION=noise,MODEL_SIZE="$SIZE" slurm/extract_corrupt_features.sbatch)")
        done
        for MODE in mask shuffle; do
            JOBS+=("$("${SUBMIT[@]}" -t 01:00:00 \
                --export=ALL,DATASET="$DATASET",SPLIT=test,CORRUPTION="$MODE",MODEL_SIZE="$SIZE" slurm/extract_corrupt_features.sbatch)")
        done
    done
    DEPS=$(IFS=:; echo "${JOBS[*]}")
fi

# build TAG CORRUPTION EXTRA_TEST DATASET-or-empty [extra-only]
build() {
    local tag="$1" corruption="$2" extra="$3" datasets="$4" only="${5:-}"
    local dependency=()
    [[ -n "$DEPS" ]] && dependency=(--dependency=afterok:"$DEPS")
    "${SUBMIT[@]}" ${dependency[@]+"${dependency[@]}"} \
        --export=ALL,MODEL_SIZE="$SIZE",TAG="$tag",CORRUPTION="$corruption",EXTRA_TEST="$extra",DATASETS="$datasets",EXTRA_ONLY="$only" \
        slurm/build_mixture.sbatch
}

# sweeps TAG OUT_TAG TRAIN_CORRUPTION EXTRA TIME BUILD_JOB: one sweep per head kind, prints the job ids
sweeps() {
    local tag="$1" out="$2" corruption="$3" extra="$4" limit="$5" build_job="$6" ids=()
    for KIND in pre verifier unified; do
        ids+=("$("${SUBMIT[@]}" -t "$limit" --dependency=afterok:"$build_job" \
            --export=ALL,KIND="$KIND",MODEL_SIZE="$SIZE",TAG="$tag",OUT_TAG="$out",TRAIN_CORRUPTION="$corruption",EXTRA="$extra" \
            slurm/sweep_mixture.sbatch)")
    done
    echo "${ids[*]}"
}

if [[ -z "${SKIP_POOLED:-}" ]]; then
    # Pooled mixture trained with noise, graded against swap, mask and shuffle too.
    NOISE_BUILD=$(build mixture_noise noise "swap mask shuffle" "")
    echo "build mixture_noise: $NOISE_BUILD"
    read -r -a NOISE_SWEEPS <<< "$(sweeps mixture_noise mixture_noise noise "swap mask shuffle" "${SWEEP_TIME:-05:00:00}" "$NOISE_BUILD")"
    echo "sweeps mixture_noise (pre verifier unified): ${NOISE_SWEEPS[*]}"

    # The swap mixture again, only to grade its heads against noise, mask and shuffle (written as mixture_xeval).
    XEVAL_BUILD=$(build mixture swap "noise mask shuffle" "" 1)
    echo "extra test files for the swap mixture: $XEVAL_BUILD"
    read -r -a XEVAL_SWEEPS <<< "$(sweeps mixture mixture_xeval swap "noise mask shuffle" "${SWEEP_TIME:-05:00:00}" "$XEVAL_BUILD")"
    echo "sweeps mixture_xeval (pre verifier unified): ${XEVAL_SWEEPS[*]}"

    # Sequential runs on the noise mixture, after its own sweeps give the head settings.
    CONTINUAL=()
    INDEX=0
    for KIND in pre verifier unified; do
        JOB=$("${SUBMIT[@]}" -t "${CONTINUAL_TIME:-04:00:00}" --dependency=afterok:"${NOISE_SWEEPS[$INDEX]}" \
            --export=ALL,KIND="$KIND",MODEL_SIZE="$SIZE",TAG=mixture_noise slurm/sweep_continual.sbatch)
        CONTINUAL+=("$JOB")
        echo "sequential $KIND: $JOB"
        INDEX=$((INDEX + 1))
    done

fi

# Each dataset alone, with swap and with noise.
for INDEX in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$INDEX]}"
    NAME="${SHORT[$INDEX]}"
    if [[ -n "${SINGLES:-}" && " $SINGLES " != *" $NAME "* ]]; then continue; fi
    LIMIT_TIME="01:30:00"
    [[ "$NAME" == "chartqa" ]] && LIMIT_TIME="03:00:00"
    SWAP_BUILD=$(build "${NAME}_swap" swap "noise mask shuffle" "$DATASET")
    NOISE_ONE=$(build "${NAME}_noise" noise "swap mask shuffle" "$DATASET")
    sweeps "${NAME}_swap" "${NAME}_swap" swap "noise mask shuffle" "$LIMIT_TIME" "$SWAP_BUILD" > /dev/null
    sweeps "${NAME}_noise" "${NAME}_noise" noise "swap mask shuffle" "$LIMIT_TIME" "$NOISE_ONE" > /dev/null
    echo "single dataset $NAME: builds $SWAP_BUILD $NOISE_ONE, six sweeps queued"
done

if [[ -z "${SKIP_POOLED:-}" ]]; then
    ALL=$(IFS=:; echo "${NOISE_SWEEPS[*]}")
    ANALYSIS=$("${SUBMIT[@]}" --dependency=afterok:"$ALL" --export=ALL,MODEL_SIZE="$SIZE",TAG=mixture_noise,CORRUPTION=noise slurm/analyze_mixture.sbatch)
    echo "analysis: $ANALYSIS"
fi
