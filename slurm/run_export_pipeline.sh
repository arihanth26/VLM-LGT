#!/bin/bash
# Submits the jobs that save every trained head: the winners of every sweep (pooled mixture, noise
# mixture, and each dataset alone with swap and with noise) on three seeds, and the final heads of
# the sequential runs on the two mixtures. Needs the sweeps and cached features from the earlier
# pipelines. Run from the repository root on a login node.
# Usage: bash slurm/run_export_pipeline.sh [2b|4b]

set -euo pipefail
SIZE="${1:-2b}"
mkdir -p logs
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)

for TAG in mixture mixture_noise; do
    echo "winners $TAG: $("${SUBMIT[@]}" -t 04:00:00 --export=ALL,MODE=winners,TAG="$TAG",MODEL_SIZE="$SIZE" slurm/export_heads.sbatch)"
    for KIND in pre verifier unified; do
        echo "sequential $TAG $KIND: $("${SUBMIT[@]}" -t 02:00:00 --export=ALL,MODE=sequential,TAG="$TAG",KIND="$KIND",MODEL_SIZE="$SIZE" slurm/export_heads.sbatch)"
    done
done
for DATASET in chartqa textvqa docvqa scienceqa; do
    for CORRUPTION in swap noise; do
        LIMIT_TIME="01:30:00"
        [[ "$DATASET" == "chartqa" ]] && LIMIT_TIME="03:00:00"
        echo "winners ${DATASET}_${CORRUPTION}: $("${SUBMIT[@]}" -t "$LIMIT_TIME" --export=ALL,MODE=winners,TAG="${DATASET}_${CORRUPTION}",MODEL_SIZE="$SIZE" slurm/export_heads.sbatch)"
    done
done
