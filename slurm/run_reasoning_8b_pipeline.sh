#!/bin/bash
# The 8B step-by-step reasoning answer as a candidate of the v3 ranker, as one dependency chain:
#   8B reasoning + candidate encoding, one job per dataset and split (train subset, val, test)
#   -> ranker v3 retrained with sampled reasoning, the 8B and upscaled answers, and the 8B reasoning
#   -> gating of that ranker, with the reproduction check
# Needs correction_v2, correction_v3 and correction_v4 in place (AFTER: the job that finishes correction_v4).
# Run from the repository root on a login node.  Usage: [AFTER=<job id>] bash slurm/run_reasoning_8b_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
RESULTS=experiments/approach1_latent_self_correction/results
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
JOBS=()
for D in chartqa_full textvqa docvqa scienceqa_img; do
    for SPLIT in train val test; do
        JOBS+=("$("${SUBMIT[@]}" -J "cot8b-$SPLIT-$D" --export=ALL,DATASET="$D",SPLIT="$SPLIT" slurm/run_reasoning_8b.sbatch)")
    done
done
DEPS="afterok:$(IFS=:; echo "${JOBS[*]}")"; [[ -n "${AFTER:-}" ]] && DEPS="$DEPS,afterany:$AFTER"
RANKER=$("${SUBMIT[@]}" --dependency="$DEPS" -J ranker-v3-cot8b \
    --export=ALL,RANKER_ARGS="--with-reasoning-8b --variants scalars_ds scalars_nods" slurm/train_ranker_v3.sbatch)
GATE=$("${SUBMIT[@]}" --dependency=afterok:"$RANKER" -J gate-v3-cot8b \
    --export=ALL,GATE_ARGS="--with-reasoning-8b --tag _multi_esc_cot8b --models $RESULTS/candidate_ranker_v3_multi_esc_cot8b_models.pt --reported $RESULTS/candidate_ranker_v3_multi_esc_cot8b.json" \
    slurm/gate_ranker_v3.sbatch)
echo "8B reasoning jobs: ${JOBS[*]}"
echo "ranker: $RANKER; gating: $GATE -> $RESULTS/gating_v3_multi_esc_cot8b.md"
