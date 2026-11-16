#!/bin/bash
# The 8B and upscaled answers as candidates of the v3 ranker, gated, as one dependency chain:
#   8B and upscaled answers on the ranker's train subset (GPU, sharded; val and test exist already)
#   -> encode every new answer as a ranker candidate: train, val, test (GPU, one job per dataset split)
#   -> retrain ranker v3 with sampled reasoning + escalation candidates (GPU)
#   -> gate it, with the reproduction check (GPU)
# Needs results/escalation (val, test), correction_v2 and correction_v3 in place. Run from the repository
# root on a login node. Usage: bash slurm/run_escalation_ranker_pipeline.sh
# Optional: SKIP_TRAIN_ANSWERS (if the train answers exist), MODEL_SHARDS (2), UPSCALE_SHARDS (3)

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
DATASETS=(chartqa_full textvqa docvqa scienceqa_img)
RESULTS=experiments/approach1_latent_self_correction/results
mkdir -p logs
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
CANDIDATES=()
for DATASET in "${DATASETS[@]}"; do
    ANSWERS=()
    if [[ -z "${SKIP_TRAIN_ANSWERS:-}" ]]; then
        for ((S = 0; S < ${MODEL_SHARDS:-2}; S++)); do
            ANSWERS+=("$("${SUBMIT[@]}" -J "esc-8b-train-$DATASET-$S" --export=ALL,FIX=model,SIZE=8b,SUBSET=1,DATASET="$DATASET",SPLIT=train,SHARD="$S",NUM_SHARDS="${MODEL_SHARDS:-2}" slurm/run_escalation.sbatch)")
        done
        for ((S = 0; S < ${UPSCALE_SHARDS:-3}; S++)); do
            ANSWERS+=("$("${SUBMIT[@]}" -J "esc-up-train-$DATASET-$S" --export=ALL,FIX=upscale,SUBSET=1,DATASET="$DATASET",SPLIT=train,SHARD="$S",NUM_SHARDS="${UPSCALE_SHARDS:-3}" slurm/run_escalation.sbatch)")
        done
    fi
    DEP=""
    [[ ${#ANSWERS[@]} -gt 0 ]] && DEP="--dependency=afterok:$(IFS=:; echo "${ANSWERS[*]}")"
    CANDIDATES+=("$("${SUBMIT[@]}" $DEP -J "esc-cand-train-$DATASET" --export=ALL,DATASET="$DATASET",SPLIT=train slurm/run_escalation_candidates.sbatch)")
    for SPLIT in val test; do
        CANDIDATES+=("$("${SUBMIT[@]}" -J "esc-cand-$SPLIT-$DATASET" --export=ALL,DATASET="$DATASET",SPLIT="$SPLIT" slurm/run_escalation_candidates.sbatch)")
    done
    echo "$DATASET: train answers ${ANSWERS[*]:-skipped}; candidates submitted"
done
RANKER=$("${SUBMIT[@]}" --dependency=afterok:"$(IFS=:; echo "${CANDIDATES[*]}")" \
    --export=ALL,RANKER_ARGS="--with-escalation --variants scalars_ds scalars_nods full_ds" slurm/train_ranker_v3.sbatch)
GATE=$("${SUBMIT[@]}" --dependency=afterok:"$RANKER" \
    --export=ALL,GATE_ARGS="--with-escalation --tag _multi_esc --models $RESULTS/candidate_ranker_v3_multi_esc_models.pt --reported $RESULTS/candidate_ranker_v3_multi_esc.json" \
    slurm/gate_ranker_v3.sbatch)
echo "candidate jobs: ${CANDIDATES[*]}"
echo "ranker v3 + escalation: $RANKER; gating: $GATE"
echo "Results: $RESULTS/candidate_ranker_v3_multi_esc.md and $RESULTS/gating_v3_multi_esc.md"
