#!/bin/bash
# Transfer test on datasets the pipeline never trained on (AI2D, InfographicVQA), as one dependency chain.
# Every step reuses the in-domain job scripts on the split "unseen"; nothing is trained:
#   build the datasets -> question embeddings -> T=1.0 pools (greedy + 10 samples, head scores)
#   -> first-round corrections and encodings -> second round (grounded crops, reasoning, judge)
#   -> sampled reasoning;  in parallel: 8B answers -> 8B answers as ranker candidates
#   -> evaluate the frozen pipeline and the baselines (evaluate_unseen.py)
# Run from the repository root on a login node, after the datasets are built (STEP=build) or with BUILD=1.
# Usage: [BUILD=1] [RANKER_JOB=<id of the escalation ranker job>] bash slurm/run_unseen_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
SHARDS="${SHARDS:-2}"
dep() { local IFS=:; echo "--dependency=afterok:$*"; }
BASE=()
if [[ -n "${BUILD:-}" ]]; then BASE=("$("${SUBMIT[@]}" -J unseen-build --export=ALL,STEP=build slurm/run_unseen_step.sbatch)"); fi
EVAL_DEPS=()
for D in ai2d infovqa; do
    DB=(); [[ ${#BASE[@]} -gt 0 ]] && DB=("$(dep "${BASE[@]}")")
    F=$("${SUBMIT[@]}" "${DB[@]}" -J "unseen-feat-$D" --export=ALL,STEP=features,DATASET="$D" slurm/run_unseen_step.sbatch)
    P=(); C=(); V2=(); V3=()
    for ((S = 0; S < SHARDS; S++)); do
        P+=("$("${SUBMIT[@]}" "$(dep "$F")" -J "unseen-pool-$D-$S" --export=ALL,DATASET="$D",SPLIT=unseen,TEMPERATURE=1.0,SHARD="$S",NUM_SHARDS="$SHARDS" slurm/run_candidate_pool.sbatch)")
    done
    for ((S = 0; S < SHARDS; S++)); do
        C+=("$("${SUBMIT[@]}" "$(dep "${P[@]}")" -J "unseen-corr-$D-$S" --export=ALL,DATASET="$D",SPLIT=unseen,SHARD="$S",NUM_SHARDS="$SHARDS" slurm/run_correction_candidates.sbatch)")
    done
    for ((S = 0; S < SHARDS; S++)); do
        V2+=("$("${SUBMIT[@]}" "$(dep "${C[@]}")" -J "unseen-v2-$D-$S" --export=ALL,DATASET="$D",SPLIT=unseen,SHARD="$S",NUM_SHARDS="$SHARDS" slurm/run_correction_v2.sbatch)")
    done
    for ((S = 0; S < SHARDS; S++)); do
        V3+=("$("${SUBMIT[@]}" "$(dep "${V2[@]}")" -J "unseen-v3-$D-$S" --export=ALL,DATASET="$D",SPLIT=unseen,SHARD="$S",NUM_SHARDS="$SHARDS" slurm/run_correction_v3_cot.sbatch)")
    done
    E8=$("${SUBMIT[@]}" "${DB[@]}" -J "unseen-8b-$D" --export=ALL,FIX=model,SIZE=8b,DATASET="$D",SPLIT=unseen slurm/run_escalation.sbatch)
    EC=$("${SUBMIT[@]}" "$(dep "${V3[@]}" "$E8")" -J "unseen-esccand-$D" --export=ALL,DATASET="$D",SPLIT=unseen slurm/run_escalation_candidates.sbatch)
    EVAL_DEPS+=("$EC")
    echo "$D: features $F, pools ${P[*]}, corrections ${C[*]}, round 2 ${V2[*]}, sampled reasoning ${V3[*]}, 8B $E8, 8B candidates $EC"
done
EXTRA=""; [[ -n "${RANKER_JOB:-}" ]] && EXTRA=",afterany:$RANKER_JOB"
EVAL=$("${SUBMIT[@]}" --dependency=afterok:"$(IFS=:; echo "${EVAL_DEPS[*]}")$EXTRA" -J unseen-evaluate --export=ALL,STEP=evaluate slurm/run_unseen_step.sbatch)
echo "evaluation: $EVAL -> experiments/approach1_latent_self_correction/results/unseen.md"
