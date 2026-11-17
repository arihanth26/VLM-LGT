#!/bin/bash
# Follow-ups of the best ranker (8B answer + 8B reasoning as candidates, no judge input), packed so that
# several runs share one job (slurm/run_batch.sbatch) and the whole set needs 7 queue slots:
#   batch headline_seeds     two more seed sets of the best ranker (training randomness)
#   batch headline_holdout   leave-one-dataset-out for the best ranker
#   batch data_efficiency    the best ranker and the 2B-only ranker on 250 / 750 / 1,500 questions per dataset
#   batch seconds_gating     gating with the measured H100 seconds as a second cost axis
#   unseen 8B reasoning      8B reasoning answers on AI2D and InfographicVQA, encoded as candidates
#                            (after UNSEEN_AFTER: the unseen pipeline's 8B-candidate jobs), then the unseen
#                            evaluation again, now with the best ranker (after EVAL_AFTER: the first evaluation)
# Run from the repository root on a login node.
# Usage: [UNSEEN_AFTER=<id>:<id>] [EVAL_AFTER=<id>] bash slurm/run_followups_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
for B in headline_seeds headline_holdout data_efficiency seconds_gating; do
    echo "batch $B: $("${SUBMIT[@]}" -J "batch-$B" --export=ALL,BATCH="$B.txt" slurm/run_batch.sbatch)"
done
DEP=(); [[ -n "${UNSEEN_AFTER:-}" ]] && DEP=(--dependency=afterok:"$UNSEEN_AFTER")
COT=()
for D in ai2d infovqa; do
    COT+=("$("${SUBMIT[@]}" "${DEP[@]}" -J "cot8b-unseen-$D" --export=ALL,DATASET="$D",SPLIT=unseen slurm/run_reasoning_8b.sbatch)")
done
EXTRA=""; [[ -n "${EVAL_AFTER:-}" ]] && EXTRA=",afterany:$EVAL_AFTER"
EVAL=$("${SUBMIT[@]}" --dependency=afterok:"$(IFS=:; echo "${COT[*]}")$EXTRA" -J unseen-evaluate-2 --export=ALL,STEP=evaluate slurm/run_unseen_step.sbatch)
echo "unseen 8B reasoning: ${COT[*]}; evaluation with the best ranker: $EVAL -> experiments/approach1_latent_self_correction/results/unseen.md"
