#!/bin/bash
# The 8B answers with another base model (default Qwen3-VL-4B), as one submission:
#   candidates  the 8B short answer and the 8B step-by-step answer (results/escalation/, already made for every question)
#               encoded and judged by the base model, one job per dataset (slurm/batches/esc8b_candidates_<dataset>.txt)
#   ranker      ranker v3 on the base model's candidates plus both 8B answers, without the judge input
#   gating      that ranker gated with one scoring pass per candidate              -> results/base_<size>/gating_v3_multi_esc_cot8b_drop_judge.*
#   routing     the routing study with the base model as the detector and 8B as the fix -> results/base_<size>/escalation.*
# Needs the base model's candidate chains (run_base_model_pipeline.sh). AFTER_<dataset>: job ids (colon-separated) of that
# dataset's chains still running; AFTER_ROUTING: the val/test chains still running.
# Run from the repository root on a login node.
# Usage: [SIZE=4b] [AFTER_docvqa=<id>:<id>] [AFTER_scienceqa_img=...] [AFTER_ROUTING=...] bash slurm/run_base_model_escalation_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
SIZE="${SIZE:-4b}"
RESULTS=experiments/approach1_latent_self_correction/results/base_$SIZE
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
JOBS=()
for D in chartqa_full textvqa docvqa scienceqa_img; do
    VAR="AFTER_$D"; DEP=(); [[ -n "${!VAR:-}" ]] && DEP=(--dependency=afterok:"${!VAR}")
    JOBS+=("$("${SUBMIT[@]}" "${DEP[@]}" -J "esc8b-cand-$SIZE-$D" -t 03:00:00 \
        --export=ALL,VLMLGT_MODEL="$SIZE",BATCH="esc8b_candidates_$D.txt" slurm/run_batch.sbatch)")
done
RANK=$("${SUBMIT[@]}" --dependency=afterok:"$(IFS=:; echo "${JOBS[*]}")" -J "ranker-v3-$SIZE-esc8b" -t 02:00:00 \
    --export=ALL,VLMLGT_MODEL="$SIZE",RANKER_ARGS="--with-reasoning-8b --drop judge --variants scalars_ds scalars_nods" slurm/train_ranker_v3.sbatch)
GATE=$("${SUBMIT[@]}" --dependency=afterok:"$RANK" -J "gate-v3-$SIZE-esc8b" -t 01:00:00 \
    --export=ALL,VLMLGT_MODEL="$SIZE",GATE_ARGS="--with-reasoning-8b --drop judge --scoring-passes 1 --tag _multi_esc_cot8b_drop_judge --models $RESULTS/candidate_ranker_v3_multi_esc_cot8b_drop_judge_models.pt --reported $RESULTS/candidate_ranker_v3_multi_esc_cot8b_drop_judge.json" \
    slurm/gate_ranker_v3.sbatch)
RDEP=(); [[ -n "${AFTER_ROUTING:-}" ]] && RDEP=(--dependency=afterok:"$AFTER_ROUTING")
ROUTE=$("${SUBMIT[@]}" "${RDEP[@]}" -J "routing-$SIZE" -t 01:00:00 --export=ALL,VLMLGT_MODEL="$SIZE" slurm/analyze_escalation.sbatch)
echo "candidates: ${JOBS[*]}; ranker: $RANK; gating: $GATE -> $RESULTS/gating_v3_multi_esc_cot8b_drop_judge.md; routing: $ROUTE -> $RESULTS/escalation.md"
