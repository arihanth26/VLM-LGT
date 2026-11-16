#!/bin/bash
# Follow-ups of the ranker ablations and the escalation results, as one submission:
#   1. cascade        2B -> 4B -> 8B against 2B -> 8B (analyze_cascade.py), on the answers already made
#   2. no-judge cost  the ranker trained without the judge (--drop judge) needs one scoring pass per candidate
#                     instead of two; gate it with --scoring-passes 1                    -> gating_v3_multi_drop_judge.*
#   3. no-judge + 8B  the escalation ranker retrained without the judge, then gated with one scoring pass
#                                                                                        -> gating_v3_multi_esc_drop_judge.*
#   4. no-judge + 8B reasoning   the same after the 8B reasoning answers (COT8B_JOBS: their job ids, colon-separated)
#                                                                                        -> gating_v3_multi_esc_cot8b_drop_judge.*
# Needs candidate_ranker_v3_multi_drop_judge_models.pt (the judge ablation) and correction_v4.
# Run from the repository root on a login node.  Usage: [COT8B_JOBS=<id>:<id>...] bash slurm/run_next_steps_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
RESULTS=experiments/approach1_latent_self_correction/results
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)

CASCADE=$("${SUBMIT[@]}" slurm/analyze_cascade.sbatch)
echo "cascade: $CASCADE -> $RESULTS/cascade.md"

GATE_NJ=$("${SUBMIT[@]}" -J gate-v3-nojudge \
    --export=ALL,GATE_ARGS="--with-cot-samples --drop judge --scoring-passes 1 --tag _multi_drop_judge --models $RESULTS/candidate_ranker_v3_multi_drop_judge_models.pt --reported $RESULTS/candidate_ranker_v3_multi_drop_judge.json" \
    slurm/gate_ranker_v3.sbatch)
echo "no-judge gating: $GATE_NJ -> $RESULTS/gating_v3_multi_drop_judge.md"

RANK_ESC=$("${SUBMIT[@]}" -J ranker-v3-esc-nojudge \
    --export=ALL,RANKER_ARGS="--with-escalation --drop judge --variants scalars_ds scalars_nods" slurm/train_ranker_v3.sbatch)
GATE_ESC=$("${SUBMIT[@]}" -J gate-v3-esc-nojudge --dependency=afterok:"$RANK_ESC" \
    --export=ALL,GATE_ARGS="--with-escalation --drop judge --scoring-passes 1 --tag _multi_esc_drop_judge --models $RESULTS/candidate_ranker_v3_multi_esc_drop_judge_models.pt --reported $RESULTS/candidate_ranker_v3_multi_esc_drop_judge.json" \
    slurm/gate_ranker_v3.sbatch)
echo "no-judge escalation ranker: $RANK_ESC; gating: $GATE_ESC -> $RESULTS/gating_v3_multi_esc_drop_judge.md"

if [[ -n "${COT8B_JOBS:-}" ]]; then
    RANK_COT=$("${SUBMIT[@]}" -J ranker-v3-cot8b-nojudge --dependency=afterok:"$COT8B_JOBS" \
        --export=ALL,RANKER_ARGS="--with-reasoning-8b --drop judge --variants scalars_ds scalars_nods" slurm/train_ranker_v3.sbatch)
    GATE_COT=$("${SUBMIT[@]}" -J gate-v3-cot8b-nojudge --dependency=afterok:"$RANK_COT" \
        --export=ALL,GATE_ARGS="--with-reasoning-8b --drop judge --scoring-passes 1 --tag _multi_esc_cot8b_drop_judge --models $RESULTS/candidate_ranker_v3_multi_esc_cot8b_drop_judge_models.pt --reported $RESULTS/candidate_ranker_v3_multi_esc_cot8b_drop_judge.json" \
        slurm/gate_ranker_v3.sbatch)
    echo "no-judge 8B reasoning ranker: $RANK_COT; gating: $GATE_COT -> $RESULTS/gating_v3_multi_esc_cot8b_drop_judge.md"
fi
