#!/bin/bash
# Fixes from other model families (InternVL3-8B, Idefics3-8B; models/model_config.py), as one submission:
#   answers     each model on every question of each dataset: validation, test and the ranker's training subset
#               (slurm/batches/fix_<model>_<dataset>.txt), and on the unseen datasets (fix_families_unseen.txt)
#   routing     the routing study over every fix in results/escalation/            -> results/escalation.*
#   candidates  both models' answers encoded and judged by 2B, one job per dataset -> results/correction_v6/
#   ranker      the best ranker retrained with them as extra candidates, then gated -> candidate_ranker_v3_multi_esc_cot8b_families_drop_judge.*
#   latency     seconds per component on A100 and L40S (2B, 8B and both models), and both models on H100 -> results/latency_*.json
# Needs the models downloaded (sbatch --export=ALL,MODEL_SIZE=<key> slurm/prefetch_model.sbatch for each).
# Run from the repository root on a login node.  Usage: bash slurm/run_families_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
E=experiments/approach1_latent_self_correction
S=(sbatch --parsable --kill-on-invalid-dep=yes)
declare -A FIX
for D in chartqa_full textvqa docvqa scienceqa_img; do
    for M in internvl3_8b idefics3_8b; do
        FIX[$M-$D]=$("${S[@]}" -J "fix-$M-$D" -t 04:00:00 --export=ALL,BATCH="fix_${M}_${D}.txt" slurm/run_batch.sbatch)
    done
done
UNSEEN=$("${S[@]}" -J fix-families-unseen -t 03:00:00 --export=ALL,BATCH=fix_families_unseen.txt slurm/run_batch.sbatch)
ROUTE=$("${S[@]}" --dependency=afterok:"$(IFS=:; echo "${FIX[*]}")" -J routing-families -t 01:00:00 \
    --export=ALL,BATCH=routing_other_families.txt slurm/run_batch.sbatch)
ENC=()
for D in chartqa_full textvqa docvqa scienceqa_img; do
    ENC+=("$("${S[@]}" --dependency=afterok:"${FIX[internvl3_8b-$D]}:${FIX[idefics3_8b-$D]}" -J "encode-families-$D" -t 03:00:00 \
        --export=ALL,BATCH="encode_families_${D}.txt" slurm/run_batch.sbatch)")
done
RANK=$("${S[@]}" --dependency=afterok:"$(IFS=:; echo "${ENC[*]}")" -J ranker-families -t 02:00:00 --export=ALL,BATCH=families_ranker.txt slurm/run_batch.sbatch)
LAT=()
for GPU in A100-80GB L40S; do
    LAT+=("$("${S[@]}" -C "$GPU" -J "latency-$GPU" -t 02:00:00 \
        --export=ALL,LATENCY_ARGS="--models 2b 8b internvl3_8b idefics3_8b --output $E/results/latency_$GPU.json" slurm/measure_latency.sbatch)")
done
LAT+=("$("${S[@]}" -C H100 -J latency-H100-families -t 02:00:00 \
    --export=ALL,LATENCY_ARGS="--models internvl3_8b idefics3_8b --output $E/results/latency_H100_families.json" slurm/measure_latency.sbatch)")
echo "answers: ${FIX[*]}"; echo "unseen answers: $UNSEEN"; echo "routing: $ROUTE"; echo "candidates: ${ENC[*]}"
echo "ranker + gating: $RANK"; echo "latency: ${LAT[*]}"
