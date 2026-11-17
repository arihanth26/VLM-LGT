#!/bin/bash
# The whole correction study with another frozen base model (default Qwen3-VL-4B), as one dependency chain:
#   the candidate pipeline with that model's question embeddings, NUM_SHARDS jobs per dataset split
#   (run_base_model_chain.sbatch; the ranker's train subset, all of val and test)
#   -> ranker v3 with sampled reasoning (the head features are a constant 0.5: the 2B heads cannot read another model's
#      states), scalar variants only -> gating with the reproduction check
# Results under experiments/approach1_latent_self_correction/results/base_<size>/. Same questions as the 2B study.
# Options (environment):
#   DATASETS   which datasets' chains to submit (default all four), e.g. DATASETS="docvqa scienceqa_img"
#   SPLITS     which splits (default "train val test")
#   RANKER=0   submit only the chains (the ranker and gating are submitted separately, once every chain has run
#              where all chains' outputs are gathered)
# Time limits follow the measured 4B chain times (2.9-4.7 h per half split; DocVQA has the largest images), about twice
# the expected time: the scheduler counts requested GPU-minutes per user, so a smaller request lets more jobs run at once.
# Every stage is resumable: a chain that times out continues where it stopped when resubmitted.
# Run from the repository root on a login node.  Usage: [SIZE=4b] [NUM_SHARDS=2] [DATASETS=...] [RANKER=0] bash slurm/run_base_model_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
SIZE="${SIZE:-4b}"; SHARDS="${NUM_SHARDS:-2}"  # two jobs per dataset split
DATASETS=(${DATASETS:-chartqa_full textvqa docvqa scienceqa_img}); SPLITS=(${SPLITS:-train val test})
declare -A HOURS=([chartqa_full]=07:00:00 [textvqa]=07:00:00 [docvqa]=10:00:00 [scienceqa_img]=07:00:00)
if (( SHARDS >= 4 )); then HOURS=([chartqa_full]=04:00:00 [textvqa]=04:00:00 [docvqa]=06:00:00 [scienceqa_img]=04:00:00); fi  # quarter splits
RESULTS=experiments/approach1_latent_self_correction/results/base_$SIZE
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
CHAINS=()
for D in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        for ((S = 0; S < SHARDS; S++)); do
            CHAINS+=("$("${SUBMIT[@]}" -J "base-$SIZE-$SPLIT-$D-$S" -t "${HOURS[$D]:-10:00:00}" \
                --export=ALL,VLMLGT_MODEL="$SIZE",DATASET="$D",SPLIT="$SPLIT",SHARD="$S",NUM_SHARDS="$SHARDS" slurm/run_base_model_chain.sbatch)")
        done
    done
done
echo "candidate chains (${#CHAINS[@]}): ${CHAINS[*]}"
[[ "${RANKER:-1}" == "0" ]] && exit 0
RANKER_JOB=$("${SUBMIT[@]}" --dependency=afterok:"$(IFS=:; echo "${CHAINS[*]}")" -J "ranker-v3-$SIZE" -t 02:00:00 \
    --export=ALL,VLMLGT_MODEL="$SIZE",RANKER_ARGS="--with-cot-samples --variants scalars_ds scalars_nods" slurm/train_ranker_v3.sbatch)
GATE=$("${SUBMIT[@]}" --dependency=afterok:"$RANKER_JOB" -J "gate-v3-$SIZE" -t 01:00:00 \
    --export=ALL,VLMLGT_MODEL="$SIZE",GATE_ARGS="--with-cot-samples --tag _multi --models $RESULTS/candidate_ranker_v3_multi_models.pt --reported $RESULTS/candidate_ranker_v3_multi.json" \
    slurm/gate_ranker_v3.sbatch)
echo "ranker: $RANKER_JOB; gating: $GATE -> $RESULTS/gating_v3_multi.md"
