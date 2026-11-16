#!/bin/bash
# The whole correction study with another frozen base model (default Qwen3-VL-4B), as one dependency chain:
#   the candidate pipeline with that model's question embeddings, NUM_SHARDS jobs per dataset split
#   (run_base_model_chain.sbatch; the ranker's train subset, all of val and test)
#   -> ranker v3 with sampled reasoning (the head features are a constant 0.5: the 2B heads cannot read another model's
#      states), scalar variants only -> gating with the reproduction check
# Results under experiments/approach1_latent_self_correction/results/base_<size>/. Same questions as the 2B study.
# Run from the repository root on a login node.  Usage: [SIZE=4b] [NUM_SHARDS=2] bash slurm/run_base_model_pipeline.sh

set -euo pipefail
[[ -d experiments && -d models ]] || { echo "Run from the repository root."; exit 2; }
mkdir -p logs
SIZE="${SIZE:-4b}"; SHARDS="${NUM_SHARDS:-1}"  # one job per dataset split: 14 jobs in all
RESULTS=experiments/approach1_latent_self_correction/results/base_$SIZE
SUBMIT=(sbatch --parsable --kill-on-invalid-dep=yes)
CHAINS=()
for D in chartqa_full textvqa docvqa scienceqa_img; do
    for SPLIT in train val test; do
        for ((S = 0; S < SHARDS; S++)); do
            CHAINS+=("$("${SUBMIT[@]}" -J "base-$SIZE-$SPLIT-$D-$S" \
                --export=ALL,VLMLGT_MODEL="$SIZE",DATASET="$D",SPLIT="$SPLIT",SHARD="$S",NUM_SHARDS="$SHARDS" slurm/run_base_model_chain.sbatch)")
        done
    done
done
RANKER=$("${SUBMIT[@]}" --dependency=afterok:"$(IFS=:; echo "${CHAINS[*]}")" -J "ranker-v3-$SIZE" \
    --export=ALL,VLMLGT_MODEL="$SIZE",RANKER_ARGS="--with-cot-samples --variants scalars_ds scalars_nods" slurm/train_ranker_v3.sbatch)
GATE=$("${SUBMIT[@]}" --dependency=afterok:"$RANKER" -J "gate-v3-$SIZE" \
    --export=ALL,VLMLGT_MODEL="$SIZE",GATE_ARGS="--with-cot-samples --tag _multi --models $RESULTS/candidate_ranker_v3_multi_models.pt --reported $RESULTS/candidate_ranker_v3_multi.json" \
    slurm/gate_ranker_v3.sbatch)
echo "candidate chains (${#CHAINS[@]}): ${CHAINS[*]}"
echo "ranker: $RANKER; gating: $GATE -> $RESULTS/gating_v3_multi.md"
