#!/bin/bash
#SBATCH -J vlm-lgt-base
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH -o logs/base-model-%j.out

# GPU job: the whole candidate pipeline for one shard of one dataset split with another frozen base model
# (VLMLGT_MODEL, default 4b): T=1.0 pools (greedy + 10 samples), first round (contrastive decoding, zoomed
# crops), second round (grounded crops, reasoning, judge), sampled reasoning; every candidate encoded by that
# model. Results go to results/base_<size>/. Shards follow VLMLGT_CHAIN, so each stage finds exactly the
# questions its earlier stage produced; every stage is resumable, so a resubmitted job continues.
# The first step computes that model's question embeddings for the split's ranker subset (skipped if present).
# Usage: sbatch --export=ALL,VLMLGT_MODEL=4b,DATASET=textvqa,SPLIT=val,SHARD=0,NUM_SHARDS=2 slurm/run_base_model_chain.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export VLMLGT_MODEL="${VLMLGT_MODEL:-4b}" VLMLGT_CHAIN=1
: "${DATASET:?Set DATASET}"
: "${SPLIT:?Set SPLIT}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
M=experiments.approach1_latent_self_correction
S=(--dataset "$DATASET" --split "$SPLIT" --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}")
FEATURES=experiments/approach1_latent_self_correction/q_features/${DATASET}_${SPLIT}_${VLMLGT_MODEL}.pt
[[ -f "$FEATURES" ]] || srun "$PYTHON_BIN" -m $M.extract_prompt_features --dataset "$DATASET" --split "$SPLIT" --subset
srun "$PYTHON_BIN" -m $M.candidate_pools "${S[@]}" --temperature 1.0 --subset
srun "$PYTHON_BIN" -m $M.correction_candidates "${S[@]}"
srun "$PYTHON_BIN" -m $M.correction_v2 "${S[@]}"
srun "$PYTHON_BIN" -m $M.correction_v3_cot "${S[@]}"
