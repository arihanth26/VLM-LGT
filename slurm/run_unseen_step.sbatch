#!/bin/bash
#SBATCH -J vlm-lgt-unseen
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/unseen-%j.out

# GPU job: one step of the unseen-dataset transfer test (run_unseen_pipeline.sh), chosen by STEP:
#   build      download and normalize AI2D and InfographicVQA (build_unseen_splits.py)
#   features   the constant question embedding of DATASET's unseen split (extract_prompt_features.py)
#   features_subset  the same for DATASET's SPLIT, the ranker's subset only (with VLMLGT_MODEL: another base model)
#   check      compare extract_prompt_features.py with extract_q_features.py on DATASET's split SPLIT
#   evaluate   tests, then the frozen pipeline and the baselines on the unseen split (evaluate_unseen.py)
# Usage: sbatch --export=ALL,STEP=features,DATASET=ai2d slurm/run_unseen_step.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
M=experiments.approach1_latent_self_correction
case "${STEP:?Set STEP}" in
    build)    srun "$PYTHON_BIN" -m data.approach1_latent_self_correction.build_unseen_splits ;;
    features) srun "$PYTHON_BIN" -m $M.extract_prompt_features --dataset "${DATASET:?}" --split unseen ;;
    features_subset) srun "$PYTHON_BIN" -m $M.extract_prompt_features --dataset "${DATASET:?}" --split "${SPLIT:?}" --subset ;;
    check)    srun "$PYTHON_BIN" -m $M.extract_prompt_features --dataset "${DATASET:?}" --split "${SPLIT:?}" --check ;;
    evaluate) srun "$PYTHON_BIN" -m tests.run_tests tests.test_baselines tests.test_unseen
              srun "$PYTHON_BIN" -m $M.evaluate_unseen ;;
    *) echo "unknown STEP $STEP"; exit 2 ;;
esac
