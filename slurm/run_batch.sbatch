#!/bin/bash
#SBATCH -J vlm-lgt-batch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p coc-gpu,ice-gpu
#SBATCH -C "H200|H100|L40S|A100-80GB|A100-40GB|A40"
#SBATCH --mem=128G
#SBATCH -t 16:00:00
#SBATCH -o logs/batch-%j.out

# GPU job: several short experiments in one job, so more of them fit under the 50-job queue limit.
# BATCH names a file in slurm/batches/; every non-empty, non-comment line is one run, written as the module
# under experiments/approach1_latent_self_correction followed by its options, e.g.
#   train_ranker_v3 --with-reasoning-8b --drop judge --seed-offset 100
# The runs go one after another; a failed run is logged and the next one still runs; the job fails at the
# end if any run failed. Unit tests for the modules run first.
# Usage: sbatch --export=ALL,BATCH=headline_seeds.txt slurm/run_batch.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -d experiments && -d models ]] || { echo "Submit from the repository root."; exit 2; }
: "${BATCH:?Set BATCH to a file in slurm/batches/}"
module load anaconda3
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"  # model weights, for runs that load a model
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Run: sbatch slurm/setup_environment.sbatch"; exit 2; }
srun "$PYTHON_BIN" -m tests.run_tests tests.test_gating tests.test_ranker_v3 tests.test_escalation_candidates tests.test_cascade tests.test_escalation
failed=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    echo "[batch] $(date +%H:%M:%S) start: $line"
    read -r -a run <<< "$line"
    if srun "$PYTHON_BIN" -m "experiments.approach1_latent_self_correction.${run[0]}" "${run[@]:1}" < /dev/null; then
        echo "[batch] $(date +%H:%M:%S) done: $line"
    else
        echo "[batch] $(date +%H:%M:%S) FAILED: $line"; failed+=("$line")
    fi
done < "slurm/batches/$BATCH"
(( ${#failed[@]} == 0 )) || { printf '[batch] failed run: %s\n' "${failed[@]}"; exit 1; }
