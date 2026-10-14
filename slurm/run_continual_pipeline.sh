#!/bin/bash
# Submits the sequential-training experiments, one job per head kind. Each job needs the
# mixture sweep of the same kind (for its head configuration) and the mixture files.
# Run from the repository root on a login node.
# Usage: bash slurm/run_continual_pipeline.sh [2b|4b]
#   or, to wait for sweeps that are already queued:
#   PRE_JOB=<id> VERIFIER_JOB=<id> UNIFIED_JOB=<id> bash slurm/run_continual_pipeline.sh

set -euo pipefail
SIZE="${1:-2b}"
mkdir -p logs

for KIND in pre verifier unified; do
    VAR="$(echo "$KIND" | tr a-z A-Z)_JOB"
    DEP=()
    [[ -n "${!VAR:-}" ]] && DEP=(--dependency=afterok:"${!VAR}")
    JOB=$(sbatch --parsable --kill-on-invalid-dep=yes "${DEP[@]}" \
        --export=ALL,KIND="$KIND",MODEL_SIZE="$SIZE" slurm/sweep_continual.sbatch)
    echo "continual $KIND: $JOB"
done
echo "When all finish: build_mixture_report on a machine with matplotlib (see README)"
