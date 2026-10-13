#!/bin/bash
# Validates every Slurm script in this folder without running anything:
#   1. bash syntax,
#   2. every `python -m <module>` it launches resolves in the project environment,
#   3. on a machine with Slurm, sbatch --test-only accepts the resource request.
# Run from the repository root. Exits non-zero if anything fails.
# Usage: bash slurm/check_scripts.sh

set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN=".venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python"
failures=0

for script in slurm/*.sbatch slurm/*.sh; do
    if ! bash -n "$script" 2> /tmp/check_syntax.txt; then
        echo "FAIL syntax: $script"; cat /tmp/check_syntax.txt; failures=$((failures + 1)); continue
    fi
    for module in $(grep -oE "python\"? -m [A-Za-z0-9_.]+|\\$PYTHON_BIN\" -m [A-Za-z0-9_.]+" "$script" | awk '{print $NF}' | sort -u); do
        if ! "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module') else 1)" 2> /dev/null; then
            echo "FAIL module: $script launches '$module', which does not resolve"; failures=$((failures + 1))
        fi
    done
    if [[ "$script" == *.sbatch ]] && command -v sbatch > /dev/null; then
        if ! out=$(sbatch --test-only "$script" 2>&1); then
            echo "FAIL sbatch --test-only: $script"; echo "$out"; failures=$((failures + 1)); continue
        fi
    fi
    echo "ok: $script"
done

if [[ $failures -ne 0 ]]; then
    echo "$failures problem(s) found"
    exit 1
fi
echo "all scripts passed"
