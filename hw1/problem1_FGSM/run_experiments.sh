#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then
    echo "Usage: $0 <target-class: 0|1>" >&2
    exit 2
fi
target="$1"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
output_file="results_t${target}.txt"
failures=0
{
    printf 'Target: %s\nTrials: 5\nSeed: 13 (reset in every process)\n' "$target"
    printf 'UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for trial in {1..5}; do
        printf '\n=== Trial %s ===\n' "$trial"
        printf 'Command: uv run --locked python -u fgsm.py --t %s\n' "$target"
        if uv run --locked python -u fgsm.py --t "$target"; then
            status=0
        else
            status=$?
            failures=$((failures + 1))
        fi
        printf 'Exit code: %s\n' "$status"
    done
} > "$output_file" 2>&1
printf 'Saved 5 trials to %s (%s failed).\n' "$output_file" "$failures"
if (( failures > 0 )); then
    exit 1
fi
