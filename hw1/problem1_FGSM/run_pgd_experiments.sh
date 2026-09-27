#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then
    echo "Usage: $0 <target-class: 0|1>" >&2
    exit 2
fi
target="$1"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
output_file="results_pgd_t${target}.txt"
failures=0
{
    printf 'Target: %s\nTrials: 10\nModel/input seed: 13\nAttack seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9\n' "$target"
    printf 'UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for seed in {0..9}; do
        printf '\n=== Trial %s ===\n' "$((seed + 1))"
        printf 'Command: uv run python -u pgd.py --t %s --seed %s --eps 0.5 --alpha 0.01 --steps 100 --restarts 100\n' "$target" "$seed"
        if uv run python -u pgd.py --t "$target" --seed "$seed" \
            --eps 0.5 --alpha 0.01 --steps 100 --restarts 100; then
            status=0
        else
            status=$?
            failures=$((failures + 1))
        fi
        printf 'Exit code: %s\n' "$status"
    done
} > "$output_file" 2>&1
printf 'Saved 10 trials to %s (%s failed).\n' "$output_file" "$failures"
if (( failures > 0 )); then
    exit 1
fi
