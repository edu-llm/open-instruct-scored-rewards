#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$REPO"

mkdir -p "$PROJECT_DIR/runs/logs"
active_jobs="$(
    squeue -h -u "$USER" -o '%j' |
        awk '$1 == "olmoe_ft_smoke" || $1 == "olmoe_ft_sweep" || $1 == "olmoe_full_ft" { print }'
)"
if [[ -n "$active_jobs" ]]; then
    echo "refusing to submit a duplicate; active OLMoE full-FT jobs:" >&2
    printf '%s\n' "$active_jobs" >&2
    exit 2
fi

smoke_job="$(sbatch --parsable "$PROJECT_DIR/smoke.sbatch")"
sweep_job="$(
    sbatch --parsable \
        --dependency="afterok:$smoke_job" \
        --kill-on-invalid-dep=yes \
        "$PROJECT_DIR/sweep.sbatch"
)"

commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf '%s\tcommit=%s\tsmoke=%s\tsweep=%s\n' \
    "$(date -Is)" "$commit" "$smoke_job" "$sweep_job" \
    >>"$PROJECT_DIR/runs/submissions.tsv"

echo "smoke job: $smoke_job"
echo "sweep array: $sweep_job (held until smoke succeeds)"
echo "check with: bash $PROJECT_DIR/status.sh"
