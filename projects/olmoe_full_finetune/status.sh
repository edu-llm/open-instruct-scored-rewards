#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "active jobs"
squeue -u "$USER" -o '%.18i %.10P %.24j %.10T %.12M %.12l %R' |
    awk 'NR == 1 || $3 ~ /^olmoe_(ft_|full_)/'

echo
echo "recent accounting"
since="$(date -d '7 days ago' +%F)"
sacct -u "$USER" -S "$since" -X \
    --format=JobID,JobName%24,State,Elapsed,Timelimit,ExitCode,NodeList%18 |
    awk 'NR <= 2 || $2 ~ /^olmoe_(ft_|full_)/'

echo
echo "failure signatures"
shopt -s nullglob
logs=("$PROJECT_DIR"/runs/logs/*.out "$PROJECT_DIR"/runs/logs/*.err)
if (( ${#logs[@]} == 0 )); then
    echo "no logs yet"
elif ! grep -Ein \
    'Traceback|CUDA out of memory|OutOfMemory|NCCL.*(error|timeout)|ChildFailedError|Killed|No space left|FAILED' \
    "${logs[@]}" |
    tail -n 20; then
    echo "none found"
fi

echo
echo "last submission"
if [[ -f "$PROJECT_DIR/runs/submissions.tsv" ]]; then
    tail -n 1 "$PROJECT_DIR/runs/submissions.tsv"
else
    echo "none recorded"
fi
