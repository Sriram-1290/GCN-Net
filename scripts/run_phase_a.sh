#!/usr/bin/env bash
# Phase A runs (docs/project_documentation.md 9.26). Sequential, one Python process at a time
# (host RAM ~16 GB), every arm resumable with --resume and retried on failure (host-memory
# errors: ArrayMemoryError, WinError 1455).
#
# Usage: bash scripts/run_phase_a.sh <stage> [<stage> ...]
#   stages: a2
#
# TO STOP CLEANLY: kill the PID in runs/phase_a.pid FIRST, then any surviving python;
# killing python alone lets the retry loop start the arm again.
set -u
PY=C:/Python314/python.exe
MAX_TRIES=6
echo $$ > runs/phase_a.pid
trap 'rm -f runs/phase_a.pid; exit 0' INT TERM

# run_arm <label> <command...>: rerun the command (which must carry --resume) until it exits 0.
run_arm() {
  local label="$1"; shift
  local try=1
  while [ $try -le $MAX_TRIES ]; do
    echo "=== $label (attempt $try) $(date '+%H:%M:%S') ==="
    "$@"
    local rc=$?
    if [ $rc -eq 0 ]; then
      echo "=== $label done $(date '+%H:%M:%S') ==="
      return 0
    fi
    echo "!!! $label failed (exit $rc), retrying in 20 s"
    sleep 20
    try=$((try + 1))
  done
  echo "### $label gave up after $MAX_TRIES attempts ###"
  return 1
}

stage_a2() {
  # A2 (ii): settings copied from the logged A2-shallow-binary-* arm; only --window-start differs.
  for classes in "0 1" "2 3"; do
    local suffix
    suffix=$(echo "$classes" | tr ' ' '-')
    run_arm "PA-A2-shallow-ws160-binary-$suffix" \
      $PY -u scripts/run_subject_dependent.py \
        --config configs/bandpass_8_30.yaml --classes $classes \
        --model shallow --window 64 --stride 16 --window-start 160 \
        --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 --seed 42 \
        --tag "PA-A2-shallow-ws160-binary-$suffix" --resume --quiet
  done
}

for stage in "$@"; do
  case "$stage" in
    a2) stage_a2 ;;
    *) echo "unknown stage: $stage"; exit 2 ;;
  esac
done

rm -f runs/phase_a.pid
echo "### phase A stages done: $* ###"
