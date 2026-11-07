#!/usr/bin/env bash
# Phase A job queue (docs/project_documentation.md 9.26). Runs the commands listed in a queue
# file one at a time, retrying each until it exits 0 (every command must carry --resume, so a
# retry continues where the failed attempt stopped: host-memory errors such as
# ArrayMemoryError or WinError 1455 have killed long runs before).
#
# Usage: bash scripts/phase_a_queue.sh <queue-file> [<pid-file-to-wait-for>]
#
# Queue file: one job per line, "<label>|<command>". Blank lines and lines starting with #
# are ignored. A line "END" stops the queue once everything above it is done; until then the
# queue polls the file every 60 s, so jobs can be appended while it runs (this script itself
# is never edited mid-run). Finished labels are recorded in <queue-file>.done and skipped
# on a restart.
#
# TO STOP CLEANLY: kill the PID in <queue-file>.pid FIRST, then any surviving python.
set -u
QUEUE="$1"
WAIT_FOR="${2:-}"
MAX_TRIES=6
DONE="$QUEUE.done"
touch "$DONE"
echo $$ > "$QUEUE.pid"
trap 'rm -f "$QUEUE.pid"; exit 0' INT TERM

if [ -n "$WAIT_FOR" ]; then
  while [ -f "$WAIT_FOR" ]; do sleep 60; done
fi

while true; do
  next=""
  saw_end=0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; END) saw_end=1; break ;; esac
    label="${line%%|*}"
    if ! grep -qxF "$label" "$DONE"; then next="$line"; break; fi
  done < "$QUEUE"

  if [ -z "$next" ]; then
    if [ $saw_end -eq 1 ]; then break; fi
    sleep 60
    continue
  fi

  label="${next%%|*}"
  cmd="${next#*|}"
  try=1
  ok=0
  while [ $try -le $MAX_TRIES ]; do
    echo "=== $label (attempt $try) $(date '+%Y-%m-%d %H:%M:%S') ==="
    bash -c "$cmd"
    rc=$?
    if [ $rc -eq 0 ]; then ok=1; break; fi
    echo "!!! $label failed (exit $rc), retrying in 20 s"
    sleep 20
    try=$((try + 1))
  done
  if [ $ok -eq 1 ]; then
    echo "$label" >> "$DONE"
    echo "=== $label done $(date '+%Y-%m-%d %H:%M:%S') ==="
  else
    echo "### $label gave up after $MAX_TRIES attempts; marked done-failed ###"
    echo "$label" >> "$DONE"
    echo "$label" >> "$QUEUE.failed"
  fi
done

rm -f "$QUEUE.pid"
echo "### queue $QUEUE finished $(date '+%Y-%m-%d %H:%M:%S') ###"
