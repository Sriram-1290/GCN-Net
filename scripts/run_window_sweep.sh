#!/usr/bin/env bash
# 9.20: window-length sweep on the development set (S001-S020).
#
# Sequential, one Python process at a time -- see run_tonight.sh for why.
# TO STOP CLEANLY: kill the PID in runs/window_sweep.pid FIRST, then any python.
#
# window=64 is NOT re-run; the existing A2-shallow-binary rows restricted to these
# 20 subjects are the baseline. Hyperparameters match the A2 arm exactly.
set -u
DEADLINE="$1"
echo $$ > runs/window_sweep.pid
trap 'rm -f runs/window_sweep.pid; exit 0' INT TERM

DEV="S001 S002 S003 S004 S005 S006 S007 S008 S009 S010 \
S011 S012 S013 S014 S015 S016 S017 S018 S019 S020"

for W in 128 256; do
  for classes in "0 1" "2 3"; do
    suffix=$(echo "$classes" | tr ' ' '-')
    echo "=== W$W-shallow-dev-binary-$suffix ==="
    python -u scripts/run_subject_dependent.py \
      --config configs/bandpass_8_30.yaml --classes $classes \
      --subjects $DEV --model shallow \
      --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 \
      --window "$W" --stride 16 --seed 42 \
      --tag "W$W-shallow-dev-binary-$suffix" --resume --stop-at "$DEADLINE"
  done
done

rm -f runs/window_sweep.pid
echo "### sweep done or budget spent ###"
