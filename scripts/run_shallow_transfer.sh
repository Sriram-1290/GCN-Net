#!/usr/bin/env bash
# 9.16: does the pre-training gain transfer to ShallowConvNet?
#
# Order matters. PTC (the control) runs first because it is the cheap arm that can
# invalidate the expensive one: 9.16.2 requires PTC-shallow to land near
# ShallowConvNet's 0.6725, and if it does not, the difference of differences is
# confounded and PTG-shallow should not be run at all.
#
# One absolute deadline shared by every invocation. Since error 23 a past --stop-at
# stops immediately rather than rolling to tomorrow, so the later commands in this
# list exit cleanly once the budget is spent.
set -u
DEADLINE="$1"
COMMON="--config configs/bandpass_8_30.yaml --model shallow --groups 10 --seed 42 --resume --stop-at"

for classes in "0 1" "2 3"; do
  suffix=$(echo "$classes" | tr ' ' '-')
  echo "=== PTC-shallow-binary-$suffix (control) ==="
  python -u scripts/run_pretrain_finetune.py $COMMON "$DEADLINE" \
    --classes $classes --no-pretrain --tag "PTC-shallow-binary-$suffix"
done

for classes in "0 1" "2 3"; do
  suffix=$(echo "$classes" | tr ' ' '-')
  echo "=== PTG-shallow-binary-$suffix (pre-trained) ==="
  python -u scripts/run_pretrain_finetune.py $COMMON "$DEADLINE" \
    --classes $classes --tag "PTG-shallow-binary-$suffix"
done
echo "=== budget spent or all arms complete ==="
