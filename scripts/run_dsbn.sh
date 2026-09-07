#!/usr/bin/env bash
# 9.22: domain-specific batch normalisation in the pre-training stage.
#
# One variable against the existing PTG control: whether each source subject gets
# its own batch-norm statistics. Every training hyperparameter below matches the
# PTG arm exactly -- the script defaults differ and would silently produce an
# incomparable arm (error 26).
#
# Sequential, one Python process at a time. TO STOP CLEANLY: kill the PID in
# runs/dsbn.pid FIRST, then any surviving python; killing python alone lets this
# loop start the next arm (error 27/28).
set -u
DEADLINE="$1"
echo $$ > runs/dsbn.pid
trap 'rm -f runs/dsbn.pid; exit 0' INT TERM

for classes in "0 1" "2 3"; do
  suffix=$(echo "$classes" | tr ' ' '-')
  echo "=== PTD-gbp-binary-$suffix ==="
  python -u scripts/run_pretrain_finetune.py \
    --config configs/bandpass_8_30.yaml --classes $classes \
    --model graphbp --spatial free --dsbn --groups 10 \
    --pretrain-epochs 30 --finetune-epochs 30 \
    --pretrain-lr 1e-3 --finetune-lr 1e-4 --batch-size 256 \
    --window 64 --stride 16 --seed 42 \
    --tag "PTD-gbp-binary-$suffix" --resume --stop-at "$DEADLINE"
done

rm -f runs/dsbn.pid
echo "### done or budget spent ###"
