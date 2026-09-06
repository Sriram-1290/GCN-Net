#!/usr/bin/env bash
# 9.18: is the electrode graph the wrong prior, rather than priors being useless?
#
# One variable against the existing G2 control: what the learnable spatial operator
# is initialised from. Everything else must match G2 exactly, so the training
# hyperparameters below are copied verbatim from the G-arm command in 17 --
# 50 epochs, lr 1e-3, batch 256. The script defaults are NOT those values
# (100 epochs, lr 1e-6, batch 1024), and a first launch that relied on them
# produced an arm that sat at chance and overflowed VRAM. Do not drop these flags.
set -u
DEADLINE="$1"
for classes in "0 1" "2 3"; do
  suffix=$(echo "$classes" | tr ' ' '-')
  echo "=== CI-cspinit-binary-$suffix ==="
  python -u scripts/run_subject_dependent.py \
    --config configs/bandpass_8_30.yaml --classes $classes \
    --model graphbp --spatial free --csp-init \
    --folds 5 --epochs 50 --lr 1e-3 --batch-size 256 \
    --window 64 --stride 16 --seed 42 \
    --tag "CI-cspinit-binary-$suffix" --resume --stop-at "$DEADLINE"
done
echo "=== done or budget spent ==="
