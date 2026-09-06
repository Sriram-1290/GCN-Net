#!/usr/bin/env bash
# Sequential driver: one Python process at a time, never two.
#
# Why this exists. Launching the arms as separate background loops put several
# Python processes on the card at once -- each holds ~1.7 GB of CUDA reserve, and
# four of them filled an 8 GB card. Worse, killing Python alone left the bash
# wrapper alive, which immediately spawned the next invocation and re-logged rows
# under a stale configuration. Everything runs from this one script instead.
#
# TO STOP THIS CLEANLY: kill THIS script's PID first (written to runs/tonight.pid),
# then any surviving python. Killing python alone will just start the next arm.
set -u
DEADLINE="$1"
echo $$ > runs/tonight.pid
trap 'echo "stopping"; rm -f runs/tonight.pid; exit 0' INT TERM

echo "### A3: Riemannian tangent space (CPU, ~10 min) ###"
python -u scripts/run_tangent_space.py --tag A3-tgsp

echo
echo "### 9.18 CI: CSP-initialised spatial operator (GPU, ~4.6 h) ###"
# Hyperparameters copied verbatim from the G-arm command in 17. The script
# defaults are 100 epochs / batch 1024 / lr 1e-6, which sit at chance and
# overflow VRAM. Do not drop these flags.
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

rm -f runs/tonight.pid
echo "### all done or budget spent ###"
