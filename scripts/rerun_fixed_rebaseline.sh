#!/usr/bin/env bash
# Fixed-code rebaseline for the CalMS21 ST-GCN study (commit 096d9e1 seed/best-ckpt fix).
#
# Why: pre-fix code, --seed never controlled weight init and F1 was read from the
# last-epoch (not best) model. Every quantitative number in
# 260706_behaviorlab_calms21_stgcn_gradcam.md is provisional until re-measured here.
#
# Scope (Karpathy: decisive first, skip speculative):
#   Tier A  clean baseline 5 seeds  -> true null distribution / CI  (the decisive test)
#   Tier B  data-quantity 4000 vs 6000, fixed-seed pair             (the surviving positive result)
#   Tier C  keypoint noise sweep -- OPT-IN only (RUN_NOISE=1). Re-examines an effect the
#           note already RETRACTED; low prior, ~7h. Not run by default.
#
# Flaky-kill design (handoff issue 4): each run is independent + idempotent. A finished
# run writes "Test F1-macro" to its rerun.log and is skipped on restart. If killed, just
# re-run this script; it resumes.  Each run ~25-30 min CPU. Tier A+B ~= 3 h.
set -u
cd "$(dirname "$0")/.." || exit 1

# DEVICE=cuda by default (this runner targets gpu03); override with DEVICE=cpu on mac.
# PY lets you point at a specific interpreter, e.g. PY=~/anaconda3/envs/behaviorsplatter/bin/python
DEVICE="${DEVICE:-cuda}"
PY="${PY:-python}"

done_already() {  # $1=log path
  [ -f "$1" ] && grep -q "Test F1-macro" "$1"
}

# ---- Tier A: clean baseline, fixed weight init, 5 seeds (own dirs; old clean preserved)
for s in 42 123 456 789 1000; do
  dir="outputs/calms21_stgcn_fixed_seed${s}"
  if done_already "$dir/rerun.log"; then echo "SKIP: $dir"; continue; fi
  mkdir -p "$dir"; echo "RUN(clean): seed=$s -> $dir [$DEVICE]"
  $PY scripts/train_calms21_stgcn.py --seed "$s" --device "$DEVICE" --output_dir "$dir" 2>&1 | tee "$dir/rerun.log"
done

# ---- Tier B: data-quantity lever, fixed-seed pair (clean seed42 above == the 4000 arm)
dir="outputs/calms21_stgcn_bigger_fixed_seed42"
if done_already "$dir/rerun.log"; then echo "SKIP: $dir"; else
  mkdir -p "$dir"; echo "RUN(6000): seed=42 -> $dir [$DEVICE]"
  $PY scripts/train_calms21_stgcn.py --seed 42 --max_train 6000 --max_test 1200 \
    --device "$DEVICE" --output_dir "$dir" 2>&1 | tee "$dir/rerun.log"
fi

# ---- Null control: shuffle TRAIN labels -> F1 must collapse to ~chance. Validates that
# the +0.33-over-majority "signal" is real structure, not a low-baseline artifact (Devil rec).
dir="outputs/calms21_stgcn_nullctrl_seed42"
if done_already "$dir/rerun.log"; then echo "SKIP: $dir"; else
  mkdir -p "$dir"; echo "RUN(null-control): seed=42 -> $dir [$DEVICE]"
  $PY scripts/train_calms21_stgcn.py --seed 42 --shuffle_labels \
    --device "$DEVICE" --output_dir "$dir" 2>&1 | tee "$dir/rerun.log"
fi

# ---- Tier C: OPT-IN (RUN_NOISE=1). Re-examines the RETRACTED noise effect under fixed code.
# keypoint_noise_sweep.py has no --output_dir, so it writes to its own canonical dirs
# (overwriting pre-fix noise outputs -- intended, old numbers are in the note).
if [ "${RUN_NOISE:-0}" = "1" ]; then
  for kp in nose left_ear right_ear neck left_hip right_hip tail_base; do
    for s in 42 123; do
      echo "RUN(noise): $kp seed=$s [$DEVICE]"
      $PY scripts/keypoint_noise_sweep.py --keypoint "$kp" --seed "$s" --device "$DEVICE"
    done
  done
fi

echo "DONE. Collect: grep -H 'Test F1-macro' outputs/calms21_stgcn_fixed_*/rerun.log outputs/calms21_stgcn_bigger_fixed_*/rerun.log outputs/calms21_stgcn_nullctrl_*/rerun.log"
