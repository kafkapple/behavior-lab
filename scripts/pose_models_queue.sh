#!/bin/sh
# Sequential local runs of the server pose pipeline on one clip (note 261103_AVATAR_pose_models_server_vs_local).
# One attempt per model (predata Stopping rule): a failure is logged to queue.status and the queue moves on.
# usage: pose_models_queue.sh [pid-to-wait-for]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
O=$ROOT/outputs/pose_models_261103
V=$HOME/data/subtle_upload_test/cort1_first240.mp4
M=$HOME/data/subtle_models
PY=$ROOT/.claude/worktrees/subtle_tutorial/tools/pose_runtime/.venv/bin/python
LPPY=$M/_envs/lp/bin/python
mkdir -p "$O"
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 15; done

run() {  # run NAME PYTHON args...
  name=$1; py=$2; shift 2
  [ -e "$O/$name.json.gz" ] && return 0
  echo "$(date +%T) start $name" >> "$O/queue.status"
  "$py" "$ROOT/scripts/pose_models_run.py" "$@" --video "$V" --out "$O/$name.json.gz" > "$O/$name.log" 2>&1 \
    && echo "$(date +%T) done  $name" >> "$O/queue.status" \
    || echo "$(date +%T) FAIL  $name (see $name.log)" >> "$O/queue.status"
}

run sleap1423_mps_r2 "$PY" --algorithm sleap --model "$M/SLEAP_1423/260504.single_instance.n=1423-1" --device mps
run sleap1501_mps "$PY" --algorithm sleap --model "$M/SLEAP_1501" --device mps
run yolo_mps "$PY" --algorithm yolo --model "$M/YOLO/20250124_m_527_AVATAR3D_KHU.pt" --device mps
run lp_reorganize_cpu "$LPPY" --algorithm lightning_pose --model "$M/LightningPose/20260717_reorganize.lightning_pose_model.zip" --device cpu
run lp_khu_cpu "$LPPY" --algorithm lightning_pose --model "$M/LightningPose/KHU_lightning_pose_model.zip" --device cpu
echo "$(date +%T) queue end" >> "$O/queue.status"
