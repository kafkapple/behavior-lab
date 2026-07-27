#!/bin/bash
# Run sbea_dlc_triangulate over every session in an SBeA split, N at a time.
#
# The dlc2 env has tensorflow-cpu and the GPUs are Blackwell (cc 12.0), which
# TF 2.12's CUDA 11.8 kernels cannot target — so this is CPU-bound and process
# parallelism is the only lever. Measured 260727: ~310 ms/frame in one process,
# ~400 ms/frame with four running, i.e. 4-way gives ~3.1x throughput.
#
# --resolve-lr is deliberately NOT passed. The relabelling is a partial fix
# (scripts/sbea_lr_resolve.py) and the npz keeps raw keypoints_2d, so it can be
# replayed afterwards without spending the run again.
#
# Re-running skips sessions that already have an npz, so it resumes after a kill.
#
# Usage: scripts/run_sbea_all_sessions.sh [root] [outdir] [n_parallel]
set -u

ROOT="${1:-/node_data_2/joon/data/external/SBeA/individual}"
OUT="${2:-/node_data/joon/data/sbea_kp3d_full}"
JOBS="${3:-6}"
PY=/home/joon/anaconda3/envs/dlc2/bin/python
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$OUT/logs"          # /node_data, never NFS — high-frequency writes

mkdir -p "$OUT" "$LOGDIR"
mapfile -t SESSIONS < <(ls "$ROOT"/*-caliParas.mat | xargs -n1 basename | sed 's/-caliParas\.mat//')
echo "$(date +%F_%T) | ${#SESSIONS[@]} sessions in $ROOT | $JOBS parallel | -> $OUT"

for s in "${SESSIONS[@]}"; do
    if [ -f "$OUT/$s-kp3d.npz" ]; then
        echo "  skip $s (exists)"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    (
        start=$(date +%s)
        "$PY" "$REPO/scripts/sbea_dlc_triangulate.py" \
            --root "$ROOT" --session "$s" --start 0 --end 6603 --step 1 \
            --out "$OUT" > "$LOGDIR/$s.log" 2>&1
        rc=$?
        echo "  $( [ $rc -eq 0 ] && echo done || echo FAIL:$rc ) $s ($(( ($(date +%s)-start)/60 )) min)"
    ) &
done
wait
echo "$(date +%F_%T) | finished. $(ls "$OUT"/*-kp3d.npz 2>/dev/null | wc -l) npz present"
