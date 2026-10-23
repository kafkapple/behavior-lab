#!/bin/bash
# Small overlay-only copy of a large raw video: half size, H.264 CRF 28, keyframe every 20 frames (seekable).
# Same frame count and order as the raw video (checked below), so keypoint frame n is video frame n.
# usage: make_proxy.sh raw.mp4 out.mp4 [seconds]     NOT for keypoint inference: use the raw video there.
set -euo pipefail
raw=$1 out=$2 t=${3:-}
tflag=(); [ -n "$t" ] && tflag=(-t "$t")
read -r w h < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$raw" | tr x ' ')
ffmpeg -v error -y ${tflag[@]+"${tflag[@]}"} -i "$raw" -vf "scale=$((w / 2 / 2 * 2)):$((h / 2 / 2 * 2))" \
  -c:v libx264 -crf 28 -preset veryfast -g 20 -pix_fmt yuv420p -an -movflags +faststart "$out"
n_out=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$out")
n_raw=$(ffmpeg -v info ${tflag[@]+"${tflag[@]}"} -i "$raw" -map 0:v:0 -f null - 2>&1 | grep -o 'frame= *[0-9]*' | tail -1 | tr -dc 0-9)
[ "$n_out" = "$n_raw" ] || { echo "FRAME COUNT MISMATCH raw=$n_raw proxy=$n_out" >&2; exit 1; }
echo "ok $out frames=$n_out size=$(du -h "$out" | cut -f1) (raw $(du -h "$raw" | cut -f1))"
