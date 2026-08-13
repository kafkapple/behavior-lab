#!/bin/bash
# SBeA 원본 영상 취득 (figshare 직링크). resume-safe — 크기가 맞는 파일은 건너뛴다.
#
#   ./download_sbea.sh <individual|social> <target_dir> [n_workers]
#
# 260814: gpu03 홈에 흩어져 있던 download_sbea.sh + download_sbea_social_parallel.sh 를
# 하나로 합치고 하드코딩 경로(/node_data_2/joon/…, /home/joon/…)를 인자로 뺐다.
# individual = 100파일 ~11.2 GB · social = 150파일 ~16.8 GB.
set -euo pipefail

SPLIT=${1:?"usage: $0 <individual|social> <target_dir> [n_workers]"}
TARGET=${2:?"usage: $0 <individual|social> <target_dir> [n_workers]"}
N_WORKERS=${3:-4}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TSV="$HERE/sbea_${SPLIT}_files.tsv"
[ -f "$TSV" ] || { echo "no manifest: $TSV"; exit 1; }

mkdir -p "$TARGET"
total=$(wc -l < "$TSV")
echo "[$(date)] $SPLIT: $total files -> $TARGET ($N_WORKERS workers)"

fetch_chunk() {
    while IFS=$'\t' read -r name size url; do
        out="$TARGET/$name"
        if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")" -eq "$size" ]; then
            echo "SKIP $name"; continue
        fi
        echo "DL   $name ($((size/1048576)) MB)"
        wget -q -O "$out" "$url" || { echo "FAIL $name"; rm -f "$out"; }
    done < "$1"
}

SPLITDIR=$(mktemp -d)
trap 'rm -rf "$SPLITDIR"' EXIT
split -l $((total / N_WORKERS + 1)) "$TSV" "$SPLITDIR/chunk_"
for c in "$SPLITDIR"/chunk_*; do fetch_chunk "$c" & done
wait

echo "[$(date)] done. mp4=$(ls "$TARGET"/*.mp4 2>/dev/null | wc -l) / expected=$(grep -c '\.mp4' "$TSV")"
