#!/usr/bin/env bash
# DLC inference for kp_benchmark v0.1.
#
# For a trained DLC project (per tag), runs analyze_videos on all 6 cameras,
# then triangulates 2D detections to 3D world coordinates using the original
# label3d_dannce.mat camera params. Saves predictions npz for both held-out
# test split and Li GT external frames.
#
# Usage:
#   TAG=dlc_resnet50_imagenet    bash scripts/03_infer_dlc.sh
#   TAG=dlc_superanimal_hrnet_w32 bash scripts/03_infer_dlc.sh
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dlc3}"
GPU="${CUDA_VISIBLE_DEVICES:-4}"
PROJECT_ROOT="${PROJECT_ROOT:-/node_data/joon/behavior-lab-kp-benchmark}"
DATA_ROOT="${DATA_ROOT:-/home/joon/dev/behavior-lab-kp/data}"
REPO_ROOT="${REPO_ROOT:-/home/joon/dev/behavior-lab-kp}"
VIDEO_DIR="${VIDEO_DIR:-/node_data/joon/data/raw/markerless_mouse_1_nerf/videos_undist}"
TAG="${TAG:-dlc_resnet50_imagenet}"
OUT_DIR="${OUT_DIR:-/node_data/joon/behavior-lab-kp-benchmark/predictions}"

echo "=== [03_infer_dlc TAG=$TAG] start $(date -Iseconds) ==="

source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

python - <<PYEOF
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import deeplabcut as dlc
from render_kp_overlay import load_cameras, KP_NAMES

PROJECT_ROOT = Path("$PROJECT_ROOT")
DATA_ROOT = Path("$DATA_ROOT")
OUT_DIR = Path("$OUT_DIR")
TAG = "$TAG"
VIDEO_DIR = Path("$VIDEO_DIR")

# 1) Locate the trained DLC config.yaml under PROJECT_ROOT/kp_benchmark_TAG-*
candidates = sorted(PROJECT_ROOT.glob(f"kp_benchmark_{TAG}-*/config.yaml"))
assert candidates, f"no trained DLC project at {PROJECT_ROOT}/kp_benchmark_{TAG}-*"
config_path = str(candidates[0])
print(f"[infer] config: {config_path}")

videos = [str(VIDEO_DIR / f"{i}.mp4") for i in range(6)]
dest = OUT_DIR / f"{TAG}_analyze"
dest.mkdir(parents=True, exist_ok=True)

# 2) Per-cam inference (2D + likelihood). DLC 3.0 saves CSV/H5 per video.
dlc.analyze_videos(
    config_path, videos, shuffle=1, save_as_csv=True,
    destfolder=str(dest), gputouse=$GPU,
)

# 3) Load test + Li frame ids
test_ids = pd.read_csv(DATA_ROOT / "splits/mammal_m1_test.csv")["frame_id"].values
li_ids   = pd.read_csv(DATA_ROOT / "splits/li_m1_external.csv")["frame_id"].values

# 4) Read per-cam DLC 2D outputs (h5 in destfolder)
cam_dfs = []
for cam_i in range(6):
    h5s = sorted(dest.glob(f"{cam_i}DLC_*.h5"))
    assert h5s, f"missing DLC h5 for cam {cam_i} in {dest}"
    cam_dfs.append(pd.read_hdf(h5s[0]))
    print(f"[infer] cam{cam_i}: {len(cam_dfs[-1])} frames, columns: {len(cam_dfs[-1].columns)//3} kp")

# 5) Load camera params for triangulation
cams = load_cameras(DATA_ROOT / "markerless_mouse_1/labels/label3d_dannce.mat")

def projection_matrices():
    return [c["K"] @ np.hstack([c["R"], c["t"].reshape(3, 1)]) for c in cams]
Ps = projection_matrices()

def triangulate_one(pts2d_per_cam, prob_per_cam, prob_min=0.3):
    A = []
    for P, p, q in zip(Ps, pts2d_per_cam, prob_per_cam):
        if not np.all(np.isfinite(p)) or q < prob_min:
            continue
        x, y = p
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    if len(A) < 4:
        return np.full(3, np.nan)
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-9:
        return np.full(3, np.nan)
    return X[:3] / X[3]

def triangulate_split(frame_ids):
    out = np.full((len(frame_ids), 22, 3), np.nan, dtype=np.float64)
    bodyparts = list(KP_NAMES)
    for i, fi in enumerate(frame_ids):
        for k, bp in enumerate(bodyparts):
            pts2d, probs = [], []
            for cam_i in range(6):
                df = cam_dfs[cam_i]
                if fi >= len(df):
                    pts2d.append(np.full(2, np.nan)); probs.append(0.0); continue
                row = df.iloc[fi]
                cols = [c for c in df.columns if c[-2] == bp]
                if not cols:
                    cols = [(df.columns[0][0], bp, "x"),
                            (df.columns[0][0], bp, "y"),
                            (df.columns[0][0], bp, "likelihood")]
                try:
                    x = float(row[(df.columns[0][0], bp, "x")])
                    y = float(row[(df.columns[0][0], bp, "y")])
                    q = float(row[(df.columns[0][0], bp, "likelihood")])
                except KeyError:
                    x = y = np.nan; q = 0.0
                pts2d.append(np.array([x, y])); probs.append(q)
            out[i, k] = triangulate_one(pts2d, probs)
    return out

test_pred = triangulate_split(test_ids)
li_pred = triangulate_split(li_ids)

np.savez(
    OUT_DIR / f"{TAG}_test_pred.npz",
    keypoints_3d=test_pred.astype(np.float32),
    frame_ids=np.asarray(test_ids, dtype=np.int64),
)
np.savez(
    OUT_DIR / f"{TAG}_li_pred.npz",
    keypoints_3d=li_pred.astype(np.float32),
    frame_ids=np.asarray(li_ids, dtype=np.int64),
)
n_test_valid = int(np.sum(~np.isnan(test_pred).any(axis=(1,2))))
n_li_valid = int(np.sum(~np.isnan(li_pred).any(axis=(1,2))))
print(f"[infer] saved: {TAG}_test_pred.npz (valid {n_test_valid}/{len(test_ids)})")
print(f"[infer] saved: {TAG}_li_pred.npz   (valid {n_li_valid}/{len(li_ids)})")
print(f"=== [03_infer_dlc TAG=$TAG] done $(date -Iseconds) ===")
PYEOF
