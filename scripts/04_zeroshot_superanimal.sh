#!/usr/bin/env bash
# SuperAnimal-TopViewMouse zero-shot inference on M1 video frames.
#
# v0.1.1 fallback for the SuperAnimal arm: DLC modelzoo zero-shot
# (no fine-tuning) on the same 6 cameras → 27-kp 2D per frame, then
# pick the 14 keypoints that share names with our MAMMAL 22-kp schema
# and triangulate to 3D.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6 bash scripts/04_zeroshot_superanimal.sh
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dlc3}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
REPO_ROOT="${REPO_ROOT:-/home/joon/dev/behavior-lab-kp}"
DATA_ROOT="${DATA_ROOT:-/home/joon/dev/behavior-lab-kp/data}"
VIDEO_DIR="${VIDEO_DIR:-/node_data/joon/data/raw/markerless_mouse_1_nerf/videos_undist}"
OUT_DIR="${OUT_DIR:-/node_data/joon/behavior-lab-kp-benchmark/predictions}"
TAG="dlc_superanimal_zeroshot_hrnet_w32"

echo "=== [04_zeroshot_superanimal] start $(date -Iseconds) ==="

source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT_DIR/${TAG}_analyze"
cd "$REPO_ROOT"

python - <<PYEOF
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from pathlib import Path
import numpy as np
import pandas as pd
import deeplabcut as dlc
from render_kp_overlay import load_cameras

OUT_DIR = Path("$OUT_DIR")
TAG = "$TAG"
VIDEO_DIR = Path("$VIDEO_DIR")
DATA_ROOT = Path("$DATA_ROOT")

videos = [str(VIDEO_DIR / f"{i}.mp4") for i in range(6)]
dest = OUT_DIR / f"{TAG}_analyze"

# 1) Zero-shot inference — downloads SuperAnimal-TopViewMouse weights on first call
dlc.video_inference_superanimal(
    videos=videos,
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_mobilenet_v3_large_fpn",
    videotype=".mp4",
    dest_folder=str(dest),
    scale_list=[],
)
print(f"[zeroshot] inference complete → {dest}")

# 2) Read per-cam h5 outputs, extract SuperAnimal 27-kp 2D + likelihood
cam_dfs = []
for cam_i in range(6):
    h5s = sorted(dest.glob(f"{cam_i}*superanimal*.h5"))
    if not h5s:
        h5s = sorted(dest.glob(f"{cam_i}*.h5"))
    assert h5s, f"no h5 for cam {cam_i}"
    cam_dfs.append(pd.read_hdf(h5s[0]))
    cols = cam_dfs[-1].columns
    bps = sorted(set(c[1] for c in cols if c[-1] in ("x", "y", "likelihood")))
    print(f"[zeroshot] cam{cam_i}: {len(cam_dfs[-1])} frames, {len(bps)} bps: {bps[:6]}...")

# 3) SuperAnimal 27-kp → MAMMAL 22-kp name overlap (used for fair comparison)
# Conservative mapping based on standard mouse anatomy labels
NAME_MAP = {
    "nose": "nose", "left_ear": "L_ear", "right_ear": "R_ear",
    "neck_base": "neck", "throat_base": "neck",
    "spine_2": "body_middle", "back_middle": "body_middle",
    "tail_base": "tail_root", "tail_middle": "tail_middle", "tail_end": "tail_end", "tail_tip": "tail_end",
    "left_front_paw": "L_paw", "right_front_paw": "R_paw",
    "left_back_paw": "L_foot", "right_back_paw": "R_foot",
    "left_shoulder": "L_shoulder", "right_shoulder": "R_shoulder",
    "left_hip": "L_hip", "right_hip": "R_hip",
    "left_knee": "L_knee", "right_knee": "R_knee",
    "left_elbow": "L_elbow", "right_elbow": "R_elbow",
}

cams = load_cameras(DATA_ROOT / "markerless_mouse_1/labels/label3d_dannce.mat")
Ps = [c["K"] @ np.hstack([c["R"], c["t"].reshape(3, 1)]) for c in cams]

def triangulate(pts2d_per_cam, prob_per_cam, prob_min=0.3):
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
    return X[:3] / X[3] if abs(X[3]) > 1e-9 else np.full(3, np.nan)

MAMMAL_KP = ["L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
             "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow",
             "L_shoulder", "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
             "L_foot", "L_knee", "L_hip", "R_foot", "R_knee", "R_hip"]

# Reverse: MAMMAL kp name → SuperAnimal kp name
SA_KP = {v: k for k, v in NAME_MAP.items()}

def get_2d_likelihood(df, cam_i, frame_i, sa_kp):
    # DLC h5 columns: (scorer, bodypart, [x|y|likelihood])
    scorer = df.columns[0][0]
    try:
        row = df.iloc[frame_i]
        return (float(row[(scorer, sa_kp, "x")]),
                float(row[(scorer, sa_kp, "y")]),
                float(row[(scorer, sa_kp, "likelihood")]))
    except (KeyError, IndexError):
        return np.nan, np.nan, 0.0

def triangulate_split(frame_ids):
    out = np.full((len(frame_ids), 22, 3), np.nan, dtype=np.float64)
    for i, fi in enumerate(frame_ids):
        for k_idx, mammal_name in enumerate(MAMMAL_KP):
            sa_name = SA_KP.get(mammal_name)
            if sa_name is None:
                continue
            pts2d, probs = [], []
            for cam_i in range(6):
                x, y, q = get_2d_likelihood(cam_dfs[cam_i], cam_i, int(fi), sa_name)
                pts2d.append(np.array([x, y]))
                probs.append(q)
            out[i, k_idx] = triangulate(pts2d, probs)
    return out

test_ids = pd.read_csv(DATA_ROOT / "splits/mammal_m1_test.csv")["frame_id"].values
li_ids   = pd.read_csv(DATA_ROOT / "splits/li_m1_external.csv")["frame_id"].values

test_pred = triangulate_split(test_ids)
li_pred = triangulate_split(li_ids)

mapped_kps = [k for k in MAMMAL_KP if k in SA_KP]
np.savez(
    OUT_DIR / f"{TAG}_test_pred.npz",
    keypoints_3d=test_pred.astype(np.float32),
    frame_ids=np.asarray(test_ids, dtype=np.int64),
    mapped_kp_names=np.array(mapped_kps),
)
np.savez(
    OUT_DIR / f"{TAG}_li_pred.npz",
    keypoints_3d=li_pred.astype(np.float32),
    frame_ids=np.asarray(li_ids, dtype=np.int64),
    mapped_kp_names=np.array(mapped_kps),
)
print(f"[zeroshot] {len(mapped_kps)} kp mappable from SA 27 → MAMMAL 22")
print(f"[zeroshot] saved test_pred ({test_pred.shape}, NaN frac {np.isnan(test_pred).mean():.2f})")
print(f"[zeroshot] saved li_pred  ({li_pred.shape},   NaN frac {np.isnan(li_pred).mean():.2f})")
print(f"=== [04_zeroshot_superanimal] done $(date -Iseconds) ===")
PYEOF
