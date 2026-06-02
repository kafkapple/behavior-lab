#!/usr/bin/env bash
# DLC ResNet50 (ImageNet init) train on MAMMAL M1 pseudo-GT.
#
# - Supervision = MAMMAL kp_3d projected to per-cam 2D (via label3d cams)
# - Training frames = mammal_m1_train.csv (2880 video frame indices)
# - Output project on /node_data/joon to avoid NFS write hangs (CLAUDE.md §3.1)
#
# Usage (from behavior-lab repo root, on gpu03):
#   CUDA_VISIBLE_DEVICES=4 bash scripts/01_train_dlc_resnet50.sh
#
# Time budget: ~30–60 min on idle Blackwell RTX PRO 6000.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dlc3}"
GPU="${CUDA_VISIBLE_DEVICES:-4}"
PROJECT_ROOT="${PROJECT_ROOT:-/node_data/joon/behavior-lab-kp-benchmark}"
DATA_ROOT="${DATA_ROOT:-/home/joon/dev/behavior-lab/data}"
REPO_ROOT="${REPO_ROOT:-/home/joon/dev/behavior-lab}"
VIDEO_DIR="${VIDEO_DIR:-/node_data/joon/data/raw/markerless_mouse_1_nerf/videos_undist}"
NET_TYPE="resnet_50"
TAG="dlc_resnet50_imagenet"

echo "=== [01_train_dlc_resnet50] start $(date -Iseconds) ==="
echo "  CONDA_ENV    = $CONDA_ENV"
echo "  GPU          = $GPU"
echo "  PROJECT_ROOT = $PROJECT_ROOT"
echo "  NET_TYPE     = $NET_TYPE"

source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$PROJECT_ROOT"
cd "$REPO_ROOT"

python - <<PYEOF
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
import pandas as pd
import deeplabcut as dlc
from behavior_lab.data.loaders.li2023 import Li2023Loader
from scripts.render_kp_overlay import load_cameras, project, KP_NAMES

PROJECT_ROOT = Path("$PROJECT_ROOT")
DATA_ROOT = Path("$DATA_ROOT")
TAG = "$TAG"
NET_TYPE = "$NET_TYPE"

# 1) Load camera params + MAMMAL kp3d + train split
cams = load_cameras(DATA_ROOT / "markerless_mouse_1/labels/label3d_dannce.mat")
mam = np.load(DATA_ROOT / "mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
mammal_kp = mam["keypoints"].astype(np.float64)         # (3600, 22, 3)
mammal_idx = mam["frame_indices"]                       # (3600,)
train_ids = pd.read_csv(DATA_ROOT / "splits/mammal_m1_train.csv")["frame_id"].values

video_dir = Path("$VIDEO_DIR")
videos = [str(video_dir / f"{i}.mp4") for i in range(6)]
missing = [v for v in videos if not Path(v).exists()]
assert not missing, f"missing video: {missing}"

# 2) Create DLC project (single-animal, 22 kp)
project_name = f"kp_benchmark_{TAG}"
config_path = dlc.create_new_project(
    project_name, "joon", videos,
    working_directory=str(PROJECT_ROOT),
    copy_videos=False,
    multianimal=False,
)
print(f"[dlc] project config: {config_path}")

# 3) Edit config: bodyparts + skeleton + train_fraction
from deeplabcut.utils.auxiliaryfunctions import read_config, write_config
cfg = read_config(config_path)
cfg["bodyparts"] = list(KP_NAMES)
cfg["TrainingFraction"] = [0.95]   # we do explicit split outside
cfg["numframes2pick"] = 0           # we provide frames manually
write_config(config_path, cfg)

# 4) Generate labels per video (project MAMMAL 3D → 2D per cam, only at train_ids)
import h5py
labeled_dir = Path(config_path).parent / "labeled-data"
labeled_dir.mkdir(exist_ok=True)
common_train = np.intersect1d(train_ids, mammal_idx)
print(f"[dlc] training frames (train∩MAMMAL): {len(common_train)} / {len(train_ids)}")

# DLC label format: labeled-data/{video_stem}/CollectedData_joon.h5 with columns
# MultiIndex (scorer, bodypart, x/y). Pixel coords.
import cv2
for cam_i in range(6):
    vid_stem = Path(videos[cam_i]).stem
    vid_dir = labeled_dir / vid_stem
    vid_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(videos[cam_i])
    rows = []
    img_paths = []
    for fi in common_train:
        i = int(np.where(mammal_idx == fi)[0][0])
        pts3d = mammal_kp[i]
        pts2d = project(pts3d, cams[cam_i])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        img_name = f"img{int(fi):06d}.png"
        img_path = vid_dir / img_name
        cv2.imwrite(str(img_path), frame)
        row = []
        for k in range(22):
            x, y = pts2d[k]
            row.extend([x if np.isfinite(x) else np.nan,
                        y if np.isfinite(y) else np.nan])
        rows.append(row)
        img_paths.append(f"labeled-data/{vid_stem}/{img_name}")
    cap.release()
    if not rows:
        continue
    cols = pd.MultiIndex.from_product(
        [["joon"], KP_NAMES, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df = pd.DataFrame(rows, columns=cols, index=img_paths)
    h5_path = vid_dir / "CollectedData_joon.h5"
    df.to_hdf(h5_path, key="df_with_missing", mode="w")
    df.to_csv(vid_dir / "CollectedData_joon.csv")
    print(f"[dlc] cam{cam_i}: wrote {len(rows)} labeled frames")

# 5) create_training_dataset + train_network
dlc.create_training_dataset(config_path, num_shuffles=1, net_type=NET_TYPE)
dlc.train_network(
    config_path, shuffle=1, displayiters=200, saveiters=2000,
    maxiters=20000,                  # short run to demo
    allow_growth=True,
)
print(f"=== [01_train_dlc_resnet50] done $(date -Iseconds) ===")
PYEOF
