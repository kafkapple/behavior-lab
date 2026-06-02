#!/usr/bin/env bash
# DLC HRNet-w32 (SuperAnimal-TopViewMouse init) train on MAMMAL M1 pseudo-GT.
#
# Mirrors 01_train_dlc_resnet50.sh but with SuperAnimal pretrained backbone.
# Same MAMMAL train split → isolates the pretraining effect.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=5 bash scripts/02_train_dlc_superanimal.sh
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dlc3}"
GPU="${CUDA_VISIBLE_DEVICES:-5}"
PROJECT_ROOT="${PROJECT_ROOT:-/node_data/joon/behavior-lab-kp-benchmark}"
DATA_ROOT="${DATA_ROOT:-/home/joon/dev/behavior-lab/data}"
REPO_ROOT="${REPO_ROOT:-/home/joon/dev/behavior-lab}"
VIDEO_DIR="${VIDEO_DIR:-/node_data/joon/data/raw/markerless_mouse_1_nerf/videos_undist}"
NET_TYPE="hrnet_w32"
TAG="dlc_superanimal_hrnet_w32"
SUPERANIMAL="superanimal_topviewmouse"

echo "=== [02_train_dlc_superanimal] start $(date -Iseconds) ==="
echo "  CONDA_ENV    = $CONDA_ENV"
echo "  GPU          = $GPU"
echo "  NET_TYPE     = $NET_TYPE"
echo "  SUPERANIMAL  = $SUPERANIMAL"

source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$PROJECT_ROOT"
cd "$REPO_ROOT"

# The supervision/label generation logic is identical to 01_*. We share via
# a single Python wrapper that takes net_type as an argument.
python - <<PYEOF
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

# Reuse the heredoc logic from 01_train_dlc_resnet50.sh by invoking the same
# project-creation + labeling sequence, then call modelzoo fine-tune APIs.
# Kept separate to log distinctly + allow independent retries.

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import deeplabcut as dlc
from deeplabcut.utils.auxiliaryfunctions import read_config, write_config
from behavior_lab.data.loaders.li2023 import Li2023Loader
from scripts.render_kp_overlay import load_cameras, project, KP_NAMES

PROJECT_ROOT = Path("$PROJECT_ROOT")
DATA_ROOT = Path("$DATA_ROOT")
NET_TYPE = "$NET_TYPE"
TAG = "$TAG"
SUPERANIMAL = "$SUPERANIMAL"

cams = load_cameras(DATA_ROOT / "markerless_mouse_1/labels/label3d_dannce.mat")
mam = np.load(DATA_ROOT / "mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
mammal_kp = mam["keypoints"].astype(np.float64)
mammal_idx = mam["frame_indices"]
train_ids = pd.read_csv(DATA_ROOT / "splits/mammal_m1_train.csv")["frame_id"].values

video_dir = Path("$VIDEO_DIR")
videos = [str(video_dir / f"{i}.mp4") for i in range(6)]
missing = [v for v in videos if not Path(v).exists()]
assert not missing, f"missing video: {missing}"

project_name = f"kp_benchmark_{TAG}"
config_path = dlc.create_new_project(
    project_name, "joon", videos,
    working_directory=str(PROJECT_ROOT),
    copy_videos=False, multianimal=False,
)
cfg = read_config(config_path)
cfg["bodyparts"] = list(KP_NAMES)
cfg["TrainingFraction"] = [0.95]
cfg["numframes2pick"] = 0
write_config(config_path, cfg)

labeled_dir = Path(config_path).parent / "labeled-data"
labeled_dir.mkdir(exist_ok=True)
common_train = np.intersect1d(train_ids, mammal_idx)
for cam_i in range(6):
    vid_stem = Path(videos[cam_i]).stem
    vid_dir = labeled_dir / vid_stem
    vid_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(videos[cam_i])
    rows, img_paths = [], []
    for fi in common_train:
        i = int(np.where(mammal_idx == fi)[0][0])
        pts2d = project(mammal_kp[i], cams[cam_i])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        img_name = f"img{int(fi):06d}.png"
        cv2.imwrite(str(vid_dir / img_name), frame)
        row = []
        for k in range(22):
            x, y = pts2d[k]
            row.extend([x if np.isfinite(x) else np.nan,
                        y if np.isfinite(y) else np.nan])
        rows.append(row)
        img_paths.append(f"labeled-data/{vid_stem}/{img_name}")
    cap.release()
    if rows:
        cols = pd.MultiIndex.from_product(
            [["joon"], KP_NAMES, ["x", "y"]],
            names=["scorer", "bodyparts", "coords"])
        df = pd.DataFrame(rows, columns=cols, index=img_paths)
        df.to_hdf(vid_dir / "CollectedData_joon.h5",
                  key="df_with_missing", mode="w")
        df.to_csv(vid_dir / "CollectedData_joon.csv")
        print(f"[dlc] cam{cam_i}: {len(rows)} labeled frames")

# SuperAnimal fine-tune path: load modelzoo weights, attach to new project
dlc.create_training_dataset(
    config_path, num_shuffles=1, net_type=NET_TYPE,
    weight_init=f"modelzoo:{SUPERANIMAL}",
)
dlc.train_network(
    config_path, shuffle=1, displayiters=200, saveiters=2000,
    maxiters=20000, allow_growth=True,
)
print(f"=== [02_train_dlc_superanimal] done $(date -Iseconds) ===")
PYEOF
