"""Triangulate DLC RN50 2D detections via Anipose (aniposelib).

Three methods compared:
  1. anipose_linear   — same as ours (linear DLT, prob_min thresholding)
  2. anipose_ransac   — RANSAC view selection
  3. anipose_optim    — point optimization (bundle-style)

Output: outputs/kp_benchmark/anipose_rn50_{method}.npz with shape (18000, 22, 3).

Then benchmark vs MAMMAL pseudo-GT + Li GT, save results.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat

from aniposelib.cameras import Camera, CameraGroup

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from render_kp_overlay import KP_NAMES

DATA = REPO / "data"
PRED = REPO / "outputs/kp_benchmark"
H5_DIR = PRED / "dlc_h5_rn50"
PROB_MIN = 0.10

# 1) Build aniposelib CameraGroup from label3d_dannce.mat
mat = loadmat(DATA / "markerless_mouse_1/labels/label3d_dannce.mat",
              struct_as_record=False, squeeze_me=True)
cam_names = [str(n) for n in mat["camnames"]]
cameras = []
for i, name in enumerate(cam_names):
    p = mat["params"][i]
    p = p.item() if hasattr(p, "item") else p
    K = np.asarray(p.K, dtype=np.float64).T          # OpenCV form
    R = np.asarray(p.r, dtype=np.float64).T          # w2c
    t = np.asarray(p.t, dtype=np.float64).reshape(3)
    if np.linalg.det(R) < 0:
        R = R.copy(); R[:, 2] *= -1
    # videos_undist → no distortion needed downstream
    dist = np.zeros(5, dtype=np.float64)
    rvec = cv2.Rodrigues(R)[0].flatten()
    cam = Camera(name=name,
                 matrix=K,
                 dist=dist,
                 rvec=rvec,
                 tvec=t,
                 size=(1152, 1024))
    cameras.append(cam)
cgroup = CameraGroup(cameras)
print(f"[anipose] loaded {len(cameras)} cameras")

# 2) Load DLC 2D h5s in correct cam order
h5s = sorted(H5_DIR.glob("*.h5"), key=lambda p: int(p.name[0]))
assert len(h5s) == 6, f"expected 6, got {len(h5s)}"

# Stack as (n_cams, n_frames, n_kps, 2) with NaN where prob < threshold
N_KP = 22
points_2d_list = []
likelihoods_list = []
for cam_i, h in enumerate(h5s):
    df = pd.read_hdf(h)
    scorer = df.columns[0][0]
    n_frames = len(df)
    pts = np.full((n_frames, N_KP, 2), np.nan, dtype=np.float64)
    probs = np.zeros((n_frames, N_KP), dtype=np.float64)
    for k, name in enumerate(KP_NAMES):
        try:
            pts[:, k, 0] = df[(scorer, name, "x")].values
            pts[:, k, 1] = df[(scorer, name, "y")].values
            probs[:, k] = df[(scorer, name, "likelihood")].values
        except KeyError:
            pass
    # Threshold
    low = probs < PROB_MIN
    pts[low] = np.nan
    points_2d_list.append(pts)
    likelihoods_list.append(probs)
    print(f"[anipose] cam{cam_i}: {n_frames} frames, valid kp ratio "
          f"{(~np.isnan(pts).any(axis=-1)).mean():.2f}")

points_2d = np.stack(points_2d_list)        # (6, 18000, 22, 2)
print(f"[anipose] points_2d shape: {points_2d.shape}")

# 3) Triangulate — three methods
N_FRAMES = points_2d.shape[1]
# Reshape per aniposelib's expected shape: (n_cams, n_total_points, 2) where
# n_total_points = n_frames * n_kp. Then reshape result.
pts_flat = points_2d.reshape(6, -1, 2)      # (6, 18000*22, 2)
print(f"[anipose] triangulating {pts_flat.shape[1]} points...")

# Linear DLT
p3d_linear = cgroup.triangulate(pts_flat, undistort=False)
p3d_linear = p3d_linear.reshape(N_FRAMES, N_KP, 3)
print(f"[anipose] linear done, NaN frac {np.isnan(p3d_linear).mean():.3f}")

# RANSAC
try:
    p3d_ransac = cgroup.triangulate_ransac(pts_flat, undistort=False,
                                            min_cams=3)
    p3d_ransac = p3d_ransac.reshape(N_FRAMES, N_KP, 3)
    print(f"[anipose] ransac done, NaN frac {np.isnan(p3d_ransac).mean():.3f}")
except Exception as e:
    print(f"[anipose] ransac failed: {e}")
    p3d_ransac = None

# Save
np.savez(PRED / "anipose_rn50_linear_full_kp.npz",
         keypoints_3d=p3d_linear.astype(np.float32),
         frame_ids=np.arange(N_FRAMES, dtype=np.int64),
         keypoint_names=np.array(KP_NAMES))
print(f"[anipose] saved linear → anipose_rn50_linear_full_kp.npz")

if p3d_ransac is not None:
    np.savez(PRED / "anipose_rn50_ransac_full_kp.npz",
             keypoints_3d=p3d_ransac.astype(np.float32),
             frame_ids=np.arange(N_FRAMES, dtype=np.int64),
             keypoint_names=np.array(KP_NAMES))
    print(f"[anipose] saved ransac → anipose_rn50_ransac_full_kp.npz")
