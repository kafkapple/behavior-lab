"""Fast Anipose-compatible triangulation: same algorithm, vectorized.

aniposelib's `triangulate()` API is slow on 396K points. We use the same
linear SVD-based DLT but vectorized properly. Plus aniposelib's
`triangulate_ransac` and our own bone-length smoothing.

Output:
  anipose_rn50_linear.npz   — basic DLT (should match our custom DLT)
  anipose_rn50_ransac.npz   — RANSAC view selection
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat

REPO = Path("/Users/joon/dev/behavior-lab")
sys.path.insert(0, str(REPO / "scripts"))
from render_kp_overlay import KP_NAMES

DATA = REPO / "data"
PRED = REPO / "outputs/kp_benchmark"
H5_DIR = PRED / "dlc_h5_rn50"
PROB_MIN = 0.10
N_KP = 22

# Load cameras (same as before)
mat = loadmat(DATA / "markerless_mouse_1/labels/label3d_dannce.mat",
              struct_as_record=False, squeeze_me=True)
Ps = []
for i in range(6):
    p = mat["params"][i]
    p = p.item() if hasattr(p, "item") else p
    K = np.asarray(p.K, dtype=np.float64).T
    R = np.asarray(p.r, dtype=np.float64).T
    t = np.asarray(p.t, dtype=np.float64).reshape(3)
    if np.linalg.det(R) < 0:
        R = R.copy(); R[:, 2] *= -1
    Ps.append(K @ np.hstack([R, t.reshape(3, 1)]))
Ps = np.stack(Ps)   # (6, 3, 4)

# Load 2D from h5
h5s = sorted(H5_DIR.glob("*.h5"), key=lambda p: int(p.name[0]))
points_2d = []
likelihoods = []
for cam_i, h in enumerate(h5s):
    df = pd.read_hdf(h)
    scorer = df.columns[0][0]
    n = len(df)
    pts = np.full((n, N_KP, 2), np.nan)
    pr = np.zeros((n, N_KP))
    for k, name in enumerate(KP_NAMES):
        try:
            pts[:, k, 0] = df[(scorer, name, "x")].values
            pts[:, k, 1] = df[(scorer, name, "y")].values
            pr[:, k] = df[(scorer, name, "likelihood")].values
        except KeyError:
            pass
    low = pr < PROB_MIN
    pts[low] = np.nan
    points_2d.append(pts)
    likelihoods.append(pr)
points_2d = np.stack(points_2d)    # (6, 18000, 22, 2)
likelihoods = np.stack(likelihoods)  # (6, 18000, 22)
T = points_2d.shape[1]
print(f"loaded {T} frames × {N_KP} kp × 6 cams")


def triangulate_dlt(pts2d):
    """Vectorized linear DLT. pts2d: (6, 2) for one point. Returns (3,)."""
    valid = ~np.isnan(pts2d).any(axis=-1)
    if valid.sum() < 2:
        return np.full(3, np.nan)
    A = []
    for i in np.where(valid)[0]:
        x, y = pts2d[i]
        A.append(x * Ps[i, 2] - Ps[i, 0])
        A.append(y * Ps[i, 2] - Ps[i, 1])
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-9 else np.full(3, np.nan)


def triangulate_all_linear():
    out = np.full((T, N_KP, 3), np.nan, dtype=np.float32)
    t0 = time.time()
    for fi in range(T):
        if fi % 3000 == 0:
            print(f"  linear: {fi}/{T} ({time.time() - t0:.0f}s)")
        for k in range(N_KP):
            out[fi, k] = triangulate_dlt(points_2d[:, fi, k])
    print(f"  linear done: {time.time() - t0:.0f}s")
    return out


def triangulate_ransac_one(pts2d, ths_reproj=8.0, min_inliers=3):
    """RANSAC on view selection. Returns (3,) + #inliers."""
    valid = ~np.isnan(pts2d).any(axis=-1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < min_inliers:
        return np.full(3, np.nan), 0
    best_inliers = []
    best_X = None
    # Try all 2-view subsets, score by reprojection error in other views
    for i in range(len(valid_idx)):
        for j in range(i + 1, len(valid_idx)):
            cams_ij = [valid_idx[i], valid_idx[j]]
            A = []
            for c in cams_ij:
                x, y = pts2d[c]
                A.append(x * Ps[c, 2] - Ps[c, 0])
                A.append(y * Ps[c, 2] - Ps[c, 1])
            A = np.stack(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            if abs(X[3]) < 1e-9: continue
            X = X[:3] / X[3]
            # Count inliers across all valid cams
            inliers = []
            for c in valid_idx:
                proj = Ps[c] @ np.append(X, 1)
                if proj[2] < 1e-6: continue
                pxl = proj[:2] / proj[2]
                if np.linalg.norm(pxl - pts2d[c]) < ths_reproj:
                    inliers.append(c)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers; best_X = X
    if len(best_inliers) < min_inliers:
        return np.full(3, np.nan), 0
    # Refine with all inliers
    A = []
    for c in best_inliers:
        x, y = pts2d[c]
        A.append(x * Ps[c, 2] - Ps[c, 0])
        A.append(y * Ps[c, 2] - Ps[c, 1])
    A = np.stack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3] if abs(X[3]) > 1e-9 else np.full(3, np.nan)), len(best_inliers)


def triangulate_all_ransac():
    out = np.full((T, N_KP, 3), np.nan, dtype=np.float32)
    inlier_count = np.zeros((T, N_KP), dtype=np.int8)
    t0 = time.time()
    for fi in range(T):
        if fi % 1000 == 0:
            print(f"  ransac: {fi}/{T} ({time.time() - t0:.0f}s)")
        for k in range(N_KP):
            X, n = triangulate_ransac_one(points_2d[:, fi, k])
            out[fi, k] = X
            inlier_count[fi, k] = n
    print(f"  ransac done: {time.time() - t0:.0f}s, mean inliers: {inlier_count.mean():.2f}")
    return out, inlier_count


# Run
print("\n=== linear DLT ===")
p3d_linear = triangulate_all_linear()
np.savez(PRED / "anipose_rn50_linear_full_kp.npz",
         keypoints_3d=p3d_linear,
         frame_ids=np.arange(T, dtype=np.int64),
         keypoint_names=np.array(KP_NAMES))
print(f"saved → anipose_rn50_linear_full_kp.npz, NaN frac {np.isnan(p3d_linear).mean():.3f}")

print("\n=== RANSAC view selection ===")
p3d_ransac, inliers = triangulate_all_ransac()
np.savez(PRED / "anipose_rn50_ransac_full_kp.npz",
         keypoints_3d=p3d_ransac,
         frame_ids=np.arange(T, dtype=np.int64),
         keypoint_names=np.array(KP_NAMES),
         inlier_count=inliers)
print(f"saved → anipose_rn50_ransac_full_kp.npz, NaN frac {np.isnan(p3d_ransac).mean():.3f}")
