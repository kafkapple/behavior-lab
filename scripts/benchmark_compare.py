"""Comprehensive benchmark: 5 methods × 2 splits.

Methods:
  1. RN50 raw (custom DLT, prob_min=0.10)
  2. RN50 smoothed (Savitzky-Golay window=15)
  3. RN50 + RANSAC (view-selection robust, no smoothing)
  4. RN50 + linear DLT (anipose vectorized — should match #1)
  5. MAMMAL mesh-fit direct (vs Li intersect=17)
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/joon/dev/behavior-lab")
DATA = REPO / "data"
PRED = REPO / "outputs/kp_benchmark"
ROOT = 4

mam = np.load(DATA / "mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
mam_kp = mam["keypoints"].astype(np.float64); mam_idx = mam["frame_indices"]
li = np.load(DATA / "markerless_mouse_1/labels/li_m1_gt.npz")
li_kp = li["keypoints_3d"].astype(np.float64); li_ids = li["frame_ids"]
li_valid = li["valid_mask"]

def err(p, g):
    pr = p - p[:, ROOT:ROOT+1, :]
    gr = g - g[:, ROOT:ROOT+1, :]
    return np.linalg.norm(pr - gr, axis=-1)

def boot(vals, n_boot=10000, ci=0.95, seed=42):
    v = vals[np.isfinite(vals)]
    if v.size == 0: return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = v[idx].mean(axis=1)
    a = (1-ci)/2
    return float(v.mean()), float(np.quantile(m, a)), float(np.quantile(m, 1-a)), int(v.size)

methods = [
    ("RN50 + custom DLT (raw)",        "dlc_resnet50_imagenet_full_kp.npz"),
    ("RN50 + custom DLT (smoothed)",   "dlc_resnet50_imagenet_full_kp_smoothed.npz"),
    ("RN50 + anipose linear DLT",      "anipose_rn50_linear_full_kp.npz"),
    ("RN50 + anipose RANSAC",          "anipose_rn50_ransac_full_kp.npz"),
    ("SA zero-shot + DLT (smoothed)",  "dlc_superanimal_zeroshot_hrnet_w32_full_kp_smoothed.npz"),
]

rows = []
print(f"\n{'method':>34s} | {'split':>14s} | {'MPJPE [95% CI]':>26s} | n_valid")
print("-" * 95)
for name, fname in methods:
    full = np.load(PRED / fname)["keypoints_3d"].astype(np.float64)
    for split, gt, frames in [("mammal_3600", mam_kp, mam_idx),
                              ("li_external", li_kp, li_ids)]:
        e = err(full[frames], gt)
        pf = np.nanmean(e, axis=-1)
        m, lo, hi, n = boot(pf)
        rows.append({"method": name, "split": split, "n_valid": n,
                     "mpjpe_mean_mm": m, "ci_lo": lo, "ci_hi": hi})
        print(f"{name:>34s} | {split:>14s} | {m:7.2f} [{lo:5.2f}, {hi:5.2f}] | n={n}")

# MAMMAL direct (vs Li intersect)
common = np.intersect1d(mam_idx, li_ids)
mam_at = np.stack([mam_kp[np.where(mam_idx == c)[0][0]] for c in common])
li_at = np.stack([li_kp[np.where(li_ids == c)[0][0]] for c in common])
mask_at = np.stack([li_valid[np.where(li_ids == c)[0][0]] for c in common])
e = err(mam_at, li_at)
e_m = np.where(mask_at, e, np.nan)
pf = np.nanmean(e_m, axis=-1)
m, lo, hi, n = boot(pf)
rows.append({"method": "MAMMAL mesh-fit (direct)", "split": "li_intersect",
             "n_valid": n, "mpjpe_mean_mm": m, "ci_lo": lo, "ci_hi": hi})
print(f"{'MAMMAL mesh-fit (direct)':>34s} | {'li_intersect':>14s} | {m:7.2f} [{lo:5.2f}, {hi:5.2f}] | n={n}")

df = pd.DataFrame(rows)
df.to_csv(PRED / "results_full_method_compare.csv", index=False)
print(f"\nsaved → {PRED}/results_full_method_compare.csv")
