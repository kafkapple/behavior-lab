"""Apply Savitzky-Golay temporal smoothing to full_kp.npz.

Per-keypoint × per-axis 1D smoothing. NaN positions are preserved (filled
with linear interpolation for smoothing, restored to NaN in output).

Window=15 frames @ 100fps = 150ms — short enough to preserve real mouse
movement, long enough to suppress per-frame DLT noise.

Output: {tag}_full_kp_smoothed.npz with same schema as input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED = REPO_ROOT / "outputs/kp_benchmark"

WINDOW = 15
ORDER = 3


def smooth_one_axis(x: np.ndarray, window: int = WINDOW, order: int = ORDER) -> np.ndarray:
    """1D smooth with NaN preservation."""
    mask = np.isfinite(x)
    if mask.sum() < window:
        return x.copy()
    idx = np.arange(len(x))
    # Linear-interp NaN gaps so SavGol has continuous input
    filled = np.interp(idx, idx[mask], x[mask])
    sm = savgol_filter(filled, window_length=window, polyorder=order)
    sm[~mask] = np.nan
    return sm


def smooth_kp(kp: np.ndarray) -> np.ndarray:
    """kp: (T, K, 3) → smoothed (T, K, 3)."""
    out = np.empty_like(kp)
    T, K, D = kp.shape
    for k in range(K):
        for axis in range(D):
            out[:, k, axis] = smooth_one_axis(kp[:, k, axis])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["dlc_resnet50_imagenet",
                             "dlc_superanimal_zeroshot_hrnet_w32"])
    args = ap.parse_args()

    for tag in args.tags:
        src = PRED / f"{tag}_full_kp.npz"
        if not src.exists():
            print(f"skip: {src} not found"); continue
        d = np.load(src)
        kp = d["keypoints_3d"].astype(np.float64)
        before_nan = float(np.isnan(kp).mean())
        sm = smooth_kp(kp)
        after_nan = float(np.isnan(sm).mean())
        # Per-frame jitter reduction proxy: std of per-frame velocity magnitude
        diff_before = np.diff(kp, axis=0)
        diff_after = np.diff(sm, axis=0)
        vel_before = np.nanmean(np.linalg.norm(diff_before, axis=-1))
        vel_after = np.nanmean(np.linalg.norm(diff_after, axis=-1))

        out = PRED / f"{tag}_full_kp_smoothed.npz"
        np.savez(out,
                 keypoints_3d=sm.astype(np.float32),
                 valid_mask=d["valid_mask"],
                 frame_ids=d["frame_ids"],
                 keypoint_names=d["keypoint_names"],
                 smoothing_window=WINDOW,
                 smoothing_order=ORDER)
        print(f"{tag}: NaN {before_nan:.3f}→{after_nan:.3f} | "
              f"mean velocity {vel_before:.3f}→{vel_after:.3f} mm/fr "
              f"({(1 - vel_after / vel_before) * 100:.1f}% jitter reduction)")
        print(f"  saved → {out.name}")


if __name__ == "__main__":
    main()
