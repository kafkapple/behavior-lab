"""Export our 2D keypoints to the author's DeepLabCut csv layout, one file per camera.

Why: the SBeA tracker's 3D reconstruction is callable directly —
`merge_3d_poses.triangulate_batch(csvlist, caliparam, camera_num, tempname3d)` takes 2D csvs
plus the calibration and needs neither segmentation, identity, nor the 2-day training stage.
Feeding it OUR 2D lets us compare the author's 3D reconstruction against our DLT on identical
input, which is what S2 always wanted and previously looked unaffordable.

Layout is copied from the author's own files
(`sbea_release_assets/SM_fig1_data/gt_data/*.csv`), three header rows then one row per frame:

    scorer,<scorer>,<scorer>,...            (3 columns per joint)
    bodyparts,nose,nose,nose,left_ear,...
    coords,x,y,likelihood,x,y,likelihood,...
    0,466.75,247.77,0.9999,...

Usage:
    python scripts/sbea_export_dlc_csv.py --npz <session>-kp3d.npz --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# The author's own predictions carry this scorer string; keeping it byte-identical avoids any
# chance that their loader keys off the name.
SCORER = "DLC_resnet50_Mouse2DprojectApr11shuffle1_1030000"


def write_camera_csv(path: Path, kp: np.ndarray, names: list[str], frames: np.ndarray) -> None:
    """kp = (T, K, 3) x/y/likelihood for one camera."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scorer"] + [SCORER] * (3 * len(names)))
        w.writerow(["bodyparts"] + [n for n in names for _ in range(3)])
        w.writerow(["coords"] + ["x", "y", "likelihood"] * len(names))
        for t in range(kp.shape[0]):
            row: list = [int(frames[t])]
            for k in range(kp.shape[1]):
                x, y, q = kp[t, k]
                # NaN would break their parser; the author sets low-confidence points to null
                # by thresholding, so emit 0 coordinates with 0 likelihood instead.
                row += ["", "", 0.0] if not np.isfinite([x, y]).all() else [x, y, q]
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    kp2d = z["keypoints_2d"]                       # (C, T, K, 3)
    names = [str(n) for n in z["keypoint_names"]]
    frames = z["frame_indices"]
    order = [int(i) for i in z["camera_order"]]
    session = a.npz.stem.replace("-kp3d", "")

    if a.max_frames:
        kp2d, frames = kp2d[:, :a.max_frames], frames[:a.max_frames]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for c in range(kp2d.shape[0]):
        # Camera index in the filename is ours (matches <session>-camera-{c}.mp4). The npz
        # already stores P_matrices in this same order, so the pairing carries over.
        out = a.out_dir / f"{session}-camera-{c}.csv"
        write_camera_csv(out, kp2d[c], names, frames)
        print(f"wrote {out}  ({kp2d.shape[1]} frames)")

    print(f"camera->P order {order} · joints {len(names)} · columns {1 + 3 * len(names)}")


if __name__ == "__main__":
    main()
