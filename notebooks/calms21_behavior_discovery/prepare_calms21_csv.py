"""Convert CalMS21 task1 raw JSON to SUBTLE-compatible flattened CSV.

CalMS21 raw JSON shape per session: (T, 2_mice, 2_xy, 7_keypoints)
SUBTLE expected input: (T, K*D) flat float CSV (no header, comma-separated)

This converter:
- Loads continuous keypoint trajectories from calms21_task1_{split}.json
- Selects ONE mouse (resident, index 0) for single-subject demo
  matching the original ACTNOVA notebook's per-animal flow
- Transposes (2_xy, 7_kp) -> (7_kp, 2_xy) so column order is x0,y0,x1,y1,...,x6,y6
- Drops the tail_base keypoint (index 6) BEFORE writing — i.e. fixed at source.
  Demonstrates an alternative pattern to the notebook's
  `columns_to_remove` parameter (post-hoc index drop).
- Also emits annotation arrays for later evaluation against SUBTLE clusters.

CalMS21 7-keypoint skeleton:
    0: nose, 1: left_ear, 2: right_ear, 3: neck,
    4: left_hip, 5: right_hip, 6: tail_base   <- excluded
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_NAMES_FULL = ["nose", "left_ear", "right_ear", "neck",
                    "left_hip", "right_hip", "tail_base"]
TAIL_INDEX = 6  # tail_base; matches CALMS21_MOUSE_SKELETON in core.skeleton


def convert_session(kp_TMDK: np.ndarray, *, mouse: int = 0,
                    drop_joints: tuple[int, ...] = (TAIL_INDEX,)) -> np.ndarray:
    """Convert one session's keypoints to SUBTLE flat layout.

    Args:
        kp_TMDK: shape (T, 2_mice, 2_xy, 7_kp) as stored in calms21 JSON
        mouse: 0 = resident (black), 1 = intruder (white)
        drop_joints: keypoint indices to remove (e.g. tail)

    Returns:
        flat: shape (T, (7 - len(drop_joints)) * 2)
              column order: x0,y0,x1,y1,...  (joint-major, then xy)
    """
    arr = np.asarray(kp_TMDK, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1:3] != (2, 2) or arr.shape[3] != 7:
        raise ValueError(f"unexpected CalMS21 keypoint shape {arr.shape}; "
                         "expected (T, 2_mice, 2_xy, 7_kp)")

    one_mouse = arr[:, mouse, :, :]              # (T, 2_xy, 7_kp)
    one_mouse = one_mouse.transpose(0, 2, 1)     # (T, 7_kp, 2_xy)

    keep = [j for j in range(7) if j not in set(drop_joints)]
    one_mouse = one_mouse[:, keep, :]            # (T, K_keep, 2)

    T, K, D = one_mouse.shape
    return one_mouse.reshape(T, K * D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("data/calms21/task1_classic_classification/"
                                 "calms21_task1_train.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/calms21_behavior_discovery/raw_csv"))
    ap.add_argument("--annot-dir", type=Path,
                    default=Path("data/calms21_behavior_discovery/annotations"))
    ap.add_argument("--mouse", type=int, default=0,
                    help="0=resident, 1=intruder")
    ap.add_argument("--max-sessions", type=int, default=8,
                    help="cap number of sessions for quick demo")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.annot_dir.mkdir(parents=True, exist_ok=True)

    with open(args.src) as f:
        bundle = json.load(f)

    annot_block = bundle[next(iter(bundle))]
    sessions = list(annot_block.items())[: args.max_sessions]
    print(f"converting {len(sessions)} sessions from {args.src.name}")

    meta_rows = []
    for key, seq in sessions:
        name = key.split("/")[-1]
        kp = np.array(seq["keypoints"], dtype=np.float32)
        flat = convert_session(kp, mouse=args.mouse)

        csv_path = args.out_dir / f"{name}.csv"
        pd.DataFrame(flat).to_csv(csv_path, header=False, index=False)

        annot_path = args.annot_dir / f"{name}.npy"
        np.save(annot_path, np.array(seq["annotations"], dtype=np.int16))

        meta_rows.append({
            "session": name, "frames": flat.shape[0],
            "dims": flat.shape[1], "csv": str(csv_path),
        })
        print(f"  {name}: {flat.shape[0]} frames -> {flat.shape[1]} cols")

    info_path = args.out_dir.parent / "data_info.csv"
    pd.DataFrame(meta_rows).to_csv(info_path, index=False)
    print(f"wrote summary: {info_path}")


if __name__ == "__main__":
    main()
