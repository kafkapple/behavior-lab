"""Convert CalMS21 raw JSON → 24-dim ego-centric dyadic features.

This is the v2 preparation script. v1 (prepare_calms21_csv.py) used only the
resident mouse in pixel coords (12 dim). v2 builds the proper social
representation by calling behavior_lab.data.features.dyadic.ego_centric_dyadic.

Output:
    data/calms21_behavior_discovery/raw_csv_v2/<session>.csv  — flat 24-dim CSVs
    data/calms21_behavior_discovery/annotations/<session>.npy  — unchanged (reused)
    data/calms21_behavior_discovery/data_info_v2.csv          — summary + per-session
                                                    frac_nan_axis (degeneracy)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_lab.data.features.dyadic import ego_centric_dyadic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=REPO / "data" / "calms21"
                    / "task1_classic_classification"
                    / "calms21_task1_train.json")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "data" / "calms21_behavior_discovery" / "raw_csv_v2")
    ap.add_argument("--annot-dir", type=Path,
                    default=REPO / "data" / "calms21_behavior_discovery" / "annotations")
    ap.add_argument("--max-sessions", type=int, default=8)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.annot_dir.mkdir(parents=True, exist_ok=True)

    with open(args.src) as f:
        bundle = json.load(f)

    annot_block = bundle[next(iter(bundle))]
    sessions = list(annot_block.items())[: args.max_sessions]
    print(f"converting {len(sessions)} sessions → 24-dim ego-centric dyadic features")
    rows = []
    for key, seq in sessions:
        name = key.split("/")[-1]
        kp = np.array(seq["keypoints"], dtype=np.float32)
        feats, info = ego_centric_dyadic(kp, fps=args.fps)
        csv_path = args.out_dir / f"{name}.csv"
        pd.DataFrame(feats).to_csv(csv_path, header=False, index=False)
        np.save(args.annot_dir / f"{name}.npy",
                np.array(seq["annotations"], dtype=np.int16))
        rows.append({"session": name, "frames": feats.shape[0],
                     "dims": feats.shape[1],
                     "frac_nan_axis": info["frac_nan_axis"]})
        print(f"  {name}: {feats.shape[0]} frames × {feats.shape[1]} dims  "
              f"frac_degen={info['frac_nan_axis']:.4f}")

    info_csv = REPO / "data" / "calms21_behavior_discovery" / "data_info_v2.csv"
    pd.DataFrame(rows).to_csv(info_csv, index=False)
    print(f"\nwrote summary: {info_csv}")


if __name__ == "__main__":
    main()
