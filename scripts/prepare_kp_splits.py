"""Prepare deterministic train/test splits for kp_benchmark v0.1.

Generates three CSV files under data/splits/:
    mammal_m1_train.csv   — 80% of MAMMAL M1 frame IDs (seed=42)
    mammal_m1_test.csv    — 20% held-out
    li_m1_external.csv    — all Li 2023 M1 GT timepoints (OOD test)

Usage
-----
    python scripts/prepare_kp_splits.py \\
        --mammal-npz <path/to/keypoints_22_3d.npz> \\
        --li-label3d <path/to/label3d_dannce.mat> \\
        --output-dir data/splits

Designed to work both locally (after scp) and on gpu03 with default paths.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow running as `python scripts/prepare_kp_splits.py` from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavior_lab.data.loaders.li2023 import Li2023Loader
from behavior_lab.data.loaders.mammal_mouse import MammalMouseLoader


SEED = 42
TRAIN_RATIO = 0.8


def make_mammal_splits(npz_path: Path, output_dir: Path) -> tuple[int, int]:
    """Deterministic 80/20 split over MAMMAL M1 frame IDs.

    Returns (n_train, n_test).
    """
    loader = MammalMouseLoader(npz_path=npz_path)
    session = loader.load()
    n_frames = session.keypoints_3d.shape[0]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_frames)
    n_train = int(n_frames * TRAIN_RATIO)
    train_ids = np.sort(perm[:n_train])
    test_ids = np.sort(perm[n_train:])

    _write_csv(output_dir / "mammal_m1_train.csv", train_ids, header=["frame_id"])
    _write_csv(output_dir / "mammal_m1_test.csv", test_ids, header=["frame_id"])
    return len(train_ids), len(test_ids)


def make_li_external(label3d_path: Path, output_dir: Path) -> int:
    """Extract Li M1 GT timepoint frame IDs as OOD test set."""
    loader = Li2023Loader(label3d_path=label3d_path)
    session = loader.load()
    frame_ids = np.sort(session.frame_ids)
    _write_csv(
        output_dir / "li_m1_external.csv",
        frame_ids,
        header=["frame_id"],
    )
    return len(frame_ids)


def _write_csv(path: Path, ids: np.ndarray, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for v in ids:
            writer.writerow([int(v)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mammal-npz", type=Path, required=True)
    ap.add_argument("--li-label3d", type=Path, required=True)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "splits",
    )
    args = ap.parse_args()

    print(f"[prepare_kp_splits] seed={SEED} train_ratio={TRAIN_RATIO}")
    n_train, n_test = make_mammal_splits(args.mammal_npz, args.output_dir)
    print(f"  mammal_m1_train.csv : {n_train} frames")
    print(f"  mammal_m1_test.csv  : {n_test} frames")
    n_li = make_li_external(args.li_label3d, args.output_dir)
    print(f"  li_m1_external.csv  : {n_li} frames (OOD)")
    print(f"[done] splits → {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
