"""kp_benchmark v0.1 orchestrator — DLC ResNet50 vs MobileNet-SuperAnimal.

Stage 0 (this script, v0.1 scope):
    Given prediction npz files (one per predictor) + GT, compute
    root-relative MPJPE + bootstrap CI on (a) MAMMAL M1 held-out test,
    (b) Li M1 external. Write results CSV + markdown report.

Stages 1-2 (deferred to follow-up commits):
    01_train_dlc_resnet50.sh   — DLC train on mammal_m1_train.csv frames
    02_train_dlc_superanimal.sh — DLC train SuperAnimal fine-tune
    (these produce the prediction npz files)

Usage
-----
    python scripts/benchmark_kp_dlc.py \\
        --gt-npz data/splits/li_m1_gt.npz \\
        --pred-resnet50 outputs/kp_benchmark/predictions/dlc_resnet50.npz \\
        --pred-superanimal outputs/kp_benchmark/predictions/dlc_superanimal.npz \\
        --root-idx 0 \\
        --output-csv outputs/kp_benchmark/results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavior_lab.evaluation.mpjpe import bootstrap_ci, root_relative_mpjpe


def evaluate_predictor(
    pred: np.ndarray,
    gt: np.ndarray,
    root_idx: int,
    n_bootstrap: int = 10000,
) -> dict:
    per_frame = root_relative_mpjpe(pred, gt, root_idx=root_idx)
    mean, lo, hi = bootstrap_ci(per_frame, n_bootstrap=n_bootstrap)
    return {
        "n_frames": int(per_frame.size),
        "mpjpe_mean": mean,
        "mpjpe_ci_lo": lo,
        "mpjpe_ci_hi": hi,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-npz", type=Path, required=True,
                    help="ground truth (N, 22, 3) npz, key='keypoints_3d'")
    ap.add_argument("--pred-resnet50", type=Path, required=True)
    ap.add_argument("--pred-superanimal", type=Path, required=True)
    ap.add_argument("--root-idx", type=int, default=0)
    ap.add_argument("--output-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "kp_benchmark" / "results.csv")
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    args = ap.parse_args()

    gt = _load_npz(args.gt_npz)
    results: list[dict] = []
    for tag, p in [("dlc_resnet50", args.pred_resnet50),
                   ("dlc_superanimal", args.pred_superanimal)]:
        pred = _load_npz(p)
        if pred.shape != gt.shape:
            raise ValueError(
                f"{tag} pred shape {pred.shape} != gt {gt.shape}"
            )
        row = {"predictor": tag, **evaluate_predictor(
            pred, gt, args.root_idx, args.bootstrap_n
        )}
        results.append(row)
        print(f"{tag:>20s} | N={row['n_frames']:>4d} | "
              f"MPJPE={row['mpjpe_mean']:7.3f} "
              f"[{row['mpjpe_ci_lo']:7.3f}, {row['mpjpe_ci_hi']:7.3f}]")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[done] results → {args.output_csv.resolve()}")
    return 0


def _load_npz(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    data = np.load(path)
    key = "keypoints_3d" if "keypoints_3d" in data else list(data.keys())[0]
    arr = data[key]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected (N, K, 3) in {path}, got {arr.shape}")
    return arr.astype(np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
