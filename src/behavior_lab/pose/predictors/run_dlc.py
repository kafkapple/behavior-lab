"""Zero-shot DeepLabCut SuperAnimal on per-camera videos built from the labelling images, run on the GPU host.

Standalone (needs only deeplabcut + pandas):
    python run_dlc.py --videos cam1.mp4 ... --model superanimal_topviewmouse --out-dir out/

Writes `<model>_<video stem>.csv` per video: header `,<kp>_x,<kp>_y,<kp>_likelihood,...`, row i = frame i,
x = y = -1 for a missing detection (the layout `tables.from_dlc_video_csv` reads).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def h5_to_csv(h5: Path, out: Path) -> Path:
    """DLC prediction h5 (one animal) -> flat CSV with -1 for missing points."""
    import pandas as pd

    df = pd.read_hdf(h5)
    if "individuals" in df.columns.names:
        df = df.xs(df.columns.get_level_values("individuals")[0], axis=1, level="individuals")
    df.columns = df.columns.droplevel("scorer") if "scorer" in df.columns.names else df.columns
    names = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    flat = pd.DataFrame(index=range(len(df)))
    for name in names:
        part = df[name]
        flat[f"{name}_x"], flat[f"{name}_y"], flat[f"{name}_likelihood"] = (part[c].to_numpy() for c in ("x", "y", "likelihood"))
    flat = flat.fillna(-1.0)
    flat.to_csv(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--model", required=True, choices=["superanimal_quadruped", "superanimal_topviewmouse"])
    ap.add_argument("--net", default="hrnet_w32")
    ap.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import deeplabcut

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deeplabcut.video_inference_superanimal(
        args.videos, args.model, model_name=args.net, detector_name=args.detector, device="cuda",
        dest_folder=str(out_dir), max_individuals=1, create_labeled_video=False, plot_trajectories=False,
        batch_size=8, detector_batch_size=8)
    for video in args.videos:
        stem = Path(video).stem
        h5 = next(out_dir.glob(f"{stem}_{args.model}_*.h5"))
        print(h5_to_csv(h5, out_dir / f"{args.model}_{stem}.csv"))


if __name__ == "__main__":
    main()
