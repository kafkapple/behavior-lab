"""Zero-shot DeepLabCut SuperAnimal on per-camera videos built from the labelling images, run on the GPU host.

Standalone (needs only deeplabcut + pandas):
    python run_dlc.py --videos cam1.mp4 ... --model superanimal_topviewmouse --out-dir out/
    python run_dlc.py --images-glob 'frames/cam{c}_*.png' --cameras 6 --model ... --out-dir out/

The image mode feeds the PNGs directly (no video re-encoding) and writes the same CSV per camera, rows in
sorted file-name order.

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


def images_to_csv(images: list[Path], model: str, net: str, detector: str, bbox_threshold: float, out: Path) -> Path:
    """DLC image API on one camera's images -> flat CSV (row i = images[i]); no detection -> -1."""
    import deeplabcut
    import numpy as np
    import pandas as pd
    import yaml
    from deeplabcut.pose_estimation_pytorch.apis.analyze_images import superanimal_analyze_images

    cfg = Path(deeplabcut.__file__).parent / "modelzoo" / "project_configs" / f"{model}.yaml"
    names = yaml.safe_load(cfg.read_text())["bodyparts"]
    pred = superanimal_analyze_images(model, net, detector, [str(p) for p in images], max_individuals=1,
                                      out_folder=str(out.parent / f"{out.stem}_plots"), device="cuda",
                                      bbox_threshold=bbox_threshold, plot_skeleton=False)
    flat = pd.DataFrame(index=range(len(images)))
    rows, absent = [], 0
    for p in images:
        bp = np.asarray(pred[str(p)]["bodyparts"], float) if str(p) in pred else np.zeros((0,))  # no detection -> key absent
        absent += str(p) not in pred
        rows.append(bp[0] if bp.ndim == 3 and len(bp) else np.full((len(names), 3), -1.0))
    print(f"[run_dlc] {out.name}: {absent} of {len(images)} images absent from the predictions (no detection)")
    arr = np.nan_to_num(np.stack(rows), nan=-1.0)
    assert arr.shape == (len(images), len(names), 3), arr.shape
    for k, name in enumerate(names):
        flat[f"{name}_x"], flat[f"{name}_y"], flat[f"{name}_likelihood"] = arr[:, k, 0], arr[:, k, 1], arr[:, k, 2]
    flat.to_csv(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--images-glob", help="pattern with {c} for the camera number, e.g. frames/cam{c}_*.png")
    ap.add_argument("--cameras", type=int, default=0)
    ap.add_argument("--bbox-threshold", type=float, default=0.9, help="image mode; 0.9 = video API default")
    ap.add_argument("--model", required=True, choices=["superanimal_quadruped", "superanimal_topviewmouse"])
    ap.add_argument("--net", default="hrnet_w32")
    ap.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import deeplabcut

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.images_glob:
        assert args.cameras > 0, "--cameras is required with --images-glob"
        for c in range(1, args.cameras + 1):
            pat = Path(args.images_glob.format(c=c))
            images = sorted(pat.parent.glob(pat.name))
            assert images, pat
            print(images_to_csv(images, args.model, args.net, args.detector, args.bbox_threshold, out_dir / f"{args.model}_cam{c}.csv"))
        return
    assert args.videos, "--videos or --images-glob"
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
