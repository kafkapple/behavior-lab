"""Assemble every available prediction on the labelling images into one long table and score it label-free.

    python scripts/kp_compare.py assemble --avatar-dir <260929_AVATAR_SUBTLE_overlay_compare> --out preds/all.csv
    python scripts/kp_compare.py boxes --table preds/all.csv --model sleap_1423 --images-dir <gt_label dir> --out boxes.csv
    python scripts/kp_compare.py score --table preds/all.csv --calib config.toml --out results.json
    python scripts/kp_compare.py report --avatar-dir <AVATAR dir> --out comparison.html   (needs all.csv and results.json in kp_gt_261003/)

Inputs of `assemble` (AVATAR dir): selection.csv frames in `gt_label_261002/`, `outputs/<clip>_<model>.json.gz` (SUBTLE
schema), `kp_gt_261003/<model>/<prefix>_cam{1-5}.csv` (DLC video CSV, one row per selection frame in order, so the video must be
built from the selection images and nothing else), `kp_gt_261003/extra/*.csv` (long tables from run_vitpose.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_lab.pose.predictors import compare, tables
from behavior_lab.pose.predictors.joints import PART_MAP
from behavior_lab.pose.predictors.run_vitpose import boxes_from_table

CLIP = "far_20231205"
PRED_DIR = "kp_gt_261003"
SUBTLE_MODELS = ("sleap_1423", "sleap_tailless_1501", "yolo_avatar3d_train", "yolo_avatar3d_balbc", "yolo_khu_527", "rtdetr")
DLC_MODELS = {"superanimal_quadruped": "sa", "superanimal_topviewmouse": "tv"}


def assemble(avatar_dir: Path, out: Path) -> pd.DataFrame:
    gt = avatar_dir / "gt_label_261002"
    frames = pd.read_csv(gt / "selection.csv")["frame"].tolist()
    parts = [tables.from_subtle_json(avatar_dir / "outputs" / f"{CLIP}_{m}.json.gz", m, frames) for m in SUBTLE_MODELS]
    for model, prefix in DLC_MODELS.items():
        files = [avatar_dir / PRED_DIR / model / f"{prefix}_cam{c}.csv" for c in range(1, 6)]
        if all(f.exists() for f in files):
            parts += [tables.from_dlc_video_csv(f, model, f"cam_{c}", frames) for c, f in enumerate(files, start=1)]
    extra_dir = avatar_dir / PRED_DIR / "extra"
    for extra in sorted(extra_dir.glob("*.csv")) if extra_dir.exists() else []:
        parts.append(tables.read_table(extra))
    df = pd.concat(parts, ignore_index=True)
    tables.write_table(df, out)
    return df


def image_boxes(df: pd.DataFrame, model: str, images_dir: Path, pad: float) -> pd.DataFrame:
    """boxes_from_table + the image path of each (frame, camera): images are `f{frame:03d}_cam{n}.jpg`."""
    box = boxes_from_table(df, model, pad)
    box["image"] = [f"images/f{f:03d}_cam{c.split('_')[1]}.jpg" for f, c in zip(box["frame"], box["camera"])]
    missing = [i for i in box["image"] if not (images_dir / i).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing under {images_dir}, e.g. {missing[0]}")
    return box


def score(df: pd.DataFrame, calib: Path) -> dict:
    from behavior_lab.rig.multiview import load_calib

    models = [m for m in df["model"].unique() if m in PART_MAP]
    cams = load_calib(calib)
    return {"models": models, "detection": compare.detection_rate(df, models),
            "agreement": compare.agreement(df, models).to_dict("records"),
            "reprojection": [compare.reprojection(df, m, cams) for m in models]}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--avatar-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("boxes")
    b.add_argument("--table", type=Path, required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--images-dir", type=Path, required=True)
    b.add_argument("--pad", type=float, default=0.25)
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--table", type=Path, required=True)
    s.add_argument("--calib", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--avatar-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "report":
        from behavior_lab.pose.predictors.report import build

        print(build(args.avatar_dir, args.out))
    elif args.cmd == "assemble":
        df = assemble(args.avatar_dir, args.out)
        print(args.out, len(df), "rows;", sorted(df["model"].unique()))
    elif args.cmd == "boxes":
        box = image_boxes(tables.read_table(args.table), args.model, args.images_dir, args.pad)
        box.to_csv(args.out, index=False)
        print(args.out, len(box), "boxes")
    else:
        res = score(tables.read_table(args.table), args.calib)
        args.out.write_text(json.dumps(res, indent=1))
        print(args.out)


if __name__ == "__main__":
    main()
