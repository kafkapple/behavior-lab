"""Keypoint label table -> training-label formats.

One tidy table (one row per image x keypoint) is the single source of truth for hand labels.
Columns follow the AVATAR GT template (`frame,camera,image,image_w,image_h,keypoint,x_px,y_px,
visible,note`). `visible`: 1 = seen, 0 = occluded but located, blank = not labelled (no coordinates).

Exporters write what the keypoint trainers read:
  - DeepLabCut / Lightning Pose: `CollectedData_<scorer>.csv` (3 header rows, one CSV per camera)
  - COCO keypoints json (visibility 2 / 1 / 0), the input of SuperAnimal fine-tuning, mmpose, YOLO-pose
  - SLEAP `.slp` through the optional `sleap-io` extra
DANNCE `label3d` needs 3D points and calibration, so it is not produced here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

COLUMNS = ("frame", "camera", "image", "image_w", "image_h", "keypoint", "x_px", "y_px", "visible", "note")


def load_label_table(source: str | Path | pd.DataFrame, keypoints: Sequence[str] | None = None) -> pd.DataFrame:
    """Read and validate a label table; `keypoints` fixes the order (default: first-seen order)."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    missing = [c for c in COLUMNS[:-1] if c not in df.columns]
    if missing:
        raise ValueError(f"label table lacks columns {missing}")
    for column in ("x_px", "y_px", "visible", "image_w", "image_h"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    bad_visible = df["visible"].notna() & ~df["visible"].isin([0, 1])
    if bad_visible.any():
        raise ValueError(f"visible must be 0, 1 or blank; rows {df.index[bad_visible].tolist()[:5]}")
    unlabeled_with_xy = df["visible"].isna() & (df["x_px"].notna() | df["y_px"].notna())
    labeled_without_xy = df["visible"].notna() & (df["x_px"].isna() | df["y_px"].isna())
    if unlabeled_with_xy.any() or labeled_without_xy.any():
        rows = df.index[unlabeled_with_xy | labeled_without_xy].tolist()[:5]
        raise ValueError(f"coordinates and visible disagree; rows {rows}")
    outside = df["visible"].notna() & ((df["x_px"] < 0) | (df["x_px"] > df["image_w"]) | (df["y_px"] < 0) | (df["y_px"] > df["image_h"]))
    if outside.any():
        raise ValueError(f"points outside their image; rows {df.index[outside].tolist()[:5]}")
    order = list(keypoints) if keypoints is not None else list(dict.fromkeys(df["keypoint"]))
    unknown = sorted(set(df["keypoint"]) - set(order))
    if unknown:
        raise ValueError(f"keypoints not in the given order: {unknown}")
    df["keypoint"] = pd.Categorical(df["keypoint"], categories=order, ordered=True)
    return df.sort_values(["camera", "frame", "keypoint"], kind="stable").reset_index(drop=True)


def labeled_fraction(df: pd.DataFrame) -> float:
    """Share of rows that carry a label (visible 0 or 1)."""
    return float(df["visible"].notna().mean())


def to_dlc_csv(df: pd.DataFrame, out_dir: str | Path, scorer: str = "labeler") -> list[Path]:
    """DLC / Lightning Pose `CollectedData_<scorer>.csv`, one file per camera, image path as the index."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keypoints = list(df["keypoint"].cat.categories)
    paths = []
    for camera, part in df.groupby("camera", sort=True, observed=True):
        wide = {}
        for image, group in part.groupby("image", sort=False):
            by_name = group.set_index("keypoint")
            row = {}
            for name in keypoints:
                row[(scorer, name, "x")] = by_name.at[name, "x_px"] if name in by_name.index else np.nan
                row[(scorer, name, "y")] = by_name.at[name, "y_px"] if name in by_name.index else np.nan
            wide[image] = row
        table = pd.DataFrame.from_dict(wide, orient="index")
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=["scorer", "bodyparts", "coords"])
        path = out_dir / f"CollectedData_{scorer}_{camera}.csv"
        table.to_csv(path)
        paths.append(path)
    return paths


def to_coco(df: pd.DataFrame, edges: Sequence[tuple[str, str]] = (), category: str = "animal") -> dict:
    """COCO keypoints: one annotation per image. Visibility 2 = seen, 1 = occluded but located, 0 = unlabelled."""
    keypoints = list(df["keypoint"].cat.categories)
    index = {name: i + 1 for i, name in enumerate(keypoints)}
    images, annotations = [], []
    for image_id, (image, group) in enumerate(df.groupby("image", sort=True), start=1):
        width, height = int(group["image_w"].iloc[0]), int(group["image_h"].iloc[0])
        images.append({"id": image_id, "file_name": image, "width": width, "height": height})
        by_name = group.set_index("keypoint")
        flat, labeled = [], []
        for name in keypoints:
            visible = by_name.at[name, "visible"] if name in by_name.index else np.nan
            if np.isnan(visible):
                flat += [0.0, 0.0, 0]
            else:
                x, y = float(by_name.at[name, "x_px"]), float(by_name.at[name, "y_px"])
                flat += [x, y, 2 if visible == 1 else 1]
                labeled.append((x, y))
        if labeled:
            xs, ys = zip(*labeled)
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            pad_x, pad_y = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            x0, y0 = max(0.0, x0 - pad_x), max(0.0, y0 - pad_y)
            x1, y1 = min(float(width), x1 + pad_x), min(float(height), y1 + pad_y)
        else:
            x0 = y0 = x1 = y1 = 0.0
        annotations.append({"id": image_id, "image_id": image_id, "category_id": 1, "keypoints": flat,
                            "num_keypoints": len(labeled), "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
    skeleton = [[index[a], index[b]] for a, b in edges]
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": category, "keypoints": keypoints, "skeleton": skeleton}]}


def write_coco(df: pd.DataFrame, path: str | Path, edges: Sequence[tuple[str, str]] = (), category: str = "animal") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_coco(df, edges, category)))
    return path


def to_sleap(df: pd.DataFrame, images_root: str | Path, out_path: str | Path, edges: Sequence[tuple[str, str]] = ()) -> Path:
    """SLEAP `.slp` (needs `pip install sleap-io`): one image-sequence video per camera, one instance per image."""
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise ImportError("to_sleap needs the 'sleap' extra: pip install behavior-lab[sleap]") from exc
    keypoints = list(df["keypoint"].cat.categories)
    skeleton = sio.Skeleton(nodes=keypoints, edges=[(a, b) for a, b in edges])
    frames = []
    for _, part in df.groupby("camera", sort=True, observed=True):
        images = list(dict.fromkeys(part.sort_values("frame", kind="stable")["image"]))
        video = sio.Video.from_filename([str(Path(images_root) / image) for image in images])
        for frame_idx, image in enumerate(images):
            group = part[part["image"] == image].set_index("keypoint")
            points = np.full((len(keypoints), 2), np.nan)
            for i, name in enumerate(keypoints):
                if name in group.index and not np.isnan(group.at[name, "visible"]):
                    points[i] = (group.at[name, "x_px"], group.at[name, "y_px"])
            frames.append(sio.LabeledFrame(video=video, frame_idx=frame_idx,
                                           instances=[sio.Instance.from_numpy(points, skeleton=skeleton)]))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(sio.Labels(labeled_frames=frames), str(out_path))
    return out_path
