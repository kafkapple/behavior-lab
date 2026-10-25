"""Long prediction table. Coordinates are pixels of the camera cell (the image a human labels), NaN = missing."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from behavior_lab.rig.avatar import COMPOSITE, ORIGIN

COLUMNS = ("model", "frame", "camera", "keypoint", "x_px", "y_px", "conf")


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(COLUMNS))


def from_subtle_json(path: str | Path, model: str, frames: Sequence[int]) -> pd.DataFrame:
    """SUBTLE/SLEAP json: keypoint[frame][cam index][k] = [x, y, conf], composite-normalised, zeros = missing."""
    p = Path(path)
    d = json.load(gzip.open(p) if p.suffix == ".gz" else open(p))
    names = d["node"]["id"]
    cams = list(ORIGIN)
    out = []
    for f in frames:
        per_cam = d["keypoint"].get(str(f), {})
        for c, name_cam in enumerate(cams):
            kps = per_cam.get(str(c), {})
            for k, name in enumerate(names):
                x, y, conf = kps.get(str(k), [0.0, 0.0, 0.0])
                if x == 0.0 and y == 0.0:
                    out.append((model, f, name_cam, name, np.nan, np.nan, np.nan))
                else:
                    ox, oy = ORIGIN[name_cam]
                    out.append((model, f, name_cam, name, x * COMPOSITE[0] - ox, y * COMPOSITE[1] - oy, conf))
    return _frame(out)


def from_dlc_video_csv(path: str | Path, model: str, camera: str, frames: Sequence[int]) -> pd.DataFrame:
    """One-camera CSV written from a DLC video inference (row i = frames[i]); x = y = -1 means not detected."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    if len(rows) != len(frames):
        raise ValueError(f"{path}: {len(rows)} rows for {len(frames)} frames")
    names = [h[:-2] for h in header if h.endswith("_x")]
    col = {h: i for i, h in enumerate(header)}
    out = []
    for f, row in zip(frames, rows):
        for name in names:
            x, y, conf = (float(row[col[f"{name}_{s}"]]) for s in ("x", "y", "likelihood"))
            missing = x == -1.0 and y == -1.0
            out.append((model, f, camera, name, np.nan if missing else x, np.nan if missing else y, np.nan if missing else conf))
    return _frame(out)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    return df[list(COLUMNS)]
