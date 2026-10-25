"""Label-free comparison of prediction tables. Agreement between models and reprojection stability are not accuracy."""
from __future__ import annotations

import itertools
from typing import Mapping

import numpy as np
import pandas as pd

from .joints import PAIRED, PART_MAP, PARTS

CONF_MIN = 0.5


def part_points(df: pd.DataFrame, model: str, conf_min: float = CONF_MIN) -> dict[tuple[int, str, str], np.ndarray]:
    """(frame, camera, part) -> array (n, 2) of confident points of that part. Raises if a mapped name is absent."""
    mapping = PART_MAP[model]
    sub = df[df["model"] == model]
    absent = sorted(set(mapping) - set(sub["keypoint"]))
    if absent:
        raise KeyError(f"{model}: mapped keypoints missing from the table: {absent}")
    sub = sub[sub["keypoint"].isin(mapping) & sub["x_px"].notna() & (sub["conf"] >= conf_min)]
    out: dict[tuple[int, str, str], list] = {}
    for row in sub.itertuples(index=False):
        out.setdefault((row.frame, row.camera, mapping[row.keypoint]), []).append((row.x_px, row.y_px))
    return {k: np.array(v, float) for k, v in out.items()}


def _distance(a: np.ndarray, b: np.ndarray, part: str) -> float | None:
    n = PAIRED.get(part, 1)
    if len(a) != n or len(b) != n:
        return None
    if n == 1:
        return float(np.hypot(*(a[0] - b[0])))
    straight = np.hypot(*(a - b).T).sum()
    crossed = np.hypot(*(a - b[::-1]).T).sum()
    return float(min(straight, crossed) / 2)


def agreement(df: pd.DataFrame, models: list[str], conf_min: float = CONF_MIN) -> pd.DataFrame:
    """Per model pair and part: median pixel distance over (frame, camera) where both see the whole part."""
    pts = {m: part_points(df, m, conf_min) for m in models}
    rows = []
    for a, b in itertools.combinations(models, 2):
        for part in PARTS:
            d = [x for key in pts[a].keys() & pts[b].keys() if key[2] == part
                 if (x := _distance(pts[a][key], pts[b][key], part)) is not None]
            if d:
                rows.append({"a": a, "b": b, "part": part, "n": len(d), "median_px": float(np.median(d)),
                             "p90_px": float(np.quantile(d, 0.9))})
    return pd.DataFrame(rows, columns=["a", "b", "part", "n", "median_px", "p90_px"])


def reprojection(df: pd.DataFrame, model: str, cams: list[dict], conf_min: float = CONF_MIN, min_cams: int = 3) -> dict:
    """DLT over native keypoints seen by >= min_cams cameras, then pixel distance back to each camera.

    Needs the rig calibration (behavior_lab.rig.multiview.load_calib). Stability measure: a biased but consistent
    model can score well, and a keypoint on a different body spot per camera scores badly.
    """
    from behavior_lab.rig.multiview import reproj_px, triangulate

    sub = df[(df["model"] == model) & df["x_px"].notna() & (df["conf"] >= conf_min)]
    cam_index = {f"cam_{i + 1}": i for i in range(len(cams))}
    errs: list[float] = []
    seen = tri = 0
    for (_, _), g in sub.groupby(["frame", "keypoint"]):
        seen += 1
        obs = [(cam_index[r.camera], np.array([r.x_px, r.y_px])) for r in g.itertuples(index=False)]
        if len(obs) < min_cams:
            continue
        tri += 1
        X = triangulate(obs, cams)
        errs += [reproj_px(X, c, xy, cams) for c, xy in obs]
    e = np.array(errs)
    return {"model": model, "n_triangulated": tri, "n_candidates": seen, "n_residuals": int(e.size),
            "median_px": float(np.median(e)) if e.size else None,
            "p90_px": float(np.quantile(e, 0.9)) if e.size else None}


def detection_rate(df: pd.DataFrame, models: list[str], conf_min: float = CONF_MIN) -> Mapping[str, dict]:
    """Share of (frame, camera, keypoint) cells that carry a point, and a confident one."""
    out = {}
    for m in models:
        sub = df[df["model"] == m]
        out[m] = {"cells": int(len(sub)), "detected": float(sub["x_px"].notna().mean()),
                  "confident": float((sub["x_px"].notna() & (sub["conf"] >= conf_min)).mean())}
    return out
