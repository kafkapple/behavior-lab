"""Read SLEAP `.slp` label files with h5py and check them without the images.

SUBTLE trains SLEAP from a package `.pkg.slp` whose images are embedded; the label files shipped next to the model
(`labels_gt.train.0.slp`, `labels_gt.val.0.slp`) hold coordinates only. These helpers read such files (sleap-io fails on a
val file whose video table is shorter than its frame table), then report counts, occlusion rates, points outside the
image, train/val leakage and a left/right geometry check. Occluded points keep their coordinates (`visible` False);
sleap-io `numpy()` turns them into NaN, which changes any per-side statistic.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

AVATAR_CELLS = ([1000, 1200], [1200, 1200])  # (height, width) of the AVATAR camera cells


def load_slp(path: str | Path) -> dict:
    """Skeleton nodes and edges, video table, and one point array (n_nodes, 3: x, y, visible) per frame."""
    with h5py.File(path, "r") as f:
        meta = json.loads(f["metadata"].attrs["json"])
        videos = [json.loads(x) for x in f["videos_json"][:]]
        frames, inst, pts = f["frames"][:], f["instances"][:], f["points"][:]
        pred = f["pred_points"][:] if "pred_points" in f else np.zeros(0)
    by_frame = {int(i["frame_id"]): i for i in inst[::-1]}  # first instance of each frame wins
    out, skipped = [], 0
    for fr in frames:
        vid = int(fr["video"])
        i = by_frame.get(int(fr["frame_id"]))
        if vid >= len(videos) or i is None:
            skipped += 1
            continue
        v = videos[vid]
        src = (v.get("source_video") or {}).get("filename") or v["backend"]["filename"]
        a = pts[int(i["point_id_start"]):int(i["point_id_end"])]
        out.append({"src": src.replace("\\", "/").split("/")[-1], "shape": v["backend"].get("shape"), "frame_idx": int(fr["frame_idx"]),
                    "pts": np.array([[p["x"], p["y"], float(p["visible"])] for p in a], float)})
    sk = meta["skeletons"][0]
    return {"nodes": [n["name"] for n in meta["nodes"]], "edges": [[l["source"], l["target"]] for l in sk["links"]],
            "frames": out, "skipped": skipped, "n_frames": len(frames), "has_pred": len(pred) > 0}


def load_predictions(path: str | Path) -> list[dict]:
    """Same layout for a `labels_pr.*.slp` file (predicted points)."""
    with h5py.File(path, "r") as f:
        videos = [json.loads(x) for x in f["videos_json"][:]]
        frames, inst, pts = f["frames"][:], f["instances"][:], f["pred_points"][:]
    by_frame = {int(i["frame_id"]): i for i in inst[::-1]}
    out = []
    for fr in frames:
        vid, i = int(fr["video"]), by_frame.get(int(fr["frame_id"]))
        if vid >= len(videos) or i is None:
            continue
        v = videos[vid]
        src = (v.get("source_video") or {}).get("filename") or v["backend"]["filename"]
        a = pts[int(i["point_id_start"]):int(i["point_id_end"])]
        out.append({"src": src.replace("\\", "/").split("/")[-1], "frame_idx": int(fr["frame_idx"]),
                    "pts": np.array([[p["x"], p["y"], float(p["visible"])] for p in a], float)})
    return out


def is_avatar(frame: dict) -> bool:
    s = frame["shape"]
    return bool(s) and list(s[1:3]) in [list(c) for c in AVATAR_CELLS]


def label_rates(frames: Sequence[dict]) -> dict:
    """Per node: share visible, share occluded-but-located, share without coordinates."""
    a = np.array([f["pts"][:, 2] for f in frames])
    nan = np.array([np.isnan(f["pts"][:, 0]) for f in frames])
    return {"visible": ((a == 1) & ~nan).mean(0).round(3).tolist(), "occluded": ((a == 0) & ~nan).mean(0).round(3).tolist(),
            "missing": nan.mean(0).round(3).tolist()}


def outside_image(frames: Sequence[dict]) -> dict:
    """'WxH' -> [points outside the image, points]; frames without a video shape are skipped."""
    out: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for f in frames:
        if not f["shape"]:
            continue
        h, w = f["shape"][1], f["shape"][2]
        p = f["pts"]
        ok = ~np.isnan(p[:, 0])
        bad = ok & ((p[:, 0] < 0) | (p[:, 0] > w) | (p[:, 1] < 0) | (p[:, 1] > h))
        out[f"{w}x{h}"][0] += int(bad.sum())
        out[f"{w}x{h}"][1] += int(ok.sum())
    return dict(sorted(out.items()))


def leakage(train: Sequence[dict], val: Sequence[dict], within: Sequence[int] = (0, 5, 30, 100)) -> dict:
    """Val frames that have a train frame of the same source video within k frames."""
    by_src: dict[str, list[int]] = collections.defaultdict(list)
    for f in train:
        by_src[f["src"]].append(f["frame_idx"])
    return {str(k): sum(any(abs(f["frame_idx"] - t) <= k for t in by_src[f["src"]]) for f in val) for k in within}


def val_error(val: Sequence[dict], pred: Sequence[dict]) -> np.ndarray:
    """(n matched frames, n nodes) distance in px between val labels and predictions of the same (video, frame)."""
    pm = {(p["src"], p["frame_idx"]): p for p in pred}
    rows = [np.hypot(*(f["pts"][:, :2] - pm[(f["src"], f["frame_idx"])]["pts"][:, :2]).T) for f in val if (f["src"], f["frame_idx"]) in pm]
    return np.array(rows)


def side_check(frames: Sequence[dict], nodes: Sequence[str], axis: tuple[str, str], left: str, visible_only: bool = False) -> dict:
    """Share of frames where the `left` point lies on the positive side of cross(axis, left - axis start) in image coordinates.

    On a ventral view a consistent convention gives a share near 1 (or 0). Side views are ambiguous, so use bottom views.
    """
    ix = {n: i for i, n in enumerate(nodes)}
    a, b, c = ix[axis[0]], ix[axis[1]], ix[left]
    sides = []
    for f in frames:
        p = f["pts"]
        if visible_only and not (p[[a, b, c], 2] == 1).all():
            continue
        v, w = p[b, :2] - p[a, :2], p[c, :2] - p[a, :2]
        sides.append(np.sign(v[0] * w[1] - v[1] * w[0]))
    s = np.array(sides)
    return {"n": int(len(s)), "left_positive": round(float((s == 1).mean()), 2) if len(s) else None}


def analyze(train: dict, val: dict, pred_val: Sequence[dict] | None = None) -> dict:
    """Everything the training-label dashboard shows, as plain JSON-able data."""
    nodes = train["nodes"]
    allf = train["frames"] + val["frames"]
    av = [f for f in allf if is_avatar(f)]
    res = {"nodes": nodes, "edges": train["edges"], "counts": {"train": len(train["frames"]), "val": len(val["frames"])},
           "skipped": {"train": train["skipped"], "val": val["skipped"]},
           "avatar_counts": {"train": sum(map(is_avatar, train["frames"])), "val": sum(map(is_avatar, val["frames"]))},
           "label_rates": {"avatar": label_rates(av) if av else None, "other": label_rates([f for f in allf if not is_avatar(f)])},
           "oob_by_size": outside_image(allf), "val_with_train_within": leakage(train["frames"], val["frames"]), "val_n": len(val["frames"])}
    if pred_val is not None:
        e = val_error(val["frames"], pred_val)
        ea = val_error([f for f in val["frames"] if is_avatar(f)], pred_val)
        res["val_err_px_median"] = {"all": np.nanmedian(e, 0).round(1).tolist() if len(e) else None, "avatar": np.nanmedian(ea, 0).round(1).tolist() if len(ea) else None}
        res["val_err_n"] = {"all": len(e), "avatar": len(ea)}
    legs = {"foreleg": ("tailstart1", "neck1", "forelegL1"), "hindleg": ("tailstart1", "neck1", "hindlegL1"), "ear": ("neck1", "nose1", "earL1")}
    lr = {}
    if set(sum(map(list, legs.values()), [])) <= set(nodes):
        for src in sorted({f["src"] for f in av if f["src"].endswith("_bot.mp4")}):
            fr = [f for f in av if f["src"] == src]
            lr[src] = {k: {"all": side_check(fr, nodes, (a, b), c), "visible_only": side_check(fr, nodes, (a, b), c, True)} for k, (a, b, c) in legs.items()}
    res["lr_bottom_view"] = lr
    res["avatar_frames"] = [{"split": s, "src": f["src"], "frame": f["frame_idx"], "wh": [f["shape"][2], f["shape"][1]], "pts": f["pts"].round(1).tolist()}
                            for s, d in (("train", train), ("val", val)) for f in d["frames"] if is_avatar(f)]
    return res
