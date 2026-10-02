"""Shared helpers of the grid pages (report, gallery, player): loading, tables, colors."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

# Slices that are the same frames under different pose post-processing: labels are comparable
# frame by frame across the slices of one group.
SAME_FRAMES_PREFIXES = ("avatar_",)
COLS = ["dataset", "method", "n_frames", "n_clusters", "median_bout_sec", "noise_frac",
        "n_repeats", "repeat_ari_mean", "repeat_ari_min", "elapsed_sec"]
# Flags (own loose rule, no literature threshold): a row that trips one is described, not ranked.
MIN_CLUSTERS, MAX_NOISE = 3, 0.3
GALLERY_CLUSTERS, GALLERY_MAX_SEC, GALLERY_PAD_SEC = 6, 2.0, 0.5
GALLERY_GIF_FPS = 8  # frames are subsampled to about this rate: 3 slices come to about 9 MB
SKELETONS = {"subtle_": "subtle_mouse", "shank3ko_": "shank3ko"}


def _flags(r: dict, slice_sec: float) -> list[str]:
    """Reasons why a cell's segmentation is degenerate; empty list = eligible for the highlight."""
    out = []
    if (r.get("n_clusters") or 0) < MIN_CLUSTERS:
        out.append(f"fewer than {MIN_CLUSTERS} clusters")
    step = slice_sec / r["n_frames"]  # seconds per label
    if r.get("median_bout_sec") is not None and r["median_bout_sec"] <= step * 1.01:
        out.append("median bout = 1 label step")
    if (r.get("noise_frac") or 0) > MAX_NOISE:
        out.append(f"noise > {MAX_NOISE:.0%}")
    return out


def _method_color(methods: list[str]) -> dict[str, str]:
    return {m: f"var(--c{1 + i % 5})" for i, m in enumerate(methods)}


def _fmt(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.2f}" if isinstance(v, float) else html.escape(str(v))


def _table(header: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _img(fig) -> str:
    import matplotlib.pyplot as plt

    from .html_report import fig_to_base64

    uri = fig_to_base64(fig, dpi=90)  # already a full data URI
    assert uri.startswith("data:image/png;base64,")
    tag = f'<img src="{uri}" alt="">'
    plt.close(fig)
    return tag


def _stem(method: str) -> str:  # same rule as the batch script's label file names
    return method.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _labels(batch_dir: Path, dataset: str) -> dict[str, np.ndarray]:
    """Label files of cells that have a finished row (a running cell leaves a file but no row)."""
    rows = json.loads((batch_dir / "batch_results.json").read_text())
    done = {_stem(r["method"]) for r in rows if r["dataset"] == dataset and r["status"] == "ok"}
    files = sorted((batch_dir / "arrays" / dataset).glob("*_labels.npy"))
    return {p.stem.removesuffix("_labels"): np.load(p) for p in files
            if p.stem.removesuffix("_labels") in done}


def _pairs(agr: dict) -> list[list[object]]:
    n = agr["names"]
    return [[n[i], n[j], float(agr["ari"][i, j]), float(agr["null_ari"][i, j]),
             float(agr["ami"][i, j]), float(agr["homogeneity"][i, j]),
             float(agr["homogeneity"][j, i])]
            for i in range(len(n)) for j in range(i + 1, len(n))]


PAIR_HEADER = ["A", "B", "ARI", "ARI, shifted null", "AMI", "B inside A", "A inside B"]


def _subtle_map(batch_dir: Path, dataset: str):
    """``(tag, arrays)`` of one SUBTLE run with matching embedding and cluster levels, or None.

    Prefers the first seed of the table's runs; falls back to the extra map-only run."""
    files = sorted((batch_dir / "arrays" / dataset).glob("subtle_map_seed*.npz")) \
        or sorted((batch_dir / "arrays" / dataset).glob("subtle_map_extra.npz"))
    if not files:
        return None
    return files[0].stem.removeprefix("subtle_map_"), dict(np.load(files[0]))


def _skeleton(name: str, node_names: list[str] | None, n_joints: int):
    from ..core.skeleton import SkeletonDefinition, get_skeleton

    for prefix, key in SKELETONS.items():
        if name.startswith(prefix):
            return get_skeleton(key), "bones from the skeleton registry"
    names = node_names or [f"kp{i}" for i in range(n_joints)]
    return (SkeletonDefinition(name=name, num_joints=n_joints, joint_names=list(names),
                               joint_parents=[-1] * n_joints, edges=[]),
            "points only: this layout has no bone list in behavior-lab")


FAMILIES = {"subtle": "SUBTLE mouse recordings", "shank3ko": "Shank3KO recordings",
            "avatar": "AVATAR pose variants"}


def _family(name: str) -> str:
    return name.split("_")[0]


def _keypoints(batch_dir: Path, dataset: str) -> np.ndarray | None:
    p = batch_dir / "arrays" / dataset / "keypoints.npy"
    return np.load(p) if p.exists() else None


def _details(summary: str, body: str, *, open_: bool = False) -> str:
    return (f'<details{" open" if open_ else ""}><summary><b>{html.escape(summary)}</b>'
            f"</summary>{body}</details>")


def _write(out: Path, title: str, parts: list[str]) -> Path:
    out.write_text("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>"
                   f"{html.escape(title)}</title></head>\n<body>\n" + "\n".join(parts)
                   + "\n</body></html>\n", encoding="utf-8")
    return out
