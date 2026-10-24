"""AVATAR layout check for one frame: composite vs split cells vs keypoint overlay, one image + numbers.

Rows of the image: (a) composite with the 5 cells from `rig.avatar.cells`, (b) the split per-camera
frames placed at their origins, (c) keypoints drawn on the composite (composite coords) and on the
split frames (via `to_cell`), then one zoom per camera around its keypoints (top: composite crop,
bottom: split frame). Printed numbers: max |composite point - (cell point + origin)| (0 px expected)
and mean |split - composite crop| per camera (nonzero only from the split's CRF re-encode).

    python scripts/avatar_layout_check.py --composite rec.mp4 --split-dir <dir> --prefix rec \
        --kp rec.json.gz --calib config.toml --frame 100 --out check.jpg
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from behavior_lab.rig.avatar import COMPOSITE, ORIGIN, cells, to_cell

W_OUT = 1600  # image width; each layout row is the composite scaled to this width
ZOOM = 320  # zoom tile side, px


def frame_at(path: Path, f: int) -> np.ndarray:
    """Frame f by sequential decoding (seeking an H.264 stream can land on a neighbouring frame)."""
    cap = cv2.VideoCapture(str(path))
    for _ in range(f):
        assert cap.grab(), f"{path.name} ended before frame {f}"
    ok, img = cap.read()
    cap.release()
    assert ok, f"{path.name}: no frame {f}"
    return img


def draw_kp(img: np.ndarray, pts: np.ndarray, r: int) -> None:
    for x, y in np.round(pts).astype(int):
        cv2.circle(img, (x, y), r, (0, 0, 255), -1)
        cv2.circle(img, (x, y), r + 2, (255, 255, 255), 2)


def label(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 2.4) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 4, cv2.LINE_AA)


def check(composite: Path, split_dir: Path, prefix: str, kp: Path, calib: Path, f: int, thr: float):
    cs = cells(calib)
    comp = frame_at(composite, f)
    assert comp.shape[1::-1] == COMPOSITE, f"composite is {comp.shape[1::-1]}"
    split = {c: frame_at(split_dir / f"{prefix}_{c}.mp4", f) for c in cs}
    pc = json.load(gzip.open(kp))["keypoint"][str(f)]
    pts = {  # camera -> (n, 2) composite-normalised points with conf >= thr
        c: np.array([v[:2] for v in pc.get(str(i), {}).values() if v[2] >= thr]).reshape(-1, 2)
        for i, c in enumerate(cs)
    }
    nums = {"frame": f, "conf_thr": thr, "cams": {}}
    for i, (c, (x0, y0, w, h)) in enumerate(cs.items()):
        crop = comp[y0 : y0 + h, x0 : x0 + w]
        assert split[c].shape == crop.shape, f"{c}: split {split[c].shape} vs cell {crop.shape}"
        p_comp, p_cell = pts[c] * COMPOSITE, to_cell(pts[c], i)
        nums["cams"][c] = {
            "points": len(pts[c]),
            "max_point_diff_px": float(np.abs(p_comp - (p_cell + ORIGIN[c])).max(initial=0)),
            "mean_abs_pixel_diff": float(np.abs(split[c].astype(int) - crop.astype(int)).mean()),
            # offset (dx, dy) of the composite crop that matches the split best: (0, 0) = no shift
            "best_shift_px": min(
                ((dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)),
                key=lambda d: np.abs(
                    split[c][4:-4, 4:-4].astype(int)
                    - comp[y0 + 4 + d[1] : y0 + h - 4 + d[1], x0 + 4 + d[0] : x0 + w - 4 + d[0]].astype(int)
                ).mean(),
            ),
        }

    a = comp.copy()
    b = np.zeros_like(comp)
    for c, (x0, y0, w, h) in cs.items():
        b[y0 : y0 + h, x0 : x0 + w] = split[c]
    c_comp, c_split = comp.copy(), {c: im.copy() for c, im in split.items()}
    for i, (c, (x0, y0, w, h)) in enumerate(cs.items()):
        for im in (a, b):
            cv2.rectangle(im, (x0, y0), (x0 + w - 1, y0 + h - 1), (0, 255, 0), 6)
        label(a, f"{c} ({x0},{y0}) {w}x{h}", (x0 + 20, y0 + 80))
        label(b, f"split {c}", (x0 + 20, y0 + 80))
        draw_kp(c_comp, pts[c] * COMPOSITE, 7)
        draw_kp(c_split[c], to_cell(pts[c], i), 7)
    c_tiled = np.zeros_like(comp)
    for c, (x0, y0, w, h) in cs.items():
        c_tiled[y0 : y0 + h, x0 : x0 + w] = c_split[c]
    label(a, "(a) composite + cells", (2420, 1300))
    label(b, "(b) split frames at origins", (2420, 1300))
    label(c_comp, "(c) kp on composite", (2420, 1300))
    label(c_tiled, "(c) kp on split (to_cell)", (2420, 1300))

    zooms = []
    for i, (c, (x0, y0, w, h)) in enumerate(cs.items()):
        q = to_cell(pts[c], i)
        cx, cy = (np.median(q, 0) if len(q) else (w / 2, h / 2)).astype(int)
        half = 150
        x1, y1 = int(np.clip(cx - half, 0, w - 2 * half)), int(np.clip(cy - half, 0, h - 2 * half))
        top = c_comp[y0 + y1 : y0 + y1 + 2 * half, x0 + x1 : x0 + x1 + 2 * half]
        bot = c_split[c][y1 : y1 + 2 * half, x1 : x1 + 2 * half]
        tile = np.vstack([cv2.resize(t, (ZOOM, ZOOM)) for t in (top, bot)])
        cv2.putText(tile, f"{c} comp", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(tile, f"{c} split", (8, ZOOM + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        zooms.append(tile)

    h_row = round(COMPOSITE[1] * W_OUT / 2 / COMPOSITE[0])
    half_w = [cv2.resize(im, (W_OUT // 2, h_row), interpolation=cv2.INTER_AREA) for im in (a, b, c_comp, c_tiled)]
    out = np.vstack([np.hstack(half_w[:2]), np.hstack(half_w[2:]), np.hstack(zooms)])
    return out, nums


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("--composite", "--split-dir", "--kp", "--calib", "--out"):
        ap.add_argument(name, type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5, help="keypoint confidence floor")
    a = ap.parse_args()
    img, nums = check(a.composite, a.split_dir, a.prefix, a.kp, a.calib, a.frame, a.thr)
    cv2.imwrite(str(a.out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    a.out.with_suffix(".json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
