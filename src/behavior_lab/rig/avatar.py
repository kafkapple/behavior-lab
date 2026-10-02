"""AVATAR rig (5 cameras recorded as one 3600x2000 composite): layout, split, calibration check.

The layout (composite size, cell origins, calibration date) is read from configs/rig/avatar.yaml;
the cell size is each camera's calibrated `size`. Recordings of another size are rejected.

The rig has one calibration (2024-12-24). `residuals` measures how well it explains another
day's SLEAP detections: DLT over the cameras that see a keypoint with conf >= thr (min_cams or
more: with two the residual is near zero by construction), then the reprojection distance per
camera. Detection noise is included, so the number is not comparable with the calibration's own
error. Measured 261002: recordings from 2023-08-22 on give a median of 8-16 px, recordings up
to 2023-05-23 give 69-106 px.

    python -m behavior_lab.rig.avatar residual --calib config.toml \
        --kp TAG=YYYYMMDD=<json.gz> ... --out drift.json
    python -m behavior_lab.rig.avatar plot --json drift.json --out drift.png
    python -m behavior_lab.rig.avatar split --composite rec.mp4 --calib config.toml \
        --out <dir> --prefix <name>
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import tomllib
import yaml

from .multiview import load_calib, reproj_px, triangulate

# ponytail: configs/ sits at the repo root, so this needs a source checkout (editable install).
# Ship it as package data if the package is ever installed without one.
RIG_YAML = Path(__file__).resolve().parents[3] / "configs" / "rig" / "avatar.yaml"
_RIG = yaml.safe_load(RIG_YAML.read_text())
COMPOSITE: tuple[int, int] = tuple(_RIG["composite"])
ORIGIN: dict[str, tuple[int, int]] = {c: tuple(o) for c, o in _RIG["origin"].items()}
CALIB_DATE: dt.date = _RIG["calib_date"]


def cells(calib: str | Path) -> dict[str, tuple[int, int, int, int]]:
    """camera -> (x0, y0, w, h) in the composite; asserts the cells fit and do not overlap."""
    cfg = tomllib.loads(Path(calib).expanduser().read_text())
    cs = {c: (*ORIGIN[c], *cfg[c]["size"]) for c in ORIGIN}
    boxes = list(cs.values())
    for i, (x, y, w, h) in enumerate(boxes):
        assert x + w <= COMPOSITE[0] and y + h <= COMPOSITE[1], (
            f"cell {x},{y} {w}x{h} outside the composite"
        )
        for x2, y2, w2, h2 in boxes[i + 1 :]:
            assert x + w <= x2 or x2 + w2 <= x or y + h <= y2 or y2 + h2 <= y, (
                "camera cells overlap"
            )
    return cs


def _probe(p: Path) -> tuple[int, int, int]:
    out = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                "csv=p=0",
                str(p),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(",")
    )
    return int(out[0]), int(out[1]), int(out[2])


def split_composite(
    composite: Path, calib: Path, out: Path, prefix: str, crf: int = 18
) -> list[Path]:
    """One mp4 per camera at its calibrated size; asserts size and frame count of each."""
    W, H, n = _probe(composite)
    assert (W, H) == COMPOSITE, (
        f"{composite.name} is {W}x{H}, the layout is defined for {COMPOSITE[0]}x{COMPOSITE[1]} only"
    )
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for c, (x, y, w, h) in cells(calib).items():
        dst = out / f"{prefix}_{c}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(composite),
                "-vf",
                f"crop={w}:{h}:{x}:{y}",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(dst),
            ],
            check=True,
        )
        assert _probe(dst) == (w, h, n), f"{c}: wrote {_probe(dst)}, expected {(w, h, n)}"
        written.append(dst)
    return written


def residuals(
    kp: dict, cams: list[dict], thr: float = 0.5, min_cams: int = 3, n_frames: int = 100
) -> dict:
    """kp = SUBTLE/SLEAP json: keypoint[frame][cam][k] = [x, y, conf], composite-normalised."""
    origin = np.array(list(ORIGIN.values()), float)
    frames = sorted(int(f) for f in kp["keypoint"])[:n_frames]
    per_cam: list[list[float]] = [[] for _ in cams]
    seen = tri = 0
    for f in frames:
        pc = kp["keypoint"][str(f)]
        for k in range(len(kp["node"]["id"])):
            obs = [
                (int(c), np.array(v[str(k)][:2]) * COMPOSITE - origin[int(c)])
                for c, v in pc.items()
                if str(k) in v and v[str(k)][2] >= thr
            ]
            seen += 1
            if len(obs) < min_cams:
                continue
            tri += 1
            X = triangulate(obs, cams)
            for c, xy in obs:
                per_cam[c].append(reproj_px(X, c, xy, cams))
    allv = np.array([e for v in per_cam for e in v])
    assert len(allv), f"no keypoint seen by {min_cams}+ cameras"
    return {
        "frames": len(frames),
        "triangulated_frac": tri / seen,
        "n_residuals": int(len(allv)),
        "median_px": float(np.median(allv)),
        "p90_px": float(np.percentile(allv, 90)),
        "per_cam_median_px": [float(np.median(v)) if v else None for v in per_cam],
    }


def residual_by_date(calib: Path, specs: list[str], **kw) -> dict:
    """specs = TAG=YYYYMMDD=<json.gz>; rows sorted by date with days from the calibration date."""
    cams, rows = load_calib(calib), []
    for spec in specs:
        tag, date, path = spec.split("=", 2)
        day = dt.datetime.strptime(date, "%Y%m%d").date()
        r = residuals(json.load(gzip.open(Path(path).expanduser())), cams, **kw)
        rows.append(
            {"tag": tag, "date": day.isoformat(), "days_from_calib": (day - CALIB_DATE).days, **r}
        )
    return {
        "calib_date": CALIB_DATE.isoformat(),
        **{k: kw[k] for k in ("thr", "min_cams")},
        "frames": kw["n_frames"],
        "rows": sorted(rows, key=lambda r: (r["date"], r["tag"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("residual")
    r.add_argument("--calib", type=Path, required=True)
    r.add_argument("--kp", action="append", required=True, help="TAG=YYYYMMDD=<json.gz>")
    r.add_argument("--thr", type=float, default=0.5)
    r.add_argument("--min-cams", type=int, default=3)
    r.add_argument("--frames", type=int, default=100)
    r.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("plot")
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("split")
    for name in ("--composite", "--calib", "--out"):
        s.add_argument(name, type=Path, required=True)
    s.add_argument("--prefix", required=True)
    a = ap.parse_args()
    if a.cmd == "residual":
        d = residual_by_date(a.calib, a.kp, thr=a.thr, min_cams=a.min_cams, n_frames=a.frames)
        a.out.expanduser().write_text(json.dumps(d, indent=1))
        for row in d["rows"]:
            print(
                f"{row['date']} {row['days_from_calib']:+5d} d  {row['tag']:14s} "
                f"tri {row['triangulated_frac']:.2f}  "
                f"median {row['median_px']:5.1f}  p90 {row['p90_px']:5.1f}"
            )
    elif a.cmd == "plot":
        from ..visualization.rig import plot_residual_by_date

        print(
            json.dumps(
                plot_residual_by_date(
                    json.loads(a.json.expanduser().read_text()), a.out.expanduser()
                )
            )
        )
    else:
        for dst in split_composite(
            a.composite.expanduser(), a.calib.expanduser(), a.out.expanduser(), a.prefix
        ):
            print(dst)


if __name__ == "__main__":
    main()
