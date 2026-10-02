"""AVATAR rig (5 cameras recorded as one 3600x2000 composite): layout, split, calibration check.

Composite layout (x0, y0 = cell origin; the cell size is the camera's calibrated `size`):
    cam_1 (0, 0) 1200x1000 | cam_2 (1200, 0) 1200x1000 | cam_3 (2400, 0) 1200x1200 (bottom)
    cam_4 (0, 1000) 1200x1000 | cam_5 (1200, 1000) 1200x1000 | (2400, 1200)-(3600, 2000) UI
Recordings of other sizes (3592x2000, 3840x2160, 3592x2028) do not follow this layout and are
rejected.

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

from .multiview import load_calib, reproj_px, triangulate

COMPOSITE = (3600, 2000)
ORIGIN = {
    "cam_1": (0, 0),
    "cam_2": (1200, 0),
    "cam_3": (2400, 0),
    "cam_4": (0, 1000),
    "cam_5": (1200, 1000),
}
CALIB_DATE = dt.date(2024, 12, 24)


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


def plot_residual_by_date(d: dict, out: Path, min_tri: float = 0.2) -> dict:
    """Median residual (all cameras, cam_3) by recording date, log y; clips below min_tri grey."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    day = lambda r: dt.date.fromisoformat(r["date"])  # noqa: E731
    ok = [r for r in d["rows"] if r["triangulated_frac"] >= min_tri]
    bad = [r for r in d["rows"] if r["triangulated_frac"] < min_tri]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    ax.scatter(
        [day(r) for r in ok],
        [r["median_px"] for r in ok],
        s=42,
        c="#1f77b4",
        label="all cameras, median",
        zorder=3,
    )
    ax.scatter(
        [day(r) for r in ok],
        [r["per_cam_median_px"][2] for r in ok],
        s=42,
        facecolors="none",
        edgecolors="#d62728",
        label="cam_3 (bottom), median",
        zorder=3,
    )
    if bad:
        ax.scatter(
            [day(r) for r in bad],
            [r["median_px"] for r in bad],
            s=42,
            c="#aaaaaa",
            marker="x",
            label=f"< {min_tri:.0%} of keypoints triangulated (not a calibration measure)",
            zorder=2,
        )
    cal = dt.date.fromisoformat(d["calib_date"])
    ax.axvline(cal, color="k", ls="--", lw=1)
    ax.text(cal, 0.02, " calibration", transform=ax.get_xaxis_transform(), ha="left", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("reprojection residual (px, native)")
    ax.set_xlabel("recording date")
    ax.set_title(
        f"AVATAR: residual of the {d['calib_date']} calibration by recording date\n"
        f"n = {len(d['rows'])} clips, first {d['frames']} frames, "
        f"conf >= {d['thr']}, >= {d['min_cams']} cameras",
        fontsize=9,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {"out": str(out), "clips": len(d["rows"]), "plotted": len(ok), "grey": len(bad)}


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
