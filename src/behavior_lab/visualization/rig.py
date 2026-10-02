"""Plots for multi-camera rig checks (input = the json written by `behavior_lab.rig.avatar`)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path


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
