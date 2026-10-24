"""Descriptive check of `multiview.triangulate_sequence` on one AVATAR json.gz (PRD P7).

No ground truth: every number is agreement, not accuracy.
  - reprojection px per method: over all cameras with conf > thr, and over the cameras a method
    saw but did not use (top2 only; a 2-camera DLT fits its own two cameras almost exactly);
  - vs the file's own skeleton3d: orthogonal similarity fit (reflection allowed, the recon3D frame
    may be mirrored) of skeleton3d onto the method's points, then the 3D distance after the fit and
    the reprojection of the mapped skeleton3d. The fit is an assumption, so this bounds disagreement.

    python scripts/rig_triangulation_check.py --calib config.toml --kp cort1.json.gz --step 10
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from behavior_lab.rig.avatar import kp2d_cells
from behavior_lab.rig.multiview import METHODS, load_calib, reproj_px, triangulate_sequence


def similarity(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """B ~ s Q A + t with Q orthogonal (det may be -1); least squares (Umeyama without the sign fix)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a)
    Q = U @ Vt
    s = S.sum() / (a**2).sum()
    return float(s), Q, mb - s * Q @ ma


def residual_px(xyz, kp2d, cams, thr, skip=None) -> list[float]:
    """Reprojection of finite points into every camera with conf > thr (minus `skip` mask)."""
    out = []
    for t, k in zip(*np.nonzero(np.isfinite(xyz[..., 0]))):
        for c in range(len(cams)):
            if kp2d[t, c, k, 2] > thr and (skip is None or not skip[t, k, c]):
                out.append(reproj_px(xyz[t, k], c, kp2d[t, c, k, :2], cams))
    return out


def stats(v: list[float]) -> dict:
    return {"n": len(v), "median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2)} if v else {"n": 0}


def check(calib: Path, kp_path: Path, step: int, thr: float) -> dict:
    cams = load_calib(calib)
    d = json.load(gzip.open(kp_path.expanduser()))
    frames = sorted(int(f) for f in d["keypoint"])[::step]
    kp2d = kp2d_cells(d, frames)
    sk = np.array([d["skeleton3d"][str(f)] for f in frames], float)
    assert sk.shape == kp2d.shape[:1] + kp2d.shape[2:], (sk.shape, kp2d.shape)
    out = {"file": kp_path.name, "frames": len(frames), "step": step, "thr": thr, "methods": {}}
    for m in METHODS:
        xyz, used = triangulate_sequence(kp2d, cams, m, thr)
        ok = np.isfinite(xyz[..., 0])
        assert ((used >= 2) == ok).all() and (m != "top2" or used.max() <= 2)
        r = {"triangulated_frac": round(float(ok.mean()), 4), "cams_used_mean": round(float(used[ok].mean()), 2),
             "reproj_px_seen": stats(residual_px(xyz, kp2d, cams, thr))}
        if m == "top2":  # cameras above thr that top2 left out = an out-of-fit check
            rank = np.argsort(np.argsort(-kp2d[..., 2], axis=1, kind="stable"), axis=1)  # (T, C, N)
            r["reproj_px_unused"] = stats(residual_px(xyz, kp2d, cams, thr, skip=(rank < 2).transpose(0, 2, 1)))
        both = ok & np.isfinite(sk).all(-1)
        s, Q, t = similarity(sk[both], xyz[both])
        mapped = sk @ (s * Q).T + t
        r["vs_skeleton3d"] = {"pairs": int(both.sum()), "scale": round(s, 4), "det": round(float(np.linalg.det(Q)), 3),
                              "dist3d_median": round(float(np.median(np.linalg.norm(mapped[both] - xyz[both], axis=1))), 4),
                              "reproj_px_mapped": stats(residual_px(np.where(both[..., None], mapped, np.nan), kp2d, cams, thr))}
        out["methods"][m] = r
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--kp", type=Path, required=True)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.25)
    a = ap.parse_args()
    print(json.dumps(check(a.calib, a.kp, a.step, a.thr), indent=1))
