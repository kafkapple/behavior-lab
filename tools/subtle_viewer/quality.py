"""Label-free quality metrics of the reconstructed 3D skeleton, per session. There is no ground truth yet, so these are consistency checks, not accuracy:
a bad value flags a session; a good value does not prove the pose is correct. See docs/dev/subtle_3d_quality_plan.md. No web code."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import core

CONF = 0.25          # recon3D keeps a 2D point only above this confidence (subtle_pipeline_io.md)
JUMP_BODY = 0.5      # a node moving more than this fraction of the body length in one frame (50 ms) is treated as a jump


def copied_mask(xyz: np.ndarray) -> np.ndarray:
    """(T,) True where at least one node equals its position in the previous frame in all three coordinates (recon3D copies the last value when it
    cannot triangulate). A real animal does not hold a float coordinate exactly still, so an exact repeat is a copy. Frame 0 is False."""
    m = np.zeros(len(xyz), bool)
    m[1:] = (np.diff(xyz, axis=0) == 0).all(2).any(1)
    return m


def session_quality(s: dict) -> dict:
    x = s["xyz"].astype(np.float64)
    T, K = len(x), x.shape[1]
    same = np.zeros((T, K), bool); same[1:] = (np.diff(x, axis=0) == 0).all(2)
    out = {"frames": T, "copied_frames": float(same.any(1).mean()), "copied_nodes": float(same.mean())}
    L = np.array([np.linalg.norm(x[:, b] - x[:, a], axis=1) for a, b in s["edges"]])           # (edges, T) bone lengths
    cv = L.std(1) / L.mean(1)
    head = [i for i, (a, b) in enumerate(s["edges"]) if {s["nodes"][a], s["nodes"][b]} & {"nose1", "earL1", "earR1"}]   # head edges are nearly rigid; limbs and spine are not
    out["head_bone_cv"] = float(np.median(cv[head])) if head else None                          # large = the 3D head points are noisy
    out["bone_cv_all"] = float(np.median(cv))
    body = float(np.median(np.linalg.norm(x[:, s["nodes"].index("neck1")] - x[:, s["nodes"].index("tailstart1")], axis=1))) if {"neck1", "tailstart1"} <= set(s["nodes"]) else float(np.median(L))
    step = np.linalg.norm(np.diff(x, axis=0), axis=2)                                           # (T-1, nodes) displacement per frame
    out["jump_frames"] = float((step.max(1) > JUMP_BODY * body).mean())
    if s["kp"] is not None:
        ok = (s["kp"][..., 2] > CONF).sum(1)                                                    # (T, nodes): cameras above the confidence line
        out["tri_support"] = float((ok >= 2).mean())                                           # node-frames with at least 2 cameras: triangulation possible
        out["cams_mean"] = float(ok.mean())
        out["unsupported_vs_copied"] = float(((ok < 2)[1:] & same[1:]).sum() / max(1, (ok < 2)[1:].sum()))   # of node-frames with <2 cameras, share that are copies
        res = []
        for cam in range(s["kp"].shape[1]):
            try:
                res.append(core.cam_view(s, cam)["resid_px"])
            except ValueError:
                res.append(None)
        out["reproj_px"] = res                                                                  # weak-perspective residual per camera (not a calibrated reprojection error)
    return out


def project_quality(root: Path, proj: dict) -> dict:
    rows = []
    for g, sess in proj.get("groups", {}).items():
        for r in sess:
            rows.append({"group": g, "session": Path(r).stem, "path": r, **session_quality(core.load(root, r))})
    return {"rows": rows, "note": "consistency checks without ground truth; see subtle_3d_quality_plan.md"}
