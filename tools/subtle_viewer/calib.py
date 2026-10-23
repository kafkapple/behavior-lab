"""Calibrated multi-camera geometry for the AVATAR rig (aniposelib-style toml: per camera matrix, distortions, rotation vector, translation).
Gives the standard held-out reprojection error: triangulate a node without one camera, project it into that camera, compare with the 2D detection.
Pixels are camera-cell pixels (core.CELLS), which match the calibration without any crop offset (checked by epipolar distance, 261024). No web code."""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

import core

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "avatar" / "calibration" / "config_20241224.toml"   # copy of Olaf ~/data/avatar_gslrm/config.toml (md5 d97807ae...)


def rodrigues(r: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Rig:
    def __init__(self, cfg: dict):
        self.cams = []
        for i in range(1, cfg.get("camera_count", 5) + 1):
            c = cfg[f"cam_{i}"]
            self.cams.append({"R": rodrigues(np.array(c["rotation"], float)), "t": np.array(c["translation"], float), "M": np.array(c["matrix"], float), "d": np.array(c["distortions"], float)})

    def project(self, X: np.ndarray, i: int) -> np.ndarray:
        c = self.cams[i]
        Xc = X @ c["R"].T + c["t"]
        x = Xc[:, :2] / Xc[:, 2:3]
        k1, k2, p1, p2, k3 = c["d"]
        r2 = (x ** 2).sum(1)
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = np.c_[x[:, 0] * rad + 2 * p1 * x[:, 0] * x[:, 1] + p2 * (r2 + 2 * x[:, 0] ** 2), x[:, 1] * rad + p1 * (r2 + 2 * x[:, 1] ** 2) + 2 * p2 * x[:, 0] * x[:, 1]]
        return xd * [c["M"][0, 0], c["M"][1, 1]] + c["M"][:2, 2]

    def undistort(self, px: np.ndarray, i: int) -> np.ndarray:
        """Pixel -> normalized camera coordinates (fixed-point inverse of the 5-coefficient distortion)."""
        c = self.cams[i]
        x = (px - c["M"][:2, 2]) / [c["M"][0, 0], c["M"][1, 1]]
        y = x.copy()
        k1, k2, p1, p2, k3 = c["d"]
        for _ in range(15):
            r2 = (y ** 2).sum(1)
            rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            dx = np.c_[2 * p1 * y[:, 0] * y[:, 1] + p2 * (r2 + 2 * y[:, 0] ** 2), p1 * (r2 + 2 * y[:, 1] ** 2) + 2 * p2 * y[:, 0] * y[:, 1]]
            y = (x - dx) / rad[:, None]
        return y

    def triangulate(self, uv: list[np.ndarray], cams: list[int]) -> np.ndarray:
        """DLT from normalized points of two or more cameras."""
        A = []
        for (u, v), i in zip(uv, cams):
            P = np.c_[self.cams[i]["R"], self.cams[i]["t"]]
            A += [u * P[2] - P[0], v * P[2] - P[1]]
        X = np.linalg.svd(np.array(A))[2][-1]
        return X[:3] / X[3]


def load(path: Path | None = None) -> Rig:
    with open(path or DEFAULT, "rb") as f:
        return Rig(tomllib.load(f))


def held_out_reprojection(rig: Rig, s: dict, step: int = 25, conf: float = 0.5) -> dict:
    """Per node: median px of held-out reprojection over node-frames seen by 3 or more cameras (every `step` frames). Needs 2D keypoints in the file."""
    kp = np.stack([core.kp_cell(s, c) for c in range(len(rig.cams))], 1)[::step]                  # (F, cams, K, 3), x y normalized to the cell
    px = kp[..., :2] * np.array([[core.CELLS[c][2], core.CELLS[c][3]] for c in range(len(rig.cams))])[None, :, None, :]
    und = {c: rig.undistort(px[:, c].reshape(-1, 2), c).reshape(px.shape[0], px.shape[2], 2) for c in range(len(rig.cams))}
    err = [[] for _ in range(kp.shape[2])]
    for f in range(kp.shape[0]):
        for k in range(kp.shape[2]):
            seen = [c for c in range(len(rig.cams)) if kp[f, c, k, 2] > conf]
            if len(seen) < 3:
                continue
            for c in seen:
                rest = [x for x in seen if x != c]
                X = rig.triangulate([und[x][f, k] for x in rest], rest)
                err[k].append(float(np.linalg.norm(rig.project(X[None], c)[0] - px[f, c, k])))
    nodes = [n.removesuffix("1") for n in s["nodes"]]
    return {"nodes": nodes, "median_px": [float(np.median(e)) if e else None for e in err], "n": [len(e) for e in err], "all": float(np.median(sum(err, [])))}
