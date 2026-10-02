"""aniposelib `config.toml` calibration -> cameras, DLT triangulation, reprojection residual.

Conventions (same as the file aniposelib writes): `matrix` = 3x3 K in OpenCV form, `distortions`
=
[k1, k2, p1, p2, k3], `rotation` = Rodrigues vector world->camera, `translation` =
world->camera.
Moved from BehaviorSplatter `scripts/data_prep/avatar_keypoints_to_gslrm.py` (261002); the GS-
LRM
specific part (preprocessing transform, gap filling) stays there.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tomllib


def load_calib(path: str | Path) -> list[dict]:
    """Cameras in file order: K (3, 3), D (5,), P (3, 4) = [R | t], size (w, h)."""
    cfg = tomllib.loads(Path(path).expanduser().read_text())
    out = []
    for i in range(cfg["camera_count"]):
        c = cfg[f"cam_{i + 1}"]
        R, _ = cv2.Rodrigues(np.array(c["rotation"], float))
        out.append(
            dict(
                K=np.array(c["matrix"], float),
                D=np.array(c["distortions"], float),
                P=np.hstack([R, np.array(c["translation"], float).reshape(3, 1)]),
                size=tuple(c["size"]),
            )
        )
    return out


def _normalized(xy: np.ndarray, cam: dict) -> np.ndarray:
    return cv2.undistortPoints(np.asarray(xy, float).reshape(1, 1, 2), cam["K"], cam["D"])[0, 0]


def triangulate(obs: list[tuple[int, np.ndarray]], cams: list[dict]) -> np.ndarray:
    """DLT over (camera index, distorted pixel xy) observations; needs two or more cameras."""
    assert len(obs) >= 2, "triangulation needs two or more cameras"
    rows = []
    for ci, xy in obs:
        u, P = _normalized(xy, cams[ci]), cams[ci]["P"]
        rows += [u[0] * P[2] - P[0], u[1] * P[2] - P[1]]
    X = np.linalg.svd(np.array(rows))[2][-1]
    return X[:3] / X[3]


def reproj_px(X: np.ndarray, ci: int, xy: np.ndarray, cams: list[dict]) -> float:
    """Distance between the projection of X and the observation, in that camera's native pixels."""
    u = _normalized(xy, cams[ci])
    p = cams[ci]["P"] @ np.append(X, 1.0)
    return float(np.hypot(*(p[:2] / p[2] - u)) * cams[ci]["K"][0, 0])


def rotate_image_90(cam_cfg: dict) -> dict:
    """aniposelib camera table for the same camera when the recorded image is turned a quarter turn.

    Recorded pixel = calibrated pixel rotated about the image centre so that calibrated (x, y) lands
    at (y, w - x) (square sensor, w = h). Exact: K gets fx/fy and cx/cy swapped with cy' = w - cx,
    the extrinsic is premultiplied by the same quarter turn about the optical axis, tangential
    distortion becomes (p1', p2') = (-p2, p1). Used for the AVATAR bottom camera in recordings up to
    2023-05-23 (see docs/avatar_rig.md).
    """
    w, h = cam_cfg["size"]
    assert w == h, "a quarter turn keeps the image size only on a square sensor"
    K = np.array(cam_cfg["matrix"], float)
    k1, k2, p1, p2, k3 = cam_cfg["distortions"]
    Rz = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R, _ = cv2.Rodrigues(np.array(cam_cfg["rotation"], float))
    rvec, _ = cv2.Rodrigues(Rz @ R)
    out = dict(cam_cfg)
    fx, fy, cx, cy = (float(v) for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    out["matrix"] = [[fy, 0.0, cy], [0.0, fx, w - cx], [0.0, 0.0, 1.0]]
    out["distortions"] = [k1, k2, -p2, p1, k3]
    out["rotation"] = [float(v) for v in rvec.ravel()]
    out["translation"] = [float(v) for v in Rz @ np.array(cam_cfg["translation"], float)]
    return out


def write_calib(cfg: dict, path: str | Path) -> None:
    """Minimal writer for the aniposelib layout (scalars, strings, lists, one table level)."""

    def val(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        if hasattr(v, "isoformat"):  # TOML datetime
            return v.isoformat()
        return repr(v)

    top = [f"{k} = {val(v)}" for k, v in cfg.items() if not isinstance(v, dict)]
    tables = [
        f"\n[{k}]\n" + "\n".join(f"{a} = {val(b)}" for a, b in v.items())
        for k, v in cfg.items()
        if isinstance(v, dict)
    ]
    Path(path).expanduser().write_text("\n".join(top) + "\n" + "\n".join(tables) + "\n")
