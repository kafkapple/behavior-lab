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
