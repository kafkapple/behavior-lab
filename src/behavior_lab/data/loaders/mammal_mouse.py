"""MAMMAL mouse dense 22-kp 3D loader — predictor baseline.

Reference: Li 2023 MAMMAL mesh-fit pipeline.
SSOT file (gpu03): /home/joon/data/results/MAMMAL_mouse/
                   v012345_kp22_20260126/keypoints_22_3d.npz
Shape: (3600, 22, 3), produced by per-frame articulation fit on 6-cam 2D detections.

This is one of the predictors evaluated against Li 2023 manual GT in kp_benchmark v0.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MammalMouseSession:
    """Dense 22-kp 3D predictions from MAMMAL mesh-fit pipeline."""
    session_id: str
    keypoints_3d: np.ndarray   # (T, 22, 3)
    fps: float = 50.0          # M1 capture rate


class MammalMouseLoader:
    """Loader for MAMMAL mouse dense 22-kp 3D predictions.

    Parameters
    ----------
    npz_path : str | Path
        Path to keypoints_22_3d.npz with key 'keypoints_3d' shape (T, 22, 3).
    session_id : str, optional
        Override session identifier (default: parent dir name).
    fps : float
        Acquisition frame rate (M1=50fps).
    """

    def __init__(
        self,
        npz_path: str | Path,
        session_id: Optional[str] = None,
        fps: float = 50.0,
    ):
        self.npz_path = Path(npz_path).expanduser()
        self.session_id = session_id or self.npz_path.parent.name
        self.fps = fps

    def load(self) -> MammalMouseSession:
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"MAMMAL npz not found: {self.npz_path}\n"
                f"Hint: scp gpu03:/home/joon/data/results/MAMMAL_mouse/"
                f"v012345_kp22_20260126/keypoints_22_3d.npz <local_path>"
            )

        data = np.load(self.npz_path)
        keys = list(data.keys())
        kp_key = "keypoints_3d" if "keypoints_3d" in keys else keys[0]
        kp3d = data[kp_key]
        if kp3d.ndim != 3 or kp3d.shape[1] != 22 or kp3d.shape[2] != 3:
            raise ValueError(
                f"Expected (T, 22, 3), got {kp3d.shape} from key '{kp_key}'"
            )

        return MammalMouseSession(
            session_id=self.session_id,
            keypoints_3d=kp3d.astype(np.float64),
            fps=self.fps,
        )
