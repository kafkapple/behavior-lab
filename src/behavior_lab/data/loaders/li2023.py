"""Li 2023 mouse keypoint loader — sparse 3D GT from label3d_dannce.mat.

Reference: Li et al. 2023, "MAMMAL: A multi-animal mesh modeling framework"
DOI: 10.1007/s11263-023-01756-3 (PMC10810175).

Used as evaluation reference (NOT a predictor) in kp_benchmark v0.1.
Provides sparse manually-labeled 3D keypoints per session:
  - markerless_mouse_1: 50 valid timepoints
  - markerless_mouse_2: 77 valid timepoints

Data schema (per label3d_dannce.mat):
    camnames        : (N,) str
    params[i]       : K, r, t, RDistort, TDistort
    labelData[0]    : data_3d (M, 22*3), data_frame (M,) — sparse 3D
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio


@dataclass
class Li2023Session:
    """Sparse 3D keypoint reference from Li 2023 manual labels."""
    session_id: str
    frame_ids: np.ndarray              # (M,) int — video frame indices with GT
    keypoints_3d: np.ndarray           # (M, 22, 3) float — manual 3D labels
    cam_names: list[str]
    n_frames_video: Optional[int]      # full video length (from data_frame.max() + buffer)


class Li2023Loader:
    """Loader for Li 2023 mouse sparse 3D GT.

    Parameters
    ----------
    label3d_path : str | Path
        Path to label3d_dannce.mat (per session).
    session_id : str, optional
        Override session identifier (default: parent dir name).
    """

    def __init__(
        self,
        label3d_path: str | Path,
        session_id: Optional[str] = None,
    ):
        self.label3d_path = Path(label3d_path).expanduser()
        self.session_id = session_id or self.label3d_path.parent.parent.name

    def load(self) -> Li2023Session:
        if not self.label3d_path.exists():
            raise FileNotFoundError(
                f"Li 2023 label3d_dannce.mat not found: {self.label3d_path}\n"
                f"Hint: scp gpu03:/node_data_2/joon/data/external/"
                f"markerless_mouse_1/labels/label3d_dannce.mat <local_path>"
            )

        mat = sio.loadmat(self.label3d_path, struct_as_record=False, squeeze_me=True)
        cam_names = [str(n) for n in mat["camnames"]]

        label_data = mat["labelData"]
        ld0 = label_data[0].item() if hasattr(label_data[0], "item") else label_data[0]
        d3 = np.asarray(ld0.data_3d, dtype=np.float64)          # (M, 66)
        df = np.asarray(ld0.data_frame, dtype=np.int64).reshape(-1)  # (M,)
        n_kp = d3.shape[1] // 3
        if n_kp != 22:
            raise ValueError(f"Expected 22 keypoints, got {n_kp}")

        kp3d = d3.reshape(-1, n_kp, 3)
        n_frames_video = int(df.max()) + 1 if df.size > 0 else None

        return Li2023Session(
            session_id=self.session_id,
            frame_ids=df,
            keypoints_3d=kp3d,
            cam_names=cam_names,
            n_frames_video=n_frames_video,
        )
