"""Rat7M dataset loader — 3D rat mocap from Dunn et al.

Loads Rat7M DANNCE/MAT keypoint data (single rat, 20 keypoints, 3D).
Reference: Dunn et al. (2021), "Geometric deep learning enables 3D kinematic profiling
across species and environments", Nature Methods.

Data formats:
    - .mat (MATLAB): 'mocap_markers' or 'positions_3d' fields
    - .h5 (DANNCE): 'positions' group
    - .npy: pre-extracted (T, K, 3)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.types import BehaviorSequence
from ...core.skeleton import get_skeleton, register_skeleton, SkeletonDefinition, SKELETON_REGISTRY


# Rat7M 20-joint skeleton definition
RAT7M_JOINT_NAMES = [
    "nose_tip", "head_top", "left_ear", "right_ear",
    "neck", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "spine_mid",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "tail_base", "tail_mid",
]

RAT7M_SKELETON = SkeletonDefinition(
    name="rat7m",
    num_joints=20,
    joint_names=RAT7M_JOINT_NAMES,
    joint_parents=[
        4, 4, 4, 4,  # nose, head, ears -> neck
        -1,  # neck (root)
        4, 4,  # shoulders -> neck
        5, 6,  # elbows -> shoulders
        7, 8,  # wrists -> elbows
        4,  # spine_mid -> neck
        11, 11,  # hips -> spine_mid
        12, 13,  # knees -> hips
        14, 15,  # ankles -> knees
        11, 18,  # tail_base -> spine_mid, tail_mid -> tail_base
    ],
    edges=[
        (0, 4), (1, 4), (2, 4), (3, 4),  # head -> neck
        (4, 5), (4, 6),  # neck -> shoulders
        (5, 7), (6, 8),  # shoulders -> elbows
        (7, 9), (8, 10),  # elbows -> wrists
        (4, 11),  # neck -> spine
        (11, 12), (11, 13),  # spine -> hips
        (12, 14), (13, 15),  # hips -> knees
        (14, 16), (15, 17),  # knees -> ankles
        (11, 18), (18, 19),  # spine -> tail
    ],
    symmetric_pairs=[
        (2, 3), (5, 6), (7, 8), (9, 10),
        (12, 13), (14, 15), (16, 17),
    ],
    num_channels=3,
    coordinate_system="xyz",
    body_parts={
        "head": [0, 1, 2, 3],
        "torso": [4, 5, 6, 11],
        "left_arm": [5, 7, 9],
        "right_arm": [6, 8, 10],
        "left_leg": [12, 14, 16],
        "right_leg": [13, 15, 17],
        "tail": [18, 19],
    },
    center_joint=4,
)

# Auto-register if not already present
if "rat7m" not in SKELETON_REGISTRY:
    register_skeleton("rat7m", RAT7M_SKELETON)
    register_skeleton("rat7m_20", RAT7M_SKELETON)


class Rat7MLoader:
    """Loader for Rat7M 3D rat motion capture dataset.

    Supports multiple file formats:
        - .mat: MATLAB files with mocap_markers/positions_3d
        - .h5: DANNCE HDF5 format
        - .npy: Pre-extracted numpy arrays

    Expects directory structure:
        rat7m/
        ├── session_001.mat  (or .h5, .npy)
        ├── session_002.mat
        └── ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "rat7m",
        fps: float = 120.0,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.skeleton = get_skeleton(skeleton_name)

    def load_session(self, filepath: str | Path) -> BehaviorSequence:
        """Load a single session file.

        Args:
            filepath: Path to .mat, .h5, or .npy file

        Returns:
            BehaviorSequence with shape (T, 20, 3)
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == ".npy":
            keypoints = np.load(filepath)
        elif suffix == ".mat":
            keypoints = self._load_mat(filepath)
        elif suffix in (".h5", ".hdf5"):
            keypoints = self._load_hdf5(filepath)
        else:
            raise ValueError(f"Unsupported format: {suffix}. Use .mat, .h5, or .npy")

        if keypoints.ndim != 3 or keypoints.shape[1:] != (self.skeleton.num_joints, 3):
            raise ValueError(
                f"Expected shape (T, {self.skeleton.num_joints}, 3), "
                f"got {keypoints.shape}"
            )

        return BehaviorSequence(
            keypoints=keypoints.astype(np.float32),
            skeleton_name=self.skeleton_name,
            sample_id=filepath.stem,
            fps=self.fps,
            metadata={"dataset": "rat7m", "source_file": str(filepath)},
        )

    def _load_mat(self, filepath: Path) -> np.ndarray:
        """Load from MATLAB .mat file."""
        from scipy.io import loadmat

        mat = loadmat(str(filepath))
        for key in ("mocap_markers", "positions_3d", "keypoints", "data"):
            if key in mat:
                data = mat[key]
                if data.ndim == 2:
                    T = data.shape[0]
                    data = data.reshape(T, -1, 3)
                return data
        raise KeyError(
            f"No recognized data key in {filepath}. "
            f"Found: {[k for k in mat if not k.startswith('_')]}"
        )

    def _load_hdf5(self, filepath: Path) -> np.ndarray:
        """Load from DANNCE HDF5 format."""
        import h5py

        with h5py.File(filepath, "r") as f:
            for key in ("positions", "keypoints", "predictions"):
                if key in f:
                    return np.array(f[key])
            raise KeyError(
                f"No recognized data key in {filepath}. Found: {list(f.keys())}"
            )

    def load_all(self) -> list[BehaviorSequence]:
        """Load all session files from data_dir."""
        sequences = []
        for ext in ("*.npy", "*.mat", "*.h5", "*.hdf5"):
            for fp in sorted(self.data_dir.glob(ext)):
                sequences.append(self.load_session(fp))
        return sequences
