"""SUBTLE dataset loader — 3D mouse spontaneous behavior from Kwon et al.

Loads SUBTLE temporal-link embedding dataset (single mouse, 9 keypoints, 3D).
Reference: Kwon et al. (2022), "SUBTLE: An Unsupervised Platform with Temporal
Link Embedding that Maps Animal Behavior", bioRxiv.

Data format:
    - CSV: columns = x1,y1,z1,...,x9,y9,z9 per row (one row per frame)
    - NPY: pre-extracted (T, 9, 3)

Source: https://github.com/jeakwon/subtle
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.types import BehaviorSequence
from ...core.skeleton import get_skeleton


CLASS_NAMES = ["walking", "grooming", "rearing", "standing"]


class SUBTLELoader:
    """Loader for SUBTLE mouse behavior dataset (3D keypoints).

    Expects directory structure:
        subtle/
        ├── *.csv   (or *.npy)
        └── ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "subtle",
        fps: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.skeleton = get_skeleton(skeleton_name)

    def load_csv(self, filepath: str | Path) -> BehaviorSequence:
        """Load a single CSV file.

        Expects columns: x1,y1,z1,...,x9,y9,z9 (27 columns)
        or with optional label column at the end.
        """
        filepath = Path(filepath)
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)

        n_coords = self.skeleton.num_joints * self.skeleton.num_channels  # 9*3=27
        if data.shape[1] > n_coords:
            # Last column(s) might be labels
            labels = data[:, n_coords].astype(int)
            keypoints = data[:, :n_coords]
        else:
            labels = None
            keypoints = data[:, :n_coords]

        T = keypoints.shape[0]
        keypoints = keypoints.reshape(T, self.skeleton.num_joints, self.skeleton.num_channels)

        return BehaviorSequence(
            keypoints=keypoints.astype(np.float32),
            labels=labels,
            skeleton_name=self.skeleton_name,
            sample_id=filepath.stem,
            fps=self.fps,
            metadata={"dataset": "subtle", "source_file": str(filepath)},
        )

    def load_npy(self, filepath: str | Path) -> BehaviorSequence:
        """Load pre-extracted numpy array (T, 9, 3)."""
        filepath = Path(filepath)
        keypoints = np.load(filepath).astype(np.float32)

        if keypoints.ndim == 2:
            T = keypoints.shape[0]
            keypoints = keypoints.reshape(T, self.skeleton.num_joints, self.skeleton.num_channels)

        return BehaviorSequence(
            keypoints=keypoints,
            skeleton_name=self.skeleton_name,
            sample_id=filepath.stem,
            fps=self.fps,
            metadata={"dataset": "subtle", "source_file": str(filepath)},
        )

    def load_preprocessed(self, filepath: str | Path) -> list[BehaviorSequence]:
        """Load from preprocessed .npz file (keypoints: (T, 9, 3) or batch).

        Args:
            filepath: Path to .npz file with 'keypoints' key

        Returns:
            List of BehaviorSequence
        """
        filepath = Path(filepath)
        npz = np.load(filepath, allow_pickle=True)
        keypoints = npz["keypoints"].astype(np.float32)

        if keypoints.ndim == 3:
            # Single sequence (T, K, D)
            return [BehaviorSequence(
                keypoints=keypoints,
                skeleton_name=self.skeleton_name,
                sample_id=filepath.stem,
                fps=self.fps,
                metadata={"dataset": "subtle", "source_file": str(filepath)},
            )]

        # Batch: (N, T, K, D)
        sequences = []
        for i in range(keypoints.shape[0]):
            sequences.append(BehaviorSequence(
                keypoints=keypoints[i],
                skeleton_name=self.skeleton_name,
                sample_id=f"{filepath.stem}_{i:05d}",
                fps=self.fps,
                metadata={"dataset": "subtle", "source_file": str(filepath)},
            ))
        return sequences

    def load_all(self) -> list[BehaviorSequence]:
        """Load all data files from data_dir."""
        sequences = []
        # Try preprocessed .npz first
        for fp in sorted(self.data_dir.glob("*.npz")):
            sequences.extend(self.load_preprocessed(fp))
        if sequences:
            return sequences
        # Fallback to raw files
        for fp in sorted(self.data_dir.glob("*.csv")):
            sequences.append(self.load_csv(fp))
        for fp in sorted(self.data_dir.glob("*.npy")):
            sequences.append(self.load_npy(fp))
        return sequences

    @staticmethod
    def get_class_names() -> list[str]:
        return CLASS_NAMES
