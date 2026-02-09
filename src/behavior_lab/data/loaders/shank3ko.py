"""Shank3KO dataset loader — 3D mouse behavior from Huang et al.

Loads Shank3KO knockout mouse dataset (single mouse, 16 keypoints, 3D).
Reference: Huang et al. (2021), "A hierarchical 3D-motion learning framework
for animal spontaneous behavior mapping", Nature Communications.

Data format:
    - .mat (MATLAB): keypoint data + optional behavior labels
    - .npy: pre-extracted (T, 16, 3)

Source: https://doi.org/10.5281/zenodo.4629544
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.types import BehaviorSequence
from ...core.skeleton import get_skeleton


CLASS_NAMES = [
    "running", "trotting", "stepping", "diving", "sniffing",
    "rising", "right_turning", "up_stretching", "falling",
    "left_turning", "walking",
]


class Shank3KOLoader:
    """Loader for Shank3KO 3D mouse behavior dataset.

    Expects directory structure:
        shank3ko/
        ├── *.mat  (MATLAB files with 3D keypoints)
        ├── *.npy  (pre-extracted arrays)
        └── ...

    Note: Raw Zenodo data may require 3D reconstruction preprocessing.
    This loader handles pre-extracted 3D keypoints.
    """

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "shank3ko",
        fps: float = 60.0,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.skeleton = get_skeleton(skeleton_name)

    def load_mat(self, filepath: str | Path) -> BehaviorSequence:
        """Load from MATLAB .mat file.

        Searches for common data keys: 'keypoints_3d', 'positions',
        'keypoints', 'data'.
        """
        from scipy.io import loadmat

        filepath = Path(filepath)
        mat = loadmat(str(filepath))

        keypoints = None
        labels = None

        # Find keypoint data
        for key in ("keypoints_3d", "positions", "keypoints", "data", "pose"):
            if key in mat:
                keypoints = np.array(mat[key], dtype=np.float32)
                break

        if keypoints is None:
            available = [k for k in mat if not k.startswith("_")]
            raise KeyError(
                f"No recognized keypoint key in {filepath}. Found: {available}"
            )

        # Reshape if needed
        if keypoints.ndim == 2:
            T = keypoints.shape[0]
            keypoints = keypoints.reshape(T, -1, 3)

        # Find labels if available
        for key in ("labels", "behavior_labels", "annotations"):
            if key in mat:
                labels = np.array(mat[key]).flatten().astype(int)
                break

        return BehaviorSequence(
            keypoints=keypoints,
            labels=labels,
            skeleton_name=self.skeleton_name,
            sample_id=filepath.stem,
            fps=self.fps,
            metadata={"dataset": "shank3ko", "source_file": str(filepath)},
        )

    def load_npy(self, filepath: str | Path) -> BehaviorSequence:
        """Load pre-extracted numpy array (T, 16, 3)."""
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
            metadata={"dataset": "shank3ko", "source_file": str(filepath)},
        )

    def load_preprocessed(self, filepath: str | Path) -> list[BehaviorSequence]:
        """Load from preprocessed .npz file.

        Args:
            filepath: Path to .npz with 'keypoints' (T, 16, 3) and optional 'labels', 'genotype'

        Returns:
            List of BehaviorSequence
        """
        filepath = Path(filepath)
        npz = np.load(filepath, allow_pickle=True)
        keypoints = npz["keypoints"].astype(np.float32)
        labels = npz["labels"].astype(int) if "labels" in npz else None
        genotype = str(npz["genotype"]) if "genotype" in npz else None

        meta = {"dataset": "shank3ko", "source_file": str(filepath)}
        if genotype:
            meta["genotype"] = genotype

        if keypoints.ndim == 3:
            return [BehaviorSequence(
                keypoints=keypoints,
                labels=labels,
                skeleton_name=self.skeleton_name,
                sample_id=filepath.stem,
                fps=self.fps,
                metadata=meta,
            )]

        sequences = []
        for i in range(keypoints.shape[0]):
            seq_labels = labels[i] if labels is not None and labels.ndim > 1 else labels
            sequences.append(BehaviorSequence(
                keypoints=keypoints[i],
                labels=seq_labels,
                skeleton_name=self.skeleton_name,
                sample_id=f"{filepath.stem}_{i:05d}",
                fps=self.fps,
                metadata=dict(meta),
            ))
        return sequences

    def load_all(self) -> list[BehaviorSequence]:
        """Load all data files from data_dir."""
        sequences = []
        # Try preprocessed .npz first
        for fp in sorted(self.data_dir.glob("*.npz")):
            try:
                sequences.extend(self.load_preprocessed(fp))
            except Exception as e:
                print(f"Warning: Could not load {fp}: {e}")
        if sequences:
            return sequences
        # Fallback to raw files
        for ext in ("*.mat", "*.npy"):
            for fp in sorted(self.data_dir.glob(ext)):
                try:
                    if fp.suffix == ".mat":
                        sequences.append(self.load_mat(fp))
                    else:
                        sequences.append(self.load_npy(fp))
                except Exception as e:
                    print(f"Warning: Could not load {fp}: {e}")
        return sequences

    @staticmethod
    def get_class_names() -> list[str]:
        return CLASS_NAMES
