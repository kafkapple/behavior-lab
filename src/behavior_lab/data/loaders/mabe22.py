"""MABe22 dataset loader — Mouse triplet social behavior from Sun et al.

Loads MABe2022 mouse triplet keypoint data (3 mice × 12 keypoints × 2D).
Reference: Sun et al. (2022), "MABe22: A Multi-Agent Behavior Dataset", ICML.

Data format:
    - .npy: mouse_triplet_{train,test}.npy — shape varies
    - .npz: preprocessed (N, T, 36, 2) from preprocess_data.py

Body parts (12 per mouse):
    nose, left_ear, right_ear, neck, left_forepaw, right_forepaw,
    center_back, left_hindpaw, right_hindpaw, tail_base, tail_middle, tail_tip

Source: https://data.caltech.edu/records/s0vdx-0k302
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.types import BehaviorSequence
from ...core.skeleton import get_skeleton


JOINT_NAMES = [
    "nose", "left_ear", "right_ear", "neck",
    "left_forepaw", "right_forepaw", "center_back",
    "left_hindpaw", "right_hindpaw",
    "tail_base", "tail_middle", "tail_tip",
]

NUM_MICE = 3
NUM_JOINTS_PER_MOUSE = 12


class MABe22Loader:
    """Loader for MABe2022 mouse triplet behavior dataset.

    Expects directory structure:
        mabe22/
        ├── mabe22_train.npz   (preprocessed)
        ├── mabe22_test.npz    (preprocessed)
        └── mouse_triplet_*.npy (raw)
    """

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "mabe22",
        fps: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.skeleton = get_skeleton(skeleton_name)

    def load_split(self, split: str = "train") -> list[BehaviorSequence]:
        """Load a data split from preprocessed .npz or raw .npy.

        Args:
            split: 'train' or 'test'

        Returns:
            List of BehaviorSequence, one per sequence
        """
        # Try preprocessed .npz first
        npz_path = self.data_dir / f"mabe22_{split}.npz"
        if npz_path.exists():
            return self._load_npz(npz_path, split)

        # Fallback to raw .npy
        npy_path = self.data_dir / f"mouse_triplet_{split}.npy"
        if npy_path.exists():
            return self._load_npy(npy_path, split)

        raise FileNotFoundError(
            f"No data found for split '{split}' in {self.data_dir}. "
            f"Expected {npz_path.name} or {npy_path.name}"
        )

    def _load_npz(self, path: Path, split: str) -> list[BehaviorSequence]:
        """Load preprocessed .npz with keypoints (N, T, K_total, D)."""
        npz = np.load(path, allow_pickle=True)
        data = npz["keypoints"].astype(np.float32)

        if data.ndim == 3:
            # Single sequence (T, K, D)
            data = data[np.newaxis]

        sequences = []
        for i in range(data.shape[0]):
            kp = data[i]  # (T, K_total, D)
            sequences.append(BehaviorSequence(
                keypoints=kp,
                skeleton_name=self.skeleton_name,
                sample_id=f"{split}_{i:05d}",
                fps=self.fps,
                metadata={
                    "dataset": "mabe22",
                    "split": split,
                    "num_mice": NUM_MICE,
                    "joints_per_mouse": NUM_JOINTS_PER_MOUSE,
                },
            ))
        return sequences

    def _load_npy(self, path: Path, split: str) -> list[BehaviorSequence]:
        """Load raw .npy with flexible shape handling."""
        data = np.load(path, allow_pickle=True).astype(np.float32)

        # Reshape to (N, T, K_total, D)
        if data.ndim == 5:
            # (N, T, n_mice, K, D) → (N, T, n_mice*K, D)
            N, T, M, K, D = data.shape
            data = data.reshape(N, T, M * K, D)
        elif data.ndim == 3:
            # (T, K, D) single sequence
            data = data[np.newaxis]
        elif data.ndim != 4:
            raise ValueError(f"Unexpected shape {data.shape} for MABe22 data")

        sequences = []
        for i in range(data.shape[0]):
            kp = data[i]  # (T, K_total, D)
            sequences.append(BehaviorSequence(
                keypoints=kp,
                skeleton_name=self.skeleton_name,
                sample_id=f"{split}_{i:05d}",
                fps=self.fps,
                metadata={
                    "dataset": "mabe22",
                    "split": split,
                    "num_mice": NUM_MICE,
                    "joints_per_mouse": NUM_JOINTS_PER_MOUSE,
                },
            ))
        return sequences

    def load_all(self) -> dict[str, list[BehaviorSequence]]:
        """Load all available splits.

        Tries standard split names first, then falls back to any .npz/.npy
        files found in the data directory.
        """
        result: dict[str, list[BehaviorSequence]] = {}
        for split in ["train", "test"]:
            try:
                result[split] = self.load_split(split)
            except FileNotFoundError:
                continue

        # Fallback: load any .npz or .npy file not yet covered
        if not result:
            for npz in sorted(self.data_dir.glob("*.npz")):
                name = npz.stem
                if name not in result:
                    result[name] = self._load_npz(npz, split=name)
            for npy in sorted(self.data_dir.glob("*.npy")):
                name = npy.stem
                if name not in result:
                    result[name] = self._load_npy(npy, split=name)

        return result

    def load_preprocessed(self, filepath: str | Path) -> list[BehaviorSequence]:
        """Load from a specific preprocessed .npz file."""
        filepath = Path(filepath)
        return self._load_npz(filepath, split=filepath.stem)

    @staticmethod
    def get_class_names() -> list[str]:
        return []  # MABe22 uses unsupervised discovery
