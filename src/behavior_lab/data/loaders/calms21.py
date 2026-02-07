"""CalMS21 dataset loader — Mouse social behavior from Caltech.

Loads CalMS21 keypoint data (2 mice × 7 keypoints × 2D) with behavior annotations.
Reference: Sun et al. (2021), "The Multi-Agent Behavior Dataset", NeurIPS.

Data format:
    - .npy files: keypoints shape (N, T, 2*K*2) or (T, 2*K*2)
    - Annotations: 'attack', 'investigation', 'mount', 'other'
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core.types import BehaviorSequence
from ...core.skeleton import get_skeleton


CLASS_NAMES = ["other", "attack", "investigation", "mount"]


class CalMS21Loader:
    """Loader for CalMS21 mouse social behavior dataset.

    Expects directory structure:
        calms21/
        ├── train_data.npy       # (N, T, F) or (T, F)
        ├── train_labels.npy     # (N, T) or (T,)
        ├── test_data.npy
        └── test_labels.npy

    Where F = num_mice * num_keypoints * 2 = 2 * 7 * 2 = 28
    """

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "calms21",
        fps: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.skeleton = get_skeleton(skeleton_name)

    def load_split(self, split: str = "train") -> list[BehaviorSequence]:
        """Load a data split and return list of BehaviorSequence.

        Args:
            split: 'train' or 'test'

        Returns:
            List of BehaviorSequence, one per sequence (or one if single-sequence)
        """
        data_path = self.data_dir / f"{split}_data.npy"
        label_path = self.data_dir / f"{split}_labels.npy"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = np.load(data_path)
        labels = np.load(label_path) if label_path.exists() else None

        return self._parse_sequences(data, labels, split)

    def _parse_sequences(
        self, data: np.ndarray, labels: np.ndarray | None, split: str
    ) -> list[BehaviorSequence]:
        """Parse raw arrays into BehaviorSequence objects."""
        K = self.skeleton.num_joints  # 7
        D = self.skeleton.num_channels  # 2
        num_mice = self.skeleton.num_persons  # 2

        if data.ndim == 2:
            data = data[np.newaxis]  # (1, T, F)
            if labels is not None and labels.ndim == 1:
                labels = labels[np.newaxis]

        sequences = []
        for i in range(data.shape[0]):
            seq_data = data[i]  # (T, F)
            T = seq_data.shape[0]

            # Reshape: (T, F) -> (T, num_mice * K, D)
            keypoints = seq_data.reshape(T, num_mice * K, D)

            seq_labels = labels[i] if labels is not None else None

            sequences.append(BehaviorSequence(
                keypoints=keypoints.astype(np.float32),
                labels=seq_labels,
                skeleton_name=self.skeleton_name,
                sample_id=f"{split}_{i:04d}",
                fps=self.fps,
                metadata={
                    "dataset": "calms21",
                    "split": split,
                    "class_names": CLASS_NAMES,
                    "num_mice": num_mice,
                },
            ))

        return sequences

    def load_all(self) -> dict[str, list[BehaviorSequence]]:
        """Load all available splits."""
        result: dict[str, list[BehaviorSequence]] = {}
        for split in ["train", "test"]:
            try:
                result[split] = self.load_split(split)
            except FileNotFoundError:
                continue
        return result

    @staticmethod
    def class_names() -> list[str]:
        return CLASS_NAMES
