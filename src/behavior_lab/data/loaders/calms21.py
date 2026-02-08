"""CalMS21 dataset loader — Mouse social behavior from Caltech.

Loads CalMS21 keypoint data (2 mice × 7 keypoints × 2D) with behavior annotations.
Reference: Sun et al. (2021), "The Multi-Agent Behavior Dataset", NeurIPS.

Data formats:
    - .npz: x_{split} shape (N, 2, T, 7, 2), y_{split} one-hot (N, 4)
    - .npy: keypoints shape (N, T, 28) flat, labels (N, T) per-frame
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

    Supports two directory layouts:

    NPZ format (preferred):
        calms21/
        └── calms21_aligned.npz   # x_train (N,2,T,7,2), y_train (N,4) one-hot

    Legacy .npy format:
        calms21/
        ├── train_data.npy        # (N, T, 28)
        └── train_labels.npy      # (N, T)
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
        """Load a data split, trying NPZ first then falling back to .npy.

        Args:
            split: 'train' or 'test'

        Returns:
            List of BehaviorSequence, one per sequence
        """
        # Try NPZ files first
        for npz_path in sorted(self.data_dir.glob("*.npz")):
            npz = np.load(npz_path, allow_pickle=True)
            x_key = f"x_{split}"
            y_key = f"y_{split}"
            if x_key in npz:
                return self._parse_npz(npz[x_key], npz.get(y_key), split)

        # Fallback to .npy
        data_path = self.data_dir / f"{split}_data.npy"
        label_path = self.data_dir / f"{split}_labels.npy"

        if not data_path.exists():
            raise FileNotFoundError(
                f"No NPZ with x_{split} key or {data_path} found"
            )

        data = np.load(data_path)
        labels = np.load(label_path) if label_path.exists() else None
        return self._parse_flat(data, labels, split)

    def _parse_npz(
        self, data: np.ndarray, labels: np.ndarray | None, split: str
    ) -> list[BehaviorSequence]:
        """Parse NPZ arrays: x (N, 2, T, 7, 2), y (N, C) one-hot."""
        K = self.skeleton.num_joints   # 7
        D = self.skeleton.num_channels  # 2
        num_mice = self.skeleton.num_persons  # 2

        # data: (N, num_mice, T, K, D) -> (N, T, num_mice*K, D)
        N, M, T, Kd, Dd = data.shape
        data_tkd = data.transpose(0, 2, 1, 3, 4).reshape(N, T, M * Kd, Dd)

        # One-hot labels -> int
        int_labels = None
        if labels is not None:
            if labels.ndim == 2 and labels.shape[1] > 1:
                int_labels = labels.argmax(axis=1)
            else:
                int_labels = labels.flatten().astype(int)

        sequences = []
        for i in range(N):
            kp = data_tkd[i]  # (T, 14, 2)
            label = int(int_labels[i]) if int_labels is not None else None
            per_frame = np.full(T, label, dtype=np.int64) if label is not None else None

            sequences.append(BehaviorSequence(
                keypoints=kp.astype(np.float32),
                labels=per_frame,
                skeleton_name=self.skeleton_name,
                sample_id=f"{split}_{i:05d}",
                fps=self.fps,
                metadata={
                    "dataset": "calms21",
                    "split": split,
                    "class_names": CLASS_NAMES,
                    "num_mice": num_mice,
                    "action_label": label,
                },
            ))
        return sequences

    def _parse_flat(
        self, data: np.ndarray, labels: np.ndarray | None, split: str
    ) -> list[BehaviorSequence]:
        """Parse legacy flat arrays: data (N, T, F), labels (N, T)."""
        K = self.skeleton.num_joints
        D = self.skeleton.num_channels
        num_mice = self.skeleton.num_persons

        if data.ndim == 2:
            data = data[np.newaxis]
            if labels is not None and labels.ndim == 1:
                labels = labels[np.newaxis]

        sequences = []
        for i in range(data.shape[0]):
            seq_data = data[i]  # (T, F)
            T = seq_data.shape[0]
            keypoints = seq_data.reshape(T, num_mice * K, D)
            seq_labels = labels[i] if labels is not None else None

            sequences.append(BehaviorSequence(
                keypoints=keypoints.astype(np.float32),
                labels=seq_labels,
                skeleton_name=self.skeleton_name,
                sample_id=f"{split}_{i:05d}",
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
