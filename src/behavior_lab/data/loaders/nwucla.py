"""NW-UCLA dataset loader — Northwestern-UCLA Action Recognition.

Loads NW-UCLA skeleton data: 10 actions, 20 joints × 3D.
Reference: Wang et al. (2014), "Cross-view Action Modeling, Learning and Recognition".

Data format:
    - .npz: x_{split} shape (N, 300, 60), y_{split} one-hot (N, 10)
    - F=60 = 20 joints × 3 channels (xyz)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.types import BehaviorSequence
from ...core.skeleton import get_skeleton


UCLA_CLASSES = [
    "pick up with one hand", "pick up with two hands",
    "drop trash", "walk around", "sit down",
    "stand up", "donning", "doffing",
    "throw", "carry",
]


class NWUCLALoader:
    """Loader for NW-UCLA multiview action recognition dataset.

    Expects:
        nwucla/
        └── nwucla_aligned.npz  # x_train (N, T=300, F=60), y_train (N, 10) one-hot
    """

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "nw_ucla",
        fps: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.skeleton = get_skeleton(skeleton_name)

    def load_split(self, split: str = "train") -> list[BehaviorSequence]:
        """Load a data split from NPZ.

        Args:
            split: 'train' or 'test'

        Returns:
            List of BehaviorSequence
        """
        for npz_path in sorted(self.data_dir.glob("*.npz")):
            npz = np.load(npz_path, allow_pickle=True)
            x_key = f"x_{split}"
            y_key = f"y_{split}"
            if x_key in npz:
                return self._parse_npz(npz[x_key], npz.get(y_key), split)

        raise FileNotFoundError(
            f"No NPZ with x_{split} key found in {self.data_dir}"
        )

    def _parse_npz(
        self, data: np.ndarray, labels: np.ndarray | None, split: str
    ) -> list[BehaviorSequence]:
        """Parse NPZ arrays: x (N, T, F), y (N, C) one-hot."""
        K = self.skeleton.num_joints   # 20
        D = self.skeleton.num_channels  # 3

        N, T, F = data.shape
        assert F == K * D, f"Feature dim {F} != {K}*{D}"

        data_tkd = data.reshape(N, T, K, D)

        int_labels = None
        if labels is not None:
            if labels.ndim == 2 and labels.shape[1] > 1:
                int_labels = labels.argmax(axis=1)
            else:
                int_labels = labels.flatten().astype(int)

        sequences = []
        for i in range(N):
            kp = data_tkd[i]
            label = int(int_labels[i]) if int_labels is not None else None
            per_frame = np.full(T, label, dtype=np.int64) if label is not None else None

            sequences.append(BehaviorSequence(
                keypoints=kp.astype(np.float32),
                labels=per_frame,
                skeleton_name=self.skeleton_name,
                sample_id=f"{split}_{i:05d}",
                fps=self.fps,
                metadata={
                    "dataset": "nwucla",
                    "split": split,
                    "class_names": UCLA_CLASSES,
                    "action_label": label,
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
        return UCLA_CLASSES
