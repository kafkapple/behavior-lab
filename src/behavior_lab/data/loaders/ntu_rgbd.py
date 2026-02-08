"""NTU RGB+D dataset loader — Human action recognition.

Loads NTU RGB+D 60/120 skeleton data in multiple formats.
Reference: Shahroudy et al. (2016), NTU RGB+D; Liu et al. (2019), NTU RGB+D 120.

Data formats:
    - .skeleton: Raw NTU text format
    - .npy: Pre-extracted (T, V*C*M) or (C, T, V, M)
    - .pkl: Pickle dict with 'data' and 'label'
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.types import BehaviorSequence
from ...core.skeleton import get_skeleton


# NTU action labels (60 classes for NTU-60)
NTU60_CLASSES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop",
    "pick up", "throw", "sit down", "stand up", "clapping",
    "reading", "writing", "tear up paper", "put on jacket", "take off jacket",
    "put on a shoe", "take off a shoe", "put on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
    "kicking something", "reach into pocket", "hopping", "jump up",
    "phone call", "play with phone/tablet", "type on a keyboard",
    "point to something", "taking a selfie", "check time (from watch)",
    "rub two hands", "nod head/bow", "shake head", "wipe face", "salute",
    "put palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling down", "headache", "chest pain",
    "back pain", "neck pain", "nausea/vomit", "fan self",
    "punch/slap", "kicking", "pushing", "pat on back", "point finger",
    "hugging", "giving object", "touch pocket", "shaking hands",
    "walking towards", "walking apart",
]


class NTURGBDLoader:
    """Loader for NTU RGB+D 60/120 skeleton dataset.

    Supports three input formats:
        1. .skeleton files (raw NTU text format)
        2. .npy files (pre-processed numpy)
        3. .pkl files (pickle dict)

    Expects directory structure:
        ntu_rgbd/
        ├── nturgb+d_skeletons/     # .skeleton files
        │   ├── S001C001P001R001A001.skeleton
        │   └── ...
        ├── ntu60_xsub_train.npz    # or .pkl, pre-processed
        └── ntu60_xsub_test.npz
    """

    def __init__(
        self,
        data_dir: str | Path,
        skeleton_name: str = "ntu",
        fps: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.skeleton_name = skeleton_name
        self.fps = fps
        self.skeleton = get_skeleton(skeleton_name)

    def load_npz(
        self, filepath: str | Path, split: str | None = None
    ) -> list[BehaviorSequence]:
        """Load pre-processed NPZ file.

        Supports two key layouts:
            - 'data'/'label': standard format (N, C, T, V, M) or (N, T, F)
            - 'x_train'/'y_train'/'x_test'/'y_test': split-based format

        Args:
            filepath: Path to NPZ file
            split: If set ('train'/'test'), load only that split from x_/y_ keys
        """
        filepath = Path(filepath)
        npz = np.load(filepath, allow_pickle=True)

        # Determine which keys to use
        if split and f"x_{split}" in npz:
            data = npz[f"x_{split}"]
            labels = npz.get(f"y_{split}", None)
        elif "data" in npz:
            data = npz["data"]
            labels = npz.get("label", None)
        elif "x_train" in npz:
            data = npz["x_train"]
            labels = npz.get("y_train", None)
        else:
            raise ValueError(f"NPZ has no recognized keys: {npz.files[:10]}")

        if labels is not None:
            # Handle one-hot encoded labels: (N, C) -> argmax
            if labels.ndim == 2 and labels.shape[1] > 1:
                labels = labels.argmax(axis=1)
            else:
                labels = labels.flatten()

        sequences = []
        for i in range(data.shape[0]):
            sample = data[i]
            keypoints = self._to_tkd(sample)

            label = int(labels[i]) if labels is not None else None

            sequences.append(BehaviorSequence(
                keypoints=keypoints.astype(np.float32),
                labels=np.full(keypoints.shape[0], label) if label is not None else None,
                skeleton_name=self.skeleton_name,
                sample_id=f"{filepath.stem}_{split or ''}_{i:05d}",
                fps=self.fps,
                metadata={
                    "dataset": "ntu_rgbd",
                    "action_label": label,
                    "source_file": str(filepath),
                },
            ))

        return sequences

    def load_skeleton_file(self, filepath: str | Path) -> BehaviorSequence:
        """Load a single .skeleton text file.

        NTU skeleton file format:
            Line 1: number of frames
            For each frame:
                Line: number of bodies
                For each body:
                    Line: body info (10 values)
                    Line: number of joints (25)
                    For each joint:
                        Line: x y z dx dy dz ox oy oz ow (10 float values)
        """
        filepath = Path(filepath)
        sample_id = filepath.stem  # e.g. S001C001P001R001A001
        action_id = int(sample_id[-3:]) - 1  # A001 -> 0

        with open(filepath, "r") as f:
            lines = f.readlines()

        idx = 0
        num_frames = int(lines[idx])
        idx += 1

        all_keypoints = []  # will be (T, M, V, C)
        max_bodies = self.skeleton.num_persons

        for _ in range(num_frames):
            num_bodies = int(lines[idx])
            idx += 1

            frame_bodies = []
            for _ in range(num_bodies):
                idx += 1  # skip body info line
                num_joints = int(lines[idx])
                idx += 1

                joints = []
                for _ in range(num_joints):
                    vals = lines[idx].split()
                    x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                    joints.append([x, y, z])
                    idx += 1
                frame_bodies.append(joints)

            # Pad to max_bodies
            while len(frame_bodies) < max_bodies:
                frame_bodies.append([[0.0, 0.0, 0.0]] * self.skeleton.num_joints)

            all_keypoints.append(frame_bodies[:max_bodies])

        # Shape: (T, M, V, C) -> (T, M*V, C)
        kp = np.array(all_keypoints, dtype=np.float32)
        T, M, V, C = kp.shape
        keypoints = kp.reshape(T, M * V, C)

        return BehaviorSequence(
            keypoints=keypoints,
            labels=np.full(T, action_id),
            skeleton_name=self.skeleton_name,
            sample_id=sample_id,
            fps=self.fps,
            metadata={
                "dataset": "ntu_rgbd",
                "action_label": action_id,
                "source_file": str(filepath),
            },
        )

    def _to_tkd(self, sample: np.ndarray) -> np.ndarray:
        """Convert various NTU formats to (T, K, D).

        Handles:
            - (C, T, V, M): standard graph format
            - (T, F): flat format where F = V * C * M
        """
        V = self.skeleton.num_joints  # 25
        C = self.skeleton.num_channels  # 3
        M = self.skeleton.num_persons  # 2

        if sample.ndim == 4:
            # (C, T, V, M) -> (T, M*V, C)
            T = sample.shape[1]
            return sample.transpose(1, 3, 2, 0).reshape(T, M * V, C)
        elif sample.ndim == 2:
            # (T, F) where F = V*C*M or V*C
            T, F = sample.shape
            if F == V * C * M:
                return sample.reshape(T, M, V, C).reshape(T, M * V, C)
            elif F == V * C:
                return sample.reshape(T, V, C)
            else:
                raise ValueError(f"Unexpected flat dim {F} for NTU (V={V}, C={C}, M={M})")
        else:
            raise ValueError(f"Unexpected shape {sample.shape}")

    def load_all_skeletons(self, subdir: str = "nturgb+d_skeletons") -> list[BehaviorSequence]:
        """Load all .skeleton files from a subdirectory."""
        skel_dir = self.data_dir / subdir
        if not skel_dir.exists():
            raise FileNotFoundError(f"Skeleton directory not found: {skel_dir}")

        sequences = []
        for fp in sorted(skel_dir.glob("*.skeleton")):
            try:
                sequences.append(self.load_skeleton_file(fp))
            except (ValueError, IndexError) as e:
                import warnings
                warnings.warn(f"Skipping {fp.name}: {e}")
        return sequences

    @staticmethod
    def class_names(num_classes: int = 60) -> list[str]:
        return NTU60_CLASSES[:num_classes]
