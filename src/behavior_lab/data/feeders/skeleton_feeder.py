"""Unified skeleton feeder replacing NTU/UCLA/MARS-specific feeders."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Tuple

from ..preprocessing.augmentation import valid_crop_resize, random_rot


class SkeletonFeeder(Dataset):
    """Config-driven skeleton dataset. Input format: (N, C, T, V, M).
    
    Replaces NTUFeeder, UCLAFeeder, MARSFeeder with a single class.
    Data loading logic is driven by skeleton config (num_joints, num_channels, etc.).
    
    Args:
        data_path: path to .npz file
        split: 'train' or 'test'
        skeleton: skeleton name (for metadata) or None
        window_size: target temporal length
        p_interval: crop ratio interval
        random_rot: apply rotation augmentation
        use_velocity: use velocity instead of position
        debug: limit to first 100 samples
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        skeleton: Optional[str] = None,
        window_size: int = 64,
        p_interval: List[float] = None,
        random_rot: bool = False,
        use_velocity: bool = False,
        debug: bool = False,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.skeleton = skeleton
        self.window_size = window_size
        self.p_interval = p_interval or [0.5, 1.0]
        self.random_rot_flag = random_rot
        self.use_velocity = use_velocity
        self.debug = debug

        self.data: np.ndarray = None  # (N, C, T, V, M)
        self.label: np.ndarray = None  # (N,)
        self._load_data()

    def _load_data(self):
        """Load and normalize NPZ data to (N, C, T, V, M) format."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")

        npz = np.load(str(self.data_path))

        # Extract split
        if f'x_{self.split}' in npz:
            data = npz[f'x_{self.split}']
            labels = npz[f'y_{self.split}']
        elif 'sequences' in npz:
            data = npz['sequences']
            labels = npz['labels']
        else:
            raise ValueError(f"Unknown NPZ format. Keys: {list(npz.keys())}")

        # Convert one-hot labels
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=-1)

        # Reshape to (N, C, T, V, M) if needed
        if data.ndim == 3:
            # (N, T, features) -> need to infer C, V, M from skeleton
            data = self._reshape_flat(data)

        # Filter NaN
        valid = ~np.isnan(data.reshape(len(data), -1).mean(axis=-1))
        self.data = data[valid]
        self.label = labels[valid]

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]

    def _reshape_flat(self, data: np.ndarray) -> np.ndarray:
        """Reshape (N, T, flat_features) to (N, C, T, V, M).
        
        Auto-detects shape from skeleton config or common patterns.
        """
        N, T, F = data.shape

        if self.skeleton:
            from behavior_lab.core.skeleton import get_skeleton
            skel = get_skeleton(self.skeleton)
            C = skel.num_channels
            V = skel.num_joints
            M = skel.num_persons

            if F == M * V * C:
                return data.reshape(N, T, M, V, C).transpose(0, 4, 1, 3, 2)
            elif F == V * C:
                return data.reshape(N, T, 1, V, C).transpose(0, 4, 1, 3, 2)

        # NTU pattern: (N, T, 150) -> (N, 3, T, 25, 2)
        if F == 150:
            return data.reshape(N, T, 2, 25, 3).transpose(0, 4, 1, 3, 2)
        # UCLA pattern: (N, T, 60) -> (N, 3, T, 20, 1)
        if F == 60:
            return data.reshape(N, T, 1, 20, 3).transpose(0, 4, 1, 3, 2)

        raise ValueError(f"Cannot reshape (N={N}, T={T}, F={F}). Provide skeleton config.")

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        data = np.array(self.data[index])  # (C, T, V, M)
        label = int(self.label[index])

        # Count valid frames
        valid_frames = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        if valid_frames == 0:
            valid_frames = data.shape[1]

        # Crop and resize
        data = valid_crop_resize(data, valid_frames, self.p_interval, self.window_size)

        # Augmentation
        if self.random_rot_flag and self.split == 'train':
            data = random_rot(data)

        # Velocity
        if self.use_velocity:
            data[:, :-1] = data[:, 1:] - data[:, :-1]
            data[:, -1] = 0

        return data.astype(np.float32), label, index

    def top_k(self, score: np.ndarray, top_k: int) -> float:
        """Calculate top-k accuracy."""
        rank = score.argsort()
        hit = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit) / len(hit)


def get_feeder(dataset_type: str = None, **kwargs) -> SkeletonFeeder:
    """Factory: create feeder with dataset-specific defaults."""
    defaults = {
        'ntu': {'skeleton': 'ntu', 'window_size': 64, 'random_rot': True},
        'ucla': {'skeleton': 'ucla', 'window_size': 52, 'random_rot': True},
        'mars': {'skeleton': 'mars', 'window_size': 64, 'random_rot': False},
        'calms21': {'skeleton': 'calms21', 'window_size': 64, 'random_rot': False},
    }

    if dataset_type and dataset_type.lower() in defaults:
        cfg = defaults[dataset_type.lower()]
        cfg.update(kwargs)
        return SkeletonFeeder(**cfg)

    return SkeletonFeeder(**kwargs)
