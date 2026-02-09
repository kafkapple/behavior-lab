"""Unified skeleton augmentation for training and SSL."""
import numpy as np
from typing import Optional


def valid_crop_resize(data: np.ndarray, valid_frame_num: int,
                      p_interval: list, window_size: int) -> np.ndarray:
    """Crop valid frames and resize to fixed window size.
    
    Args:
        data: (C, T, V, M) skeleton data
        valid_frame_num: number of non-zero frames
        p_interval: [p] for center crop or [p_min, p_max] for random crop
        window_size: target temporal size
    """
    C, T, V, M = data.shape
    begin, end = 0, valid_frame_num

    valid_size = end - begin
    if valid_size == 0:
        return data[:, :window_size, :, :]

    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        cropped = data[:, begin + bias:end - bias, :, :]
    else:
        p = np.random.uniform(p_interval[0], p_interval[1])
        cropped_length = np.clip(int(np.floor(valid_size * p)), window_size, valid_size)
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        cropped = data[:, begin + bias:begin + bias + cropped_length, :, :]

    # Resize to window_size via linear interpolation of indices
    idx = np.linspace(0, cropped.shape[1] - 1, window_size).astype(np.int64)
    return cropped[:, idx, :, :]


def random_rot(data: np.ndarray, theta_range: float = 0.3) -> np.ndarray:
    """Random rotation augmentation. 2D: Z-axis, 3D: Y-axis.
    
    Args:
        data: (C, T, V, M) skeleton data
    """
    C = data.shape[0]
    theta = np.random.uniform(-theta_range, theta_range)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if C == 2:
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    else:
        rot = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])

    return np.einsum('ij,jkvm->ikvm', rot, data)


class SkeletonAugmentor:
    """Configurable skeleton augmentor for SSL (DINO strong/weak views).
    
    Args:
        mode: 'weak', 'medium', or 'strong' intensity
    """

    def __init__(self, mode: str = 'medium'):
        intensity = {'weak': 0.5, 'medium': 1.0, 'strong': 1.5}.get(mode, 1.0)
        self.rotation_range = 30.0 * intensity
        self.scale_range = (1 - 0.2 * intensity, 1 + 0.2 * intensity)
        self.noise_std = 0.02 * intensity

    def __call__(self, x) -> np.ndarray:
        """Apply augmentations to (C,T,V,M) or batch (N,C,T,V,M)."""
        import torch
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = np.asarray(x)

        if x_np.ndim == 5:
            result = np.stack([self._augment_single(s) for s in x_np])
        else:
            result = self._augment_single(x_np)

        if isinstance(x, torch.Tensor):
            return torch.from_numpy(result).float()
        return result

    def _augment_single(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        C = x.shape[0]

        # Rotation
        if np.random.random() < 0.5:
            angle = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            if C >= 2:
                shape = x[:2].shape
                x[:2] = (rot @ x[:2].reshape(2, -1)).reshape(shape)

        # Scale
        if np.random.random() < 0.5:
            x = x * np.random.uniform(*self.scale_range)

        # Noise
        if np.random.random() < 0.5:
            x = x + np.random.randn(*x.shape) * self.noise_std

        return x
