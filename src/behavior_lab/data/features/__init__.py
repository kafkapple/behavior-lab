"""Feature extraction backends for behavior analysis.

Provides a unified FeatureBackend protocol and concrete implementations:
- SkeletonBackend: kinematic features from keypoints (velocity, spread, etc.)
- DINOv2Backend: dense visual features from video frames (requires torch)
- FeaturePipeline: compose multiple backends into a single feature matrix
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .features import FeatureExtractor, extract_features
from .pipeline import FeaturePipeline
from .temporal import aggregate_temporal


@runtime_checkable
class FeatureBackend(Protocol):
    """Any feature extractor: skeleton, visual, dense 3D, etc.

    All backends must produce an (N, D) feature matrix from arbitrary input.
    """

    name: str
    dim: int

    def extract(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        """Transform raw input into a feature matrix.

        Args:
            data: Input array — shape depends on backend
                  (e.g., (T, K, D) keypoints, (N, H, W, 3) frames)

        Returns:
            (N, self.dim) feature matrix
        """
        ...


class SkeletonBackend:
    """Wrap kinematic feature extraction as a FeatureBackend.

    Extracts velocity, acceleration, body_spread, spatial_variance → (T, 4).
    """

    name = "skeleton_kinematic"
    dim = 4

    def __init__(self, fps: float = 30.0, smooth_window: int = 5,
                 normalize_body_size: bool = False):
        self._extractor = FeatureExtractor(
            fps=fps, smooth_window=smooth_window,
            normalize_body_size=normalize_body_size,
        )

    def extract(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        """(T, K, D) keypoints → (T, 4) kinematic features."""
        joint_names = kwargs.get("joint_names")
        features = self._extractor(data, joint_names=joint_names)
        return features["feature_matrix"]


def _lazy_cebra():
    from .cebra_backend import CEBRABackend
    return CEBRABackend


__all__ = [
    "FeatureBackend",
    "SkeletonBackend",
    "FeatureExtractor",
    "FeaturePipeline",
    "aggregate_temporal",
    "extract_features",
]
