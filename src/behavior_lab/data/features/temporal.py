"""Temporal aggregation for per-frame features.

Converts per-frame feature vectors (T, D) into per-segment features (N_seg, D')
using sliding windows. This is essential because behavior is defined over
temporal segments, not individual frames.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

AggMethod = Literal["mean", "max", "concat_stats"]


def aggregate_temporal(
    features: np.ndarray,
    window_size: int = 30,
    stride: int = 15,
    method: AggMethod = "mean",
) -> np.ndarray:
    """Aggregate per-frame features into per-segment features via sliding window.

    At 30fps: window_size=30 (1 second), stride=15 (0.5s overlap) is a
    reasonable default for behavior segmentation.

    Args:
        features: (T, D) per-frame feature matrix
        window_size: number of frames per segment
        stride: step between consecutive windows
        method:
            mean — window average, output (N_seg, D)
            max  — window maximum, output (N_seg, D)
            concat_stats — mean + std + min + max, output (N_seg, 4*D)

    Returns:
        (N_seg, D') aggregated feature matrix

    Raises:
        ValueError: if features has wrong shape or window_size < 1
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D features (T, D), got shape {features.shape}"
        )
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    T, D = features.shape

    if T < window_size:
        # Not enough frames for a full window — return single segment
        return _aggregate_window(features, method).reshape(1, -1)

    segments: list[np.ndarray] = []
    for start in range(0, T - window_size + 1, stride):
        window = features[start : start + window_size]
        segments.append(_aggregate_window(window, method))

    return np.stack(segments, axis=0)


def _aggregate_window(window: np.ndarray, method: AggMethod) -> np.ndarray:
    """Aggregate a single (W, D) window into a (D',) vector."""
    if method == "mean":
        return window.mean(axis=0)
    elif method == "max":
        return window.max(axis=0)
    elif method == "concat_stats":
        return np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def get_output_dim(input_dim: int, method: AggMethod) -> int:
    """Compute output dimension for a given method and input dim."""
    if method in ("mean", "max"):
        return input_dim
    elif method == "concat_stats":
        return input_dim * 4
    raise ValueError(f"Unknown method: {method}")
