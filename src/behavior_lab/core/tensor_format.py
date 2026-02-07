"""
Tensor Format Bridge - Convert between canonical and graph formats.

Canonical format: (T, K, D) — all data uses this as the base representation.
  T = frames, K = keypoints, D = coordinate dims (2 or 3)

Graph format: (N, C, T, V, M) — required by GCN models (InfoGCN, ST-GCN, AGCN).
  N = batch, C = channels, T = frames, V = joints, M = persons

This module provides lossless round-trip conversion between the two formats.
"""
from __future__ import annotations

import numpy as np

from .skeleton import SkeletonDefinition


def sequence_to_graph(
    seq: np.ndarray,
    skeleton: SkeletonDefinition,
    num_persons: int | None = None,
    max_frames: int | None = None,
) -> np.ndarray:
    """Convert (T, K, D) or (T, M*K, D) sequence to (C, T, V, M) graph tensor.

    For batched input (N, T, K, D), returns (N, C, T, V, M).

    Args:
        seq: Keypoint sequence. Shapes supported:
            - (T, K, D): single person
            - (T, M*K, D): multi-person (flattened)
            - (N, T, K, D): batched single person
        skeleton: Skeleton definition for V and D validation.
        num_persons: Override number of persons (default: skeleton.num_persons).
        max_frames: If set, pad or crop temporal dimension.

    Returns:
        np.ndarray of shape (C, T, V, M) or (N, C, T, V, M) for batched input.
    """
    M = num_persons or skeleton.num_persons
    V = skeleton.num_joints
    D = skeleton.num_channels

    batched = seq.ndim == 4
    if batched:
        N, T, K_total, D_in = seq.shape
    else:
        assert seq.ndim == 3, f"Expected 3D or 4D array, got shape {seq.shape}"
        T, K_total, D_in = seq.shape
        N = None

    # Handle channel dimension mismatch (e.g., 2D data for 3D skeleton)
    if D_in < D:
        pad_shape = (N, T, K_total, D - D_in) if batched else (T, K_total, D - D_in)
        seq = np.concatenate([seq, np.zeros(pad_shape, dtype=seq.dtype)], axis=-1)
    elif D_in > D:
        # Truncate extra channels (e.g., confidence scores)
        if batched:
            seq = seq[..., :D]
        else:
            seq = seq[:, :, :D]

    # Separate persons if multi-person is flattened
    if K_total == M * V:
        if batched:
            data = seq.reshape(N, T, M, V, D)
        else:
            data = seq.reshape(T, M, V, D)
    elif K_total == V:
        if batched:
            data = seq.reshape(N, T, 1, V, D)
            if M > 1:
                pad = np.zeros((N, T, M - 1, V, D), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=2)
        else:
            data = seq.reshape(T, 1, V, D)
            if M > 1:
                pad = np.zeros((T, M - 1, V, D), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=1)
    else:
        raise ValueError(
            f"Keypoint count {K_total} doesn't match V={V} or M*V={M*V}"
        )

    # Temporal padding/cropping
    if max_frames is not None:
        if batched:
            current_T = data.shape[1]
        else:
            current_T = data.shape[0]

        if current_T < max_frames:
            if batched:
                pad_shape = (N, max_frames - current_T, M, V, D)
                data = np.concatenate([data, np.zeros(pad_shape, dtype=data.dtype)], axis=1)
            else:
                pad_shape = (max_frames - current_T, M, V, D)
                data = np.concatenate([data, np.zeros(pad_shape, dtype=data.dtype)], axis=0)
        elif current_T > max_frames:
            if batched:
                data = data[:, :max_frames]
            else:
                data = data[:max_frames]

    # Transpose to (C, T, V, M) or (N, C, T, V, M)
    if batched:
        # (N, T, M, V, D) -> (N, D, T, V, M)
        return data.transpose(0, 4, 1, 3, 2)
    else:
        # (T, M, V, D) -> (D, T, V, M)
        return data.transpose(3, 0, 2, 1)


def graph_to_sequence(
    tensor: np.ndarray,
    skeleton: SkeletonDefinition | None = None,
) -> np.ndarray:
    """Convert (C, T, V, M) or (N, C, T, V, M) graph tensor back to (T, K, D).

    Args:
        tensor: Graph format tensor.
        skeleton: Optional skeleton for validation.

    Returns:
        (T, V, C) for M=1, or (T, M*V, C) for M>1.
        For batched input: (N, T, V, C) or (N, T, M*V, C).
    """
    batched = tensor.ndim == 5

    if batched:
        # (N, C, T, V, M) -> (N, T, M, V, C)
        data = tensor.transpose(0, 2, 4, 3, 1)
        N, T, M, V, C = data.shape
        if M == 1:
            return data.reshape(N, T, V, C)
        return data.reshape(N, T, M * V, C)
    else:
        # (C, T, V, M) -> (T, M, V, C)
        data = tensor.transpose(1, 3, 2, 0)
        T, M, V, C = data.shape
        if M == 1:
            return data.reshape(T, V, C)
        return data.reshape(T, M * V, C)
