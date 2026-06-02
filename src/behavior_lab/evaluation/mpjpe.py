"""MPJPE metrics for 3D keypoint benchmark.

Provides:
  - mpjpe: Mean Per-Joint Position Error
  - root_relative_mpjpe: subtract root joint before error (kills global offset)
  - pmpjpe: Procrustes-aligned MPJPE (kills global similarity)
  - bootstrap_ci: percentile bootstrap confidence interval

Used by scripts/benchmark_kp_dlc.py.
"""
from __future__ import annotations

import numpy as np


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame mean per-joint position error.

    Parameters
    ----------
    pred, gt : (N, K, 3) float

    Returns
    -------
    (N,) per-frame mean L2 distance across K joints.
    """
    _check_shapes(pred, gt)
    err = np.linalg.norm(pred - gt, axis=-1)        # (N, K)
    return err.mean(axis=-1)                         # (N,)


def root_relative_mpjpe(
    pred: np.ndarray, gt: np.ndarray, root_idx: int = 0
) -> np.ndarray:
    """MPJPE after subtracting root joint (eliminates global translation)."""
    _check_shapes(pred, gt)
    pred_rel = pred - pred[:, root_idx : root_idx + 1, :]
    gt_rel = gt - gt[:, root_idx : root_idx + 1, :]
    return mpjpe(pred_rel, gt_rel)


def pmpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes-aligned MPJPE (per-frame rigid + scale alignment).

    Allows reflection. Use when global rotation/scale is uninteresting.
    """
    _check_shapes(pred, gt)
    n_frames = pred.shape[0]
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        aligned = _procrustes_align(pred[i], gt[i])
        out[i] = np.linalg.norm(aligned - gt[i], axis=-1).mean()
    return out


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over per-frame values.

    Returns
    -------
    (mean, lower, upper) — mean and (1-ci)/2 / 1-(1-ci)/2 percentiles.
    """
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return float(values.mean()), lower, upper


def _check_shapes(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"expected (N, K, 3), got {pred.shape}")


def _procrustes_align(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Procrustes alignment src → tgt with scale + rotation (reflection allowed).

    Returns aligned src.
    """
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    s = src - mu_src
    t = tgt - mu_tgt
    norm_s = np.linalg.norm(s)
    if norm_s < 1e-12:
        return tgt.copy()
    M = t.T @ s
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.trace(U @ Vt @ s.T @ t) / (norm_s ** 2 + 1e-12)
    return scale * (s @ R.T) + mu_tgt
