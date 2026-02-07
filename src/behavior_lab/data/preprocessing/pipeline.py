"""Keypoint preprocessing pipeline for behavior analysis.

Modular step-based pipeline for cleaning and normalizing skeleton keypoint data.
Each step implements the PreprocessingStep protocol and operates on (T, K, D) arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PreprocessingStep(Protocol):
    """Protocol for a single preprocessing step."""

    name: str

    def __call__(self, keypoints: np.ndarray, **kwargs) -> np.ndarray:
        """Apply this step to keypoint data.

        Args:
            keypoints: (T, K, D) array

        Returns:
            Processed (T, K, D) array
        """
        ...


@dataclass
class ConfidenceFilter:
    """Replace low-confidence keypoints with NaN.

    Useful when confidence scores are available as an extra channel.
    """

    name: str = "confidence_filter"
    threshold: float = 0.3
    confidence_channel: int = -1  # index of confidence in D dimension

    def __call__(self, keypoints: np.ndarray, **kwargs) -> np.ndarray:
        """Filter keypoints below confidence threshold.

        Args:
            keypoints: (T, K, D) where D includes confidence channel
        """
        confidences = kwargs.get("confidences", None)

        if confidences is not None:
            # External confidence array: (T, K)
            mask = confidences < self.threshold
            out = keypoints.copy()
            out[mask] = np.nan
            return out

        if keypoints.shape[-1] > 2:
            conf = keypoints[..., self.confidence_channel]
            mask = conf < self.threshold
            out = keypoints.copy()
            # Zero out spatial dims where confidence is low
            spatial = [i for i in range(keypoints.shape[-1]) if i != self.confidence_channel]
            out[mask] = np.nan
            return out

        return keypoints


@dataclass
class Interpolator:
    """Interpolate NaN/missing values via linear interpolation along time axis."""

    name: str = "interpolator"
    max_gap: int = 10  # max consecutive NaN frames to interpolate

    def __call__(self, keypoints: np.ndarray, **kwargs) -> np.ndarray:
        """Linearly interpolate missing (NaN) values.

        Args:
            keypoints: (T, K, D) with possible NaN values
        """
        out = keypoints.copy()
        T, K, D = out.shape

        for k in range(K):
            for d in range(D):
                signal = out[:, k, d]
                nans = np.isnan(signal)
                if not nans.any():
                    continue
                if nans.all():
                    signal[:] = 0.0
                    continue

                # Find valid indices
                valid = np.where(~nans)[0]
                # Interpolate
                signal[nans] = np.interp(
                    np.where(nans)[0], valid, signal[valid]
                )

                # Zero out gaps larger than max_gap
                nan_runs = np.diff(np.concatenate([[0], nans.astype(int), [0]]))
                starts = np.where(nan_runs == 1)[0]
                ends = np.where(nan_runs == -1)[0]
                for s, e in zip(starts, ends):
                    if (e - s) > self.max_gap:
                        signal[s:e] = 0.0

                out[:, k, d] = signal

        return out


@dataclass
class Normalizer:
    """Normalize keypoints: center on a joint and optionally scale."""

    name: str = "normalizer"
    center_joint: int = 0
    scale: bool = True

    def __call__(self, keypoints: np.ndarray, **kwargs) -> np.ndarray:
        """Center on reference joint and optionally scale to unit bbox.

        Args:
            keypoints: (T, K, D)
        """
        out = keypoints.copy()

        # Center on reference joint
        center = out[:, self.center_joint : self.center_joint + 1, :]  # (T, 1, D)
        out = out - center

        if self.scale:
            # Scale to unit bounding box per frame
            valid = ~np.isnan(out)
            if valid.any():
                max_val = np.nanmax(np.abs(out))
                if max_val > 1e-8:
                    out = out / max_val

        return out


@dataclass
class TemporalSmoother:
    """Apply temporal smoothing via moving average or Savitzky-Golay."""

    name: str = "temporal_smoother"
    window_size: int = 5
    method: str = "moving_average"  # 'moving_average' or 'savgol'

    def __call__(self, keypoints: np.ndarray, **kwargs) -> np.ndarray:
        """Smooth keypoints temporally.

        Args:
            keypoints: (T, K, D)
        """
        if self.window_size <= 1:
            return keypoints

        out = keypoints.copy()
        T, K, D = out.shape

        if self.method == "savgol" and T > self.window_size:
            from scipy.signal import savgol_filter
            win = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
            for k in range(K):
                for d in range(D):
                    if not np.isnan(out[:, k, d]).any():
                        out[:, k, d] = savgol_filter(out[:, k, d], win, polyorder=2)
        else:
            # Moving average
            kernel = np.ones(self.window_size) / self.window_size
            for k in range(K):
                for d in range(D):
                    signal = out[:, k, d]
                    if not np.isnan(signal).any():
                        out[:, k, d] = np.convolve(signal, kernel, mode="same")

        return out


@dataclass
class OutlierRemover:
    """Remove outlier keypoints based on velocity or distance thresholds."""

    name: str = "outlier_remover"
    velocity_threshold: float = 50.0  # pixels/frame
    replace_with: str = "nan"  # 'nan' or 'previous'

    def __call__(self, keypoints: np.ndarray, **kwargs) -> np.ndarray:
        """Remove outlier keypoints exceeding velocity threshold.

        Args:
            keypoints: (T, K, D)
        """
        out = keypoints.copy()
        T, K, D = out.shape

        # Compute per-keypoint velocity
        velocity = np.linalg.norm(np.diff(out, axis=0), axis=-1)  # (T-1, K)
        outlier_mask = velocity > self.velocity_threshold  # (T-1, K)

        for t in range(outlier_mask.shape[0]):
            for k in range(K):
                if outlier_mask[t, k]:
                    if self.replace_with == "nan":
                        out[t + 1, k, :] = np.nan
                    elif self.replace_with == "previous":
                        out[t + 1, k, :] = out[t, k, :]

        return out


class PreprocessingPipeline:
    """Chain multiple preprocessing steps.

    Usage:
        pipeline = PreprocessingPipeline([
            ConfidenceFilter(threshold=0.3),
            Interpolator(max_gap=10),
            OutlierRemover(velocity_threshold=50),
            TemporalSmoother(window_size=5),
            Normalizer(center_joint=0),
        ])
        cleaned = pipeline(keypoints)
    """

    def __init__(self, steps: list[PreprocessingStep] | None = None):
        self.steps: list[PreprocessingStep] = steps or []

    def add_step(self, step: PreprocessingStep) -> "PreprocessingPipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    def __call__(self, keypoints: np.ndarray, **kwargs) -> np.ndarray:
        """Apply all steps sequentially."""
        result = keypoints
        for step in self.steps:
            result = step(result, **kwargs)
        return result

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"PreprocessingPipeline({' → '.join(step_names)})"
