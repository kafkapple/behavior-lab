"""Morlet CWT feature backend: standalone spectrogram extraction.

Extracts time-frequency features from keypoint sequences using
Morlet continuous wavelet transform — the same spectral analysis
used internally by SUBTLE, but exposed as an independent FeatureBackend.

This enables combining SUBTLE-style spectral features with alternative
clusterers (HDBSCAN, KMeans, etc.) without the full SUBTLE pipeline.

Reference: Kwon et al. (2024), IJCV — SUBTLE.
"""
from __future__ import annotations

import numpy as np


def _morlet2(M: int, s: float, w: float = 5.0) -> np.ndarray:
    """Normalized complex Morlet wavelet (replaces scipy.signal.morlet2)."""
    t = (np.arange(M) - (M - 1.0) / 2) / s
    return np.exp(1j * w * t) * np.exp(-0.5 * t * t) * np.pi ** (-0.25)


def _cwt(data: np.ndarray, widths: np.ndarray, w: float = 5.0) -> np.ndarray:
    """Continuous wavelet transform via convolution (replaces scipy.signal.cwt)."""
    T = len(data)
    output = np.empty((len(widths), T), dtype=np.complex128)
    for i, width in enumerate(widths):
        N = max(int(np.min([10 * width, T])) * 2 + 1, 3)
        # Cap wavelet length to data length to keep output size consistent
        N = min(N, T)
        wavelet = _morlet2(N, width, w)
        conv = np.convolve(data, wavelet, mode="same")
        output[i] = conv[:T]
    return output


def morlet_spectrogram(
    signal_1d: np.ndarray,
    fs: float,
    n_channels: int = 25,
    omega: float = 5.0,
) -> np.ndarray:
    """Compute Morlet CWT spectrogram for a 1D signal.

    Args:
        signal_1d: (T,) time series
        fs: sampling rate (Hz)
        n_channels: number of frequency bands
        omega: central frequency of Morlet wavelet

    Returns:
        (n_channels, T) power spectrogram (absolute value)
    """
    f_nyquist = fs / 2
    freq = np.linspace(f_nyquist / 10, f_nyquist, n_channels)
    widths = omega * fs / (2 * freq * np.pi)
    return np.abs(_cwt(signal_1d, widths, w=omega))


class MorletCWTBackend:
    """Morlet CWT spectrogram as a standalone FeatureBackend.

    Transforms (T, K, D) keypoints into (T, n_channels * K * D) spectral features
    by applying Morlet CWT to each coordinate channel independently.

    Args:
        fs: sampling rate (Hz)
        n_channels: number of frequency bands per coordinate
        omega: central frequency of Morlet wavelet
        center: subtract global mean before CWT
    """

    name = "morlet_cwt"

    def __init__(
        self,
        fs: float = 30.0,
        n_channels: int = 25,
        omega: float = 5.0,
        center: bool = True,
    ):
        self.fs = fs
        self.n_channels = n_channels
        self.omega = omega
        self.center = center

    @property
    def dim(self) -> int:
        # Determined after first extract() call; -1 means unknown
        return getattr(self, "_dim", -1)

    def extract(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        """(T, K, D) keypoints → (T, n_channels * K * D) spectral features.

        Also accepts (T, F) pre-flattened input.
        """
        if data.ndim == 2:
            T, F = data.shape
        elif data.ndim == 3:
            T, K, D = data.shape
            if self.center:
                data = data - data.mean(axis=(0, 1), keepdims=True)
            data = data.reshape(T, -1)
            F = data.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {data.ndim}D")

        # CWT per channel → (n_channels, T) each → stack
        spectrograms = []
        for ch in range(F):
            spec = morlet_spectrogram(data[:, ch], self.fs, self.n_channels, self.omega)
            spectrograms.append(spec)  # (n_channels, T)

        # Stack: (F * n_channels, T) → transpose to (T, F * n_channels)
        features = np.vstack(spectrograms).T  # (T, F * n_channels)
        self._dim = features.shape[1]
        return features.astype(np.float32)
