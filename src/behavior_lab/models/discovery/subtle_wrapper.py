"""SUBTLE wrapper: Spectrogram-UMAP-based Temporal Link Embedding.

Reference: Kwon et al. (2024), IJCV.
Install: pip install git+https://github.com/jeakwon/subtle.git

Compatibility: scipy >= 1.12 removed signal.cwt. We monkey-patch the
SUBTLE module's morlet_cwt to use a manual convolution-based CWT.
"""
import numpy as np
from typing import Dict, List, Optional

from ...core.types import ClusteringResult


def _morlet2_compat(M, s, w=5):
    """Drop-in replacement for scipy.signal.morlet2 (removed in scipy 1.15).

    Normalized complex Morlet wavelet:
        psi(t) = (pi * s)^(-0.25) * exp(i*w*t/s) * exp(-0.5*(t/s)^2)

    Args:
        M: length of the wavelet
        s: width (scale) parameter
        w: omega0 — central frequency (default 5)

    Returns:
        complex128 array of length M
    """
    t = np.arange(0, M) - (M - 1.0) / 2
    t = t / s
    output = np.exp(1j * w * t) * np.exp(-0.5 * t * t) * np.pi ** (-0.25)
    return output


def _cwt_compat(data, wavelet, widths, **kwargs):
    """Drop-in replacement for scipy.signal.cwt (removed in scipy 1.12).

    For each scale/width, generate the wavelet, convolve with data, and stack.
    """
    output = np.empty((len(widths), len(data)), dtype=np.complex128)
    for i, width in enumerate(widths):
        N = int(np.min([10 * width, len(data)]))
        N = max(N * 2 + 1, 3)
        wavelet_data = wavelet(N, width, **kwargs)
        output[i] = np.convolve(data, wavelet_data, mode="same")
    return output


def _patch_subtle_cwt():
    """Patch subtle.module.morlet_cwt for scipy >= 1.12.

    scipy 1.12+ removed signal.cwt and 1.15+ removed signal.morlet2.
    We replace both with pure-numpy implementations.
    """
    from scipy import signal

    # Patch signal.morlet2 if missing
    if not hasattr(signal, "morlet2"):
        signal.morlet2 = _morlet2_compat

    # Patch signal.cwt if missing
    if not hasattr(signal, "cwt"):
        signal.cwt = _cwt_compat

    # Patch subtle.module directly (it caches the import)
    try:
        import subtle.module as sm

        def patched_morlet_cwt(x, fs, omega, n_channels):
            f_nyquist = fs / 2
            freq = np.linspace(f_nyquist / 10, f_nyquist, n_channels)
            widths = omega * fs / (2 * freq * np.pi)
            return np.abs(_cwt_compat(x, _morlet2_compat, widths, w=omega))

        sm.morlet_cwt = patched_morlet_cwt
    except (ImportError, AttributeError):
        pass


class SUBTLE:
    """Wrapper for SUBTLE library with behavior-lab (T, K, D) format.

    SUBTLE uses Morlet wavelet spectrograms + UMAP embedding + Phenograph clustering
    to discover hierarchical behavioral repertoires (sub/superclusters).

    Usage:
        model = SUBTLE(fps=20)
        results = model.fit([seq1, seq2])  # list of (T_i, K, 3) arrays
        results = model.predict([new_seq])
    """

    def __init__(self, fps: int = 20, n_train_frames: int = 120000,
                 embedding_method: str = 'umap', **kwargs):
        self.fps = fps
        self.n_train_frames = n_train_frames
        self.embedding_method = embedding_method
        self.kwargs = kwargs
        self._mapper = None

    def _preprocess(self, seq: np.ndarray) -> np.ndarray:
        """Convert (T, K, D) -> (T, K*D) with centering."""
        T, K, D = seq.shape
        # Global mean subtraction (SUBTLE's avatar_preprocess)
        mean = seq.mean(axis=(0, 1))
        centered = seq - mean
        return centered.reshape(T, K * D)

    def fit(self, sequences: List[np.ndarray]) -> Dict:
        """Fit SUBTLE model on skeleton sequences.

        Args:
            sequences: list of (T_i, K, 3) 3D skeleton arrays

        Returns:
            dict with 'embeddings', 'subclusters', 'superclusters',
                       'transitions', 'retention'
        """
        _patch_subtle_cwt()
        try:
            import subtle
        except ImportError:
            raise ImportError(
                "Install SUBTLE: pip install git+https://github.com/jeakwon/subtle.git")

        flat = [self._preprocess(s) for s in sequences]
        self._mapper = subtle.Mapper(
            fs=self.fps, n_train_frames=self.n_train_frames,
            embedding_method=self.embedding_method, **self.kwargs)
        data_list = self._mapper.fit(flat)

        # Results live on the mapper and Data objects
        data_obj = data_list[0] if data_list else None
        return {
            'embeddings': getattr(self._mapper, 'Z', None),
            'subclusters': getattr(self._mapper, 'y', None),
            'superclusters': getattr(self._mapper, 'Y', None),
            'transitions': getattr(data_obj, 'TP', None) if data_obj else None,
            'retention': getattr(data_obj, 'R', None) if data_obj else None,
        }

    def predict(self, sequences: List[np.ndarray]) -> Dict:
        """Apply trained model to new sequences.

        Args:
            sequences: list of (T_i, K, 3) 3D skeleton arrays

        Returns:
            same structure as fit()
        """
        if self._mapper is None:
            raise RuntimeError("Call .fit() first")

        flat = [self._preprocess(s) for s in sequences]
        data_list = self._mapper.run(flat)

        data_obj = data_list[0] if data_list else None
        return {
            'embeddings': getattr(data_obj, 'Z', None) if data_obj else None,
            'subclusters': getattr(data_obj, 'y', None) if data_obj else None,
            'superclusters': getattr(data_obj, 'Y', None) if data_obj else None,
            'transitions': getattr(data_obj, 'TP', None) if data_obj else None,
            'retention': getattr(data_obj, 'R', None) if data_obj else None,
        }

    def save(self, path: str):
        """Save trained mapper."""
        if self._mapper is None:
            raise RuntimeError("No trained model to save")
        self._mapper.save(path)

    def load(self, path: str):
        """Load pre-trained mapper."""
        _patch_subtle_cwt()
        import subtle
        self._mapper = subtle.Mapper()
        self._mapper.load(path)
        return self

    def fit_predict(self, sequences: List[np.ndarray]) -> ClusteringResult:
        """Fit and return structured ClusteringResult."""
        result = self.fit(sequences)
        subclusters = result.get("subclusters")
        embeddings = result.get("embeddings")

        labels = subclusters if subclusters is not None else np.zeros(0, dtype=int)
        n_clusters = len(set(labels.flatten())) if hasattr(labels, 'flatten') else 0

        return ClusteringResult(
            labels=labels,
            embeddings=embeddings,
            n_clusters=n_clusters,
            metadata={
                "algorithm": "subtle",
                "superclusters": result.get("superclusters"),
                "transitions": result.get("transitions"),
            },
        )

    def get_embeddings(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Extract UMAP embeddings from sequences."""
        if self._mapper is None:
            result = self.fit(sequences)
        else:
            result = self.predict(sequences)
        return result.get("embeddings", np.zeros((0, 2)))
