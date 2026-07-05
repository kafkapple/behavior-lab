"""SUBTLE wrapper: Spectrogram-UMAP-based Temporal Link Embedding.

Reference: Kwon et al. (2024), IJCV.
Install: pip install git+https://github.com/jeakwon/subtle.git

Compatibility: scipy >= 1.12 removed signal.cwt. We monkey-patch the
SUBTLE module's morlet_cwt to use a manual convolution-based CWT.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ...core.types import ClusteringResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P4: Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SUBTLEConfig:
    """Configuration for SUBTLE model.

    Args:
        fps: Sampling rate for spectral analysis (Hz)
        n_train_frames: Cap on training data size
        embedding_method: Dimensionality reduction method
        use_superclusters: Use coarser superclusters (recommended)
        isolate: Run in subprocess to prevent macOS SIGSEGV
        timeout: Subprocess timeout in seconds (only when isolate=True)
    """
    fps: int = 20
    n_train_frames: int = 120_000
    embedding_method: str = "umap"
    use_superclusters: bool = True
    isolate: bool = False
    timeout: int = 600
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> SUBTLEConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**{k: v for k, v in d.items() if k in known}, extra=extra)


# ---------------------------------------------------------------------------
# Compatibility patches (scipy >= 1.12)
# ---------------------------------------------------------------------------

def _morlet2_compat(M, s, w=5):
    """Drop-in replacement for scipy.signal.morlet2 (removed in scipy 1.15).

    Normalized complex Morlet wavelet:
        psi(t) = (pi * s)^(-0.25) * exp(i*w*t/s) * exp(-0.5*(t/s)^2)
    """
    t = np.arange(0, M) - (M - 1.0) / 2
    t = t / s
    return np.exp(1j * w * t) * np.exp(-0.5 * t * t) * np.pi ** (-0.25)


def _cwt_compat(data, wavelet, widths, **kwargs):
    """Drop-in replacement for scipy.signal.cwt (removed in scipy 1.12)."""
    output = np.empty((len(widths), len(data)), dtype=np.complex128)
    for i, width in enumerate(widths):
        N = int(np.min([10 * width, len(data)]))
        N = max(N * 2 + 1, 3)
        wavelet_data = wavelet(N, width, **kwargs)
        output[i] = np.convolve(data, wavelet_data, mode="same")
    return output


def _patch_subtle_cwt():
    """Patch subtle.module.morlet_cwt for scipy >= 1.12."""
    from scipy import signal

    if not hasattr(signal, "morlet2"):
        signal.morlet2 = _morlet2_compat
    if not hasattr(signal, "cwt"):
        signal.cwt = _cwt_compat

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


def _patch_phenograph_njobs():
    """Force phenograph.cluster to use n_jobs=1 (macOS SIGSEGV fix)."""
    try:
        import phenograph
        _orig = phenograph.cluster
        if getattr(_orig, "_patched_njobs", False):
            return

        def _safe_cluster(*args, **kwargs):
            kwargs["n_jobs"] = 1
            return _orig(*args, **kwargs)

        _safe_cluster._patched_njobs = True
        phenograph.cluster = _safe_cluster
    except ImportError:
        pass


def _patch_umap_njobs():
    """Force UMAP to use n_jobs=1 (macOS SIGSEGV fix)."""
    try:
        import umap
        _orig_init = umap.UMAP.__init__
        if getattr(_orig_init, "_patched_njobs", False):
            return

        def _safe_init(self, *args, **kwargs):
            kwargs.setdefault("n_jobs", 1)
            return _orig_init(self, *args, **kwargs)

        _safe_init._patched_njobs = True
        umap.UMAP.__init__ = _safe_init
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# SUBTLE model
# ---------------------------------------------------------------------------

class SUBTLE:
    """Wrapper for SUBTLE library with behavior-lab (T, K, D) format.

    SUBTLE uses Morlet wavelet spectrograms + UMAP embedding + Phenograph
    clustering to discover hierarchical behavioral repertoires.

    Usage:
        model = SUBTLE(fps=20)
        result = model.fit([seq1, seq2])           # → ClusteringResult
        raw = model.fit_raw([seq1, seq2])           # → dict (low-level)
        result = model.fit_predict([seq1], isolate=True)  # subprocess-safe
    """

    def __init__(self, fps: int = 20, n_train_frames: int = 120_000,
                 embedding_method: str = "umap", config: Optional[SUBTLEConfig] = None,
                 **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = SUBTLEConfig(
                fps=fps, n_train_frames=n_train_frames,
                embedding_method=embedding_method, extra=kwargs,
            )
        self._mapper = None

    @property
    def fps(self) -> int:
        return self.config.fps

    def _preprocess(self, seq: np.ndarray) -> np.ndarray:
        """Convert (T, K, D) -> (T, K*D) with centering."""
        T, K, D = seq.shape
        mean = seq.mean(axis=(0, 1))
        centered = seq - mean
        return centered.reshape(T, K * D)

    # ------------------------------------------------------------------
    # P1: fit() returns ClusteringResult directly
    # ------------------------------------------------------------------

    def fit(self, sequences: List[np.ndarray],
            use_superclusters: Optional[bool] = None) -> ClusteringResult:
        """Fit SUBTLE model and return structured ClusteringResult.

        Args:
            sequences: list of (T_i, K, 3) 3D skeleton arrays
            use_superclusters: override config.use_superclusters
        """
        raw = self.fit_raw(sequences)
        return self._to_result(raw, use_superclusters)

    def fit_raw(self, sequences: List[np.ndarray]) -> Dict:
        """Fit SUBTLE model and return raw dict (low-level access).

        Returns:
            dict with 'embeddings', 'subclusters', 'superclusters',
                       'transitions', 'retention'
        """
        _patch_subtle_cwt()
        _patch_umap_njobs()
        _patch_phenograph_njobs()
        try:
            import subtle
        except ImportError:
            raise ImportError(
                "Install SUBTLE: pip install git+https://github.com/jeakwon/subtle.git")

        flat = [self._preprocess(s) for s in sequences]
        self._mapper = subtle.Mapper(
            fs=self.config.fps, n_train_frames=self.config.n_train_frames,
            embedding_method=self.config.embedding_method, **self.config.extra)
        data_list = self._mapper.fit(flat)

        data_obj = data_list[0] if data_list else None
        return {
            "embeddings": getattr(self._mapper, "Z", None),
            "subclusters": getattr(self._mapper, "y", None),
            "superclusters": getattr(self._mapper, "Y", None),
            "transitions": getattr(data_obj, "TP", None) if data_obj else None,
            "retention": getattr(data_obj, "R", None) if data_obj else None,
        }

    def predict(self, sequences: List[np.ndarray],
                use_superclusters: Optional[bool] = None) -> ClusteringResult:
        """Apply trained model to new sequences."""
        raw = self.predict_raw(sequences)
        return self._to_result(raw, use_superclusters)

    def predict_raw(self, sequences: List[np.ndarray]) -> Dict:
        """Apply trained model to new sequences (raw dict)."""
        if self._mapper is None:
            raise RuntimeError("Call .fit() first")
        flat = [self._preprocess(s) for s in sequences]
        data_list = self._mapper.run(flat)
        data_obj = data_list[0] if data_list else None
        return {
            "embeddings": getattr(data_obj, "Z", None) if data_obj else None,
            "subclusters": getattr(data_obj, "y", None) if data_obj else None,
            "superclusters": getattr(data_obj, "Y", None) if data_obj else None,
            "transitions": getattr(data_obj, "TP", None) if data_obj else None,
            "retention": getattr(data_obj, "R", None) if data_obj else None,
        }

    # ------------------------------------------------------------------
    # P3: Subprocess isolation
    # ------------------------------------------------------------------

    def fit_predict(self, sequences: List[np.ndarray],
                    use_superclusters: Optional[bool] = None,
                    isolate: Optional[bool] = None) -> ClusteringResult:
        """Fit and return ClusteringResult, optionally in isolated subprocess.

        Args:
            sequences: list of (T_i, K, 3) arrays
            use_superclusters: override config setting
            isolate: if True, run in subprocess (prevents macOS SIGSEGV).
                     Defaults to config.isolate.
        """
        if isolate is None:
            isolate = self.config.isolate

        if isolate:
            return self._run_isolated(sequences, use_superclusters)
        return self.fit(sequences, use_superclusters)

    def _run_isolated(self, sequences: List[np.ndarray],
                      use_superclusters: Optional[bool] = None) -> ClusteringResult:
        """Run SUBTLE in an isolated subprocess to prevent SIGSEGV."""
        use_super = use_superclusters if use_superclusters is not None \
            else self.config.use_superclusters

        # Concatenate sequences for transfer
        kp = np.concatenate(sequences, axis=0) if len(sequences) > 1 else sequences[0]

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_in = f.name
            np.savez(f, keypoints=kp)
        tmp_out = tmp_in.replace(".npz", "_result.npz")

        script = f"""
import sys, json, time
sys.path.insert(0, 'src')
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from behavior_lab.models.discovery.subtle_wrapper import SUBTLE, SUBTLEConfig

kp = np.load('{tmp_in}')['keypoints']
config = SUBTLEConfig(fps={self.config.fps},
                      n_train_frames={self.config.n_train_frames},
                      use_superclusters={use_super},
                      isolate=False)
model = SUBTLE(config=config)
t0 = time.time()
cr = model.fit([kp], use_superclusters={use_super})
elapsed = time.time() - t0

np.savez('{tmp_out}',
    labels=cr.labels,
    embeddings=cr.embeddings if cr.embeddings is not None else np.array([]),
    n_clusters=np.array(cr.n_clusters),
    elapsed=np.array(elapsed),
    subclusters=cr.metadata.get('subclusters', np.array([])),
    superclusters=cr.metadata.get('superclusters', np.array([])),
)
print(json.dumps({{'n_clusters': int(cr.n_clusters), 'elapsed': elapsed}}))
"""
        env = {
            **os.environ,
            "LOKY_MAX_CPU_COUNT": "1",
            "OMP_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True,
                timeout=self.config.timeout, env=env,
            )
            if proc.returncode != 0:
                stderr_short = (proc.stderr or "unknown")[-500:]
                raise RuntimeError(
                    f"SUBTLE subprocess exit {proc.returncode}: {stderr_short}")

            data = np.load(tmp_out)
            labels = data["labels"]
            emb = data["embeddings"]
            if emb.size == 0:
                emb = None
            subclusters = data["subclusters"] if data["subclusters"].size > 0 else None
            superclusters = data["superclusters"] if data["superclusters"].size > 0 else None

            return ClusteringResult(
                labels=labels,
                embeddings=emb,
                n_clusters=int(data["n_clusters"]),
                metadata={
                    "algorithm": "subtle",
                    "use_superclusters": use_super,
                    "subclusters": subclusters,
                    "superclusters": superclusters,
                    "isolated": True,
                },
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SUBTLE subprocess timed out ({self.config.timeout}s)")
        finally:
            Path(tmp_in).unlink(missing_ok=True)
            Path(tmp_out).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_result(self, raw: Dict,
                   use_superclusters: Optional[bool] = None) -> ClusteringResult:
        """Convert raw dict to ClusteringResult."""
        use_super = use_superclusters if use_superclusters is not None \
            else self.config.use_superclusters

        subclusters = raw.get("subclusters")
        superclusters = raw.get("superclusters")
        embeddings = raw.get("embeddings")

        if use_super and superclusters is not None:
            labels = superclusters
        elif subclusters is not None:
            labels = subclusters
        else:
            labels = np.zeros(0, dtype=int)

        if hasattr(labels, "flatten") and labels.ndim > 1:
            labels = labels.flatten()

        n_clusters = len(set(labels)) if len(labels) > 0 else 0

        return ClusteringResult(
            labels=labels,
            embeddings=embeddings,
            n_clusters=n_clusters,
            metadata={
                "algorithm": "subtle",
                "use_superclusters": use_super,
                "subclusters": subclusters,
                "superclusters": superclusters,
                "transitions": raw.get("transitions"),
            },
        )

    def get_embeddings(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Extract UMAP embeddings from sequences."""
        if self._mapper is None:
            result = self.fit_raw(sequences)
        else:
            result = self.predict_raw(sequences)
        return result.get("embeddings", np.zeros((0, 2)))

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
