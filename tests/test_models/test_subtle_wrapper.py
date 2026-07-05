"""Tests for SUBTLE wrapper improvements (P1-P4).

Tests run without the SUBTLE library installed — they verify:
- P1: Interface consistency (fit → ClusteringResult, fit_raw → dict)
- P2: MorletCWTBackend standalone feature extraction
- P3: Subprocess isolation scaffolding
- P4: SUBTLEConfig dataclass
"""
import os

# Prevent macOS OpenMP SIGSEGV in sklearn/numpy (must be set before imports)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest

from behavior_lab.models.discovery.subtle_wrapper import SUBTLE, SUBTLEConfig
from behavior_lab.core.types import ClusteringResult
from behavior_lab.data.features.morlet_backend import (
    MorletCWTBackend, morlet_spectrogram, _morlet2, _cwt,
)


# ---------------------------------------------------------------------------
# P4: SUBTLEConfig
# ---------------------------------------------------------------------------

class TestSUBTLEConfig:
    def test_defaults(self):
        cfg = SUBTLEConfig()
        assert cfg.fps == 20
        assert cfg.n_train_frames == 120_000
        assert cfg.use_superclusters is True
        assert cfg.isolate is False
        assert cfg.timeout == 600

    def test_from_dict(self):
        cfg = SUBTLEConfig.from_dict({
            "fps": 30,
            "use_superclusters": False,
            "unknown_param": 42,
        })
        assert cfg.fps == 30
        assert cfg.use_superclusters is False
        assert cfg.extra == {"unknown_param": 42}

    def test_from_dict_empty(self):
        cfg = SUBTLEConfig.from_dict({})
        assert cfg.fps == 20  # default


# ---------------------------------------------------------------------------
# P1: SUBTLE interface
# ---------------------------------------------------------------------------

class TestSUBTLEInterface:
    def test_constructor_with_config(self):
        cfg = SUBTLEConfig(fps=30, isolate=True)
        model = SUBTLE(config=cfg)
        assert model.fps == 30
        assert model.config.isolate is True

    def test_constructor_kwargs(self):
        model = SUBTLE(fps=25, n_train_frames=50_000)
        assert model.config.fps == 25
        assert model.config.n_train_frames == 50_000

    def test_preprocess(self):
        model = SUBTLE(fps=20)
        seq = np.random.randn(100, 9, 3).astype(np.float32)
        flat = model._preprocess(seq)
        assert flat.shape == (100, 27)
        # Centering: global mean should be ~0
        assert np.abs(flat.mean()) < 0.5

    def test_to_result_superclusters(self):
        model = SUBTLE(fps=20)
        raw = {
            "embeddings": np.random.randn(100, 2),
            "subclusters": np.random.randint(0, 20, size=100),
            "superclusters": np.random.randint(0, 5, size=100),
            "transitions": None,
        }
        result = model._to_result(raw, use_superclusters=True)
        assert isinstance(result, ClusteringResult)
        assert result.labels.shape == (100,)
        assert result.n_clusters <= 5
        assert result.metadata["algorithm"] == "subtle"
        assert result.metadata["use_superclusters"] is True

    def test_to_result_subclusters(self):
        model = SUBTLE(fps=20)
        raw = {
            "embeddings": np.random.randn(100, 2),
            "subclusters": np.random.randint(0, 20, size=100),
            "superclusters": np.random.randint(0, 5, size=100),
            "transitions": None,
        }
        result = model._to_result(raw, use_superclusters=False)
        assert result.n_clusters <= 20
        assert result.metadata["use_superclusters"] is False

    def test_to_result_empty(self):
        model = SUBTLE(fps=20)
        raw = {
            "embeddings": None,
            "subclusters": None,
            "superclusters": None,
            "transitions": None,
        }
        result = model._to_result(raw)
        assert len(result.labels) == 0
        assert result.n_clusters == 0


# ---------------------------------------------------------------------------
# P2: MorletCWTBackend
# ---------------------------------------------------------------------------

class TestMorletCWTBackend:
    def test_morlet2_shape(self):
        w = _morlet2(128, 5.0)
        assert w.shape == (128,)
        assert w.dtype == np.complex128

    def test_cwt_shape(self):
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200))
        widths = np.array([2.0, 4.0, 8.0])
        out = _cwt(signal, widths)
        assert out.shape == (3, 200)

    def test_morlet_spectrogram(self):
        signal = np.random.randn(500)
        spec = morlet_spectrogram(signal, fs=30.0, n_channels=10)
        assert spec.shape == (10, 500)
        assert np.all(spec >= 0)  # absolute value

    def test_backend_3d_input(self):
        backend = MorletCWTBackend(fs=30.0, n_channels=10)
        kp = np.random.randn(200, 9, 3).astype(np.float32)
        features = backend.extract(kp)
        # (T, n_channels * K * D) = (200, 10 * 9 * 3) = (200, 270)
        assert features.shape == (200, 270)
        assert features.dtype == np.float32

    def test_backend_2d_input(self):
        backend = MorletCWTBackend(fs=30.0, n_channels=5)
        flat = np.random.randn(200, 27).astype(np.float32)
        features = backend.extract(flat)
        assert features.shape == (200, 135)  # 5 * 27

    def test_backend_dim_property(self):
        backend = MorletCWTBackend(fs=30.0, n_channels=10)
        assert backend.dim == -1  # unknown before extract
        kp = np.random.randn(100, 3, 2).astype(np.float32)
        backend.extract(kp)
        assert backend.dim == 60  # 10 * 3 * 2

    def test_backend_protocol(self):
        """MorletCWTBackend satisfies FeatureBackend protocol."""
        from behavior_lab.data.features import FeatureBackend
        backend = MorletCWTBackend(fs=30.0)
        assert hasattr(backend, "name")
        assert hasattr(backend, "dim")
        assert hasattr(backend, "extract")


# ---------------------------------------------------------------------------
# P3: Isolation config
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_config_isolate_default_false(self):
        model = SUBTLE(fps=20)
        assert model.config.isolate is False

    def test_config_isolate_override(self):
        cfg = SUBTLEConfig(isolate=True, timeout=300)
        model = SUBTLE(config=cfg)
        assert model.config.isolate is True
        assert model.config.timeout == 300


# ---------------------------------------------------------------------------
# Integration: MorletCWTBackend + sklearn clustering
# ---------------------------------------------------------------------------

class TestMorletWithClustering:
    """Verify that MorletCWTBackend output works with standard clusterers."""

    def test_with_kmeans(self):
        from sklearn.cluster import KMeans

        backend = MorletCWTBackend(fs=30.0, n_channels=5)
        # Two distinct behaviors: static vs moving
        static = np.zeros((100, 5, 3), dtype=np.float32)
        moving = np.random.randn(100, 5, 3).astype(np.float32) * 5
        kp = np.concatenate([static, moving], axis=0)

        features = backend.extract(kp)
        assert features.shape[0] == 200

        km = KMeans(n_clusters=2, random_state=42, n_init=5)
        labels = km.fit_predict(features)
        assert len(set(labels)) == 2
        # Labels should roughly separate static (0-99) from moving (100-199)
        label_static = labels[:100]
        label_moving = labels[100:]
        majority_static = np.bincount(label_static).argmax()
        majority_moving = np.bincount(label_moving).argmax()
        assert majority_static != majority_moving, \
            "KMeans should separate static from moving behavior"

    def test_with_evaluator(self):
        """MorletCWTBackend features work with behavior_lab Evaluator."""
        from behavior_lab.evaluation.evaluator import (
            compute_cluster_metrics, compute_behavior_metrics,
        )
        backend = MorletCWTBackend(fs=30.0, n_channels=5)
        kp = np.random.randn(200, 5, 3).astype(np.float32)
        features = backend.extract(kp)

        # Fake labels
        labels = np.repeat([0, 1, 2, 3], 50)
        cm = compute_cluster_metrics(features, labels)
        assert cm.num_clusters == 4
        assert -1 <= cm.silhouette <= 1

        bm = compute_behavior_metrics(labels, fps=30.0)
        assert bm.num_bouts == 4
        assert bm.temporal_consistency > 0.9  # 4 contiguous blocks
