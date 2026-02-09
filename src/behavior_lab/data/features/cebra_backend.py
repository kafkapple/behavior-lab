"""CEBRA feature backend: temporal contrastive learning for behavior embeddings.

CEBRA learns low-dimensional embeddings that preserve temporal structure
in behavior data. Unlike PCA/UMAP which ignore frame ordering, CEBRA
uses contrastive learning to ensure temporally nearby frames map to
similar representations.

Requires: pip install cebra

Reference:
    Schneider, Lee, Mathis (2023). "Learnable latent embeddings for joint
    behavioural and neural analysis." Nature.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CEBRABackend:
    """Temporal contrastive learning feature extractor.

    Fits a CEBRA model on time-series data and transforms it into a
    low-dimensional embedding that preserves temporal dynamics.

    Args:
        output_dim: Embedding dimension (default 32)
        max_iterations: Training iterations (default 5000)
        time_offsets: Temporal context window for contrastive loss
        batch_size: Training batch size
        device: 'cpu' or 'cuda'
        model_architecture: CEBRA model architecture
        temperature: Contrastive loss temperature
    """

    name = "cebra"

    def __init__(
        self,
        output_dim: int = 32,
        max_iterations: int = 5000,
        time_offsets: int = 10,
        batch_size: int = 512,
        device: str = "cpu",
        model_architecture: str = "offset10-model",
        temperature: float = 1.0,
    ):
        try:
            import cebra
        except ImportError:
            raise ImportError(
                "CEBRA is required. Install with: pip install cebra"
            )

        self._model = cebra.CEBRA(
            model_architecture=model_architecture,
            batch_size=batch_size,
            learning_rate=3e-4,
            temperature=temperature,
            output_dimension=output_dim,
            max_iterations=max_iterations,
            distance="cosine",
            conditional="time",
            device=device,
            verbose=False,
            time_offsets=time_offsets,
        )
        self._output_dim = output_dim
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._output_dim

    def extract(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Fit CEBRA on time-series data and return embeddings.

        Args:
            data: (T, D) time-series feature matrix.
                  If (T, K, D_coord) keypoints, will be flattened to (T, K*D).

        Returns:
            (T, output_dim) embedding matrix
        """
        if data.ndim > 2:
            T = data.shape[0]
            data = data.reshape(T, -1).astype(np.float32)
        elif data.ndim == 2:
            data = data.astype(np.float32)
        else:
            raise ValueError(f"Expected 2D+ input, got shape {data.shape}")

        # Handle NaN/Inf
        if not np.isfinite(data).all():
            logger.warning("CEBRA: replacing %d non-finite values with 0",
                          (~np.isfinite(data)).sum())
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if not self._fitted:
            logger.info("CEBRA: fitting on %s (dim=%d, iters=%d)",
                       data.shape, self._output_dim,
                       self._model.max_iterations)
            self._model.fit(data)
            self._fitted = True

        embedding = self._model.transform(data)
        logger.info("CEBRA: %s → %s", data.shape, embedding.shape)
        return embedding

    def fit_transform(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Convenience method: fit and transform in one call."""
        self._fitted = False
        return self.extract(data, **kwargs)
