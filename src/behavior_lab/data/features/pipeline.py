"""Feature pipeline: compose multiple backends into a single feature matrix.

Usage:
    pipe = FeaturePipeline(
        backends=[SkeletonBackend(), DINOv2Backend()],
        temporal_agg={"window_size": 30, "stride": 15, "method": "mean"},
    )
    features = pipe.extract(keypoints=kp, frames=video)
    result = cluster_features(features, n_clusters=8)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .temporal import AggMethod, aggregate_temporal, get_output_dim

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Compose multiple feature backends into a single feature matrix.

    Each backend extracts features from its expected input. Results are
    horizontally concatenated. Optional temporal aggregation converts
    per-frame features to per-segment features.

    Args:
        backends: list of FeatureBackend instances
        temporal_agg: optional dict with keys {window_size, stride, method}
            for temporal aggregation. Applied after per-backend extraction.
    """

    name = "feature_pipeline"

    def __init__(
        self,
        backends: list[Any],
        temporal_agg: dict[str, Any] | None = None,
    ):
        if not backends:
            raise ValueError("At least one backend is required")
        self.backends = backends
        self.temporal_agg = temporal_agg

    @property
    def dim(self) -> int:
        raw_dim = sum(b.dim for b in self.backends)
        if self.temporal_agg:
            method: AggMethod = self.temporal_agg.get("method", "mean")
            return get_output_dim(raw_dim, method)
        return raw_dim

    def extract(self, **data_dict: np.ndarray) -> np.ndarray:
        """Run all backends and concatenate results.

        Each backend receives its expected input via keyword arguments.
        The mapping from backend name to data key:
            SkeletonBackend  → 'keypoints' (T, K, D)
            DINOv2Backend    → 'frames'    (T, H, W, 3)

        For single-backend pipelines, 'data' key is also accepted as fallback.

        Args:
            **data_dict: keyword arguments mapping data keys to arrays

        Returns:
            (N, D_total) or (N_seg, D_total) feature matrix
        """
        _KEY_MAP = {
            "skeleton_kinematic": "keypoints",
            "dinov2": "frames",
            "cebra": "data",
        }

        parts: list[np.ndarray] = []
        for backend in self.backends:
            key = _KEY_MAP.get(backend.name, "data")
            if key not in data_dict:
                # Fallback: try 'data' key
                if "data" in data_dict:
                    key = "data"
                else:
                    raise KeyError(
                        f"Backend '{backend.name}' expects key '{key}', "
                        f"got keys: {list(data_dict.keys())}"
                    )

            raw = data_dict[key]
            features = backend.extract(raw)
            logger.info(
                "Backend '%s': input %s → features %s",
                backend.name, raw.shape, features.shape,
            )
            parts.append(features)

        # Align frame counts (take minimum)
        min_frames = min(p.shape[0] for p in parts)
        parts = [p[:min_frames] for p in parts]

        combined = np.concatenate(parts, axis=1)  # (T, D_total)
        logger.info("Combined features: %s", combined.shape)

        if self.temporal_agg:
            combined = aggregate_temporal(
                combined,
                window_size=self.temporal_agg.get("window_size", 30),
                stride=self.temporal_agg.get("stride", 15),
                method=self.temporal_agg.get("method", "mean"),
            )
            logger.info("After temporal aggregation: %s", combined.shape)

        return combined
