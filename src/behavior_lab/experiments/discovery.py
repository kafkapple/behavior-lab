"""Composable feature + unsupervised discovery experiment helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..core.types import ClusteringResult
from ..data.features import SkeletonBackend
from ..data.features.temporal import aggregate_temporal
from ..evaluation import compute_behavior_metrics, compute_cluster_metrics
from ..models.discovery.clustering import cluster_features


@dataclass
class DiscoveryRun:
    """One comparable unsupervised behavior-analysis result."""

    name: str
    feature_name: str
    result: ClusteringResult
    behavior_metrics: Any | None = None
    cluster_metrics: dict[str, float] | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def extract_feature_matrix(
    keypoints: np.ndarray,
    feature: str = "skeleton_kinematic",
    *,
    fps: float = 30.0,
    temporal_agg: dict[str, Any] | None = None,
) -> np.ndarray:
    """Extract a named feature matrix from canonical keypoints.

    Supported lightweight features intentionally avoid heavy optional installs:
    ``raw_keypoints`` and ``skeleton_kinematic``. Heavier representations such
    as Morlet/CEBRA/BehaveMAE can still be run through their dedicated modules.
    """
    if feature == "raw_keypoints":
        features = np.asarray(keypoints, dtype=np.float32).reshape(keypoints.shape[0], -1)
    elif feature == "skeleton_kinematic":
        features = SkeletonBackend(fps=fps, normalize_body_size=True).extract(keypoints)
    else:
        raise ValueError(f"Unknown lightweight feature '{feature}'")

    if temporal_agg:
        features = aggregate_temporal(
            features,
            window_size=int(temporal_agg.get("window_size", round(fps))),
            stride=int(temporal_agg.get("stride", round(fps / 2))),
            method=temporal_agg.get("method", "concat_stats"),
        )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def compare_discovery_methods(
    keypoints: np.ndarray,
    *,
    methods: Iterable[str] = ("kmeans_pca_umap", "bsoid"),
    feature: str = "skeleton_kinematic",
    fps: float = 30.0,
    n_clusters: int = 8,
    random_state: int = 42,
    max_frames: int | None = None,
) -> list[DiscoveryRun]:
    """Run several behavior discovery methods with comparable outputs.

    Args:
        keypoints: Canonical ``(T,K,D)`` sequence.
        methods: Names from the discovery catalog. Lightweight methods are
            always available; heavy optional packages raise clear ImportError.
        feature: Feature used for feature-matrix clusterers.
        fps: Native frame rate.
        n_clusters: Fixed cluster count for KMeans/PCA/UMAP baselines.
        random_state: Seed for stochastic baselines.
        max_frames: Optional deterministic truncation for notebook smoke runs.
    """
    kp = np.asarray(keypoints, dtype=np.float32)
    if max_frames is not None:
        kp = kp[:max_frames]

    runs: list[DiscoveryRun] = []
    for method in methods:
        key = method.lower().replace("-", "_")
        if key in {"kmeans", "kmeans_pca_umap", "clustering"}:
            feats = extract_feature_matrix(kp, feature=feature, fps=fps)
            raw = cluster_features(
                feats,
                n_clusters=n_clusters,
                use_umap=True,
                random_state=random_state,
            )
            result = ClusteringResult(
                labels=raw["labels"],
                embeddings=raw["embedding_2d"],
                n_clusters=int(raw["n_clusters"]),
                features=feats,
                metadata={"algorithm": "kmeans_pca_umap", "feature": feature},
            )
        elif key in {"bsoid", "b_soid"}:
            from ..models import get_model

            result = get_model("bsoid", fps=int(round(fps)), random_state=random_state).fit_predict(kp)
        elif key in {"moseq_fallback", "pca_hmm_fallback"}:
            from ..models.discovery.moseq import _PCAHMMFallback

            result = _PCAHMMFallback(n_states=n_clusters).fit(kp)
        else:
            from ..models import get_model

            result = get_model(key).fit_predict(kp)

        behavior_metrics = compute_behavior_metrics(result.labels, fps=fps)
        cluster_metrics = None
        if result.features is not None and len(result.labels) == result.features.shape[0]:
            try:
                cluster_metrics = compute_cluster_metrics(result.features, result.labels).__dict__
            except Exception as exc:
                cluster_metrics = {"error": str(exc)}
        runs.append(DiscoveryRun(
            name=method,
            feature_name=feature,
            result=result,
            behavior_metrics=behavior_metrics,
            cluster_metrics=cluster_metrics,
        ))
    return runs


__all__ = ["DiscoveryRun", "compare_discovery_methods", "extract_feature_matrix"]
