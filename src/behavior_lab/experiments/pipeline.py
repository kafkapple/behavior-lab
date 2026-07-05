"""Unified, config-driven comparison experiment — one path for ANY species.

The rest of the platform is already species-agnostic (``BehaviorSequence`` (T,K,D)
+ a multi-species skeleton registry with mouse/rat/human skeletons). This ties it
together: ``run_comparison(source, spec)`` ingests any keypoint source, runs the
methods, and renders the report identically whether the subject is a mouse, rat,
or human — the only thing that changes is the ``DatasetSpec`` (skeleton, fps,
agents, labels).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import BehaviorSequence
from ..visualization import render_cluster_gallery, render_comparison_report
from .discovery import compare_discovery_methods

DEFAULT_METHODS = ("kmeans_pca_umap", "bsoid", "pca_hmm_fallback")


@dataclass
class DatasetSpec:
    """Consistent, species-agnostic description of a dataset for the pipeline."""

    name: str
    skeleton: str                       # skeleton-registry key (e.g. calms21, nwucla, rat7m)
    fps: float = 30.0
    n_agents: int = 1
    n_clusters: int = 8
    species: str = "unknown"            # mouse | rat | human | ...
    has_labels: bool = False
    label_names: list[str] = field(default_factory=list)


# A small registry so "any dataset" runs the same way. Extend freely — the point
# is that mouse/rat/human all differ only by these fields, not by code path.
DATASET_SPECS: dict[str, DatasetSpec] = {
    "calms21":  DatasetSpec("CalMS21", "calms21", 30.0, 2, 8, "mouse", True,
                            ["other", "attack", "investigation", "mount"]),
    "sdannce":  DatasetSpec("s-DANNCE", "rat7m", 50.0, 1, 10, "rat"),
    "mammal":   DatasetSpec("MAMMAL", "mars_mouse", 100.0, 1, 8, "mouse"),
    "nwucla":   DatasetSpec("NW-UCLA", "nwucla", 30.0, 1, 10, "human", True,
                            ["pickup1h", "pickup2h", "droptrash", "walk", "sit",
                             "stand", "don", "doff", "throw", "carry"]),
    "ntu":      DatasetSpec("NTU-RGBD", "ntu", 30.0, 1, 12, "human", True),
}


def _to_keypoints(source: Any, max_frames: int | None) -> np.ndarray:
    if isinstance(source, (str, Path)):
        from ..data import ingest
        kp = ingest(source)[0].keypoints
    elif isinstance(source, BehaviorSequence):
        kp = source.keypoints
    else:
        kp = np.asarray(source, dtype=np.float32)
    kp = np.nan_to_num(kp, nan=0.0)
    return kp[:max_frames] if max_frames else kp


def run_comparison(source: Any, spec: DatasetSpec | str, out_html: str | Path, *,
                   methods: tuple[str, ...] = DEFAULT_METHODS,
                   extra_labels: dict[str, np.ndarray] | None = None,
                   ground_truth: Any = None, max_frames: int | None = None,
                   gallery_html: str | Path | None = None) -> dict[str, Any]:
    """Run the discovery-method comparison on any source, render the report.

    Args:
        source: keypoint file path (npz/dannce_mat/DLC/SLEAP), ``BehaviorSequence``,
            or a raw ``(T,K,D)`` array.
        spec: a ``DatasetSpec`` or a key in ``DATASET_SPECS``.
        methods: lightweight methods run in-process; heavy/isolated methods
            (keypoint-MoSeq, VAME) are passed via ``extra_labels`` (precomputed).
        extra_labels: ``{method: labels}`` from isolated conda-env runs.
        ground_truth: optional per-frame labels for ARI/NMI evaluation.
        gallery_html: if given, also render a per-cluster skeleton-GIF gallery.
    """
    spec = DATASET_SPECS[spec] if isinstance(spec, str) else spec
    kp = _to_keypoints(source, max_frames)

    results: dict[str, dict] = {}
    for run in compare_discovery_methods(kp, methods=methods, fps=spec.fps,
                                         n_clusters=spec.n_clusters):
        results[run.name] = {"labels": run.result.labels, "features": run.result.features}
    for name, labels in (extra_labels or {}).items():
        labels = np.asarray(labels)
        results[name] = {"labels": labels[: len(kp)], "features": None}

    render_comparison_report(results, out_html, fps=spec.fps, ground_truth=ground_truth,
                             title=f"{spec.name} ({spec.species}) — method comparison")
    if gallery_html is not None:
        render_cluster_gallery(kp, results, spec.skeleton, gallery_html, fps=spec.fps)
    return results


__all__ = ["DatasetSpec", "DATASET_SPECS", "run_comparison"]
