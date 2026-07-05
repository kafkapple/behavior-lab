"""SLEAP pose import adapters.

The main behavior-lab contract is ``BehaviorSequence.keypoints`` in
``(T, K, D)`` format. This module converts common SLEAP outputs into that
shape while preserving node names, track names, confidence scores, and source
paths in metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import BehaviorSequence


@dataclass(frozen=True)
class SLEAPImportResult:
    """Parsed SLEAP file contents in canonical sequence form."""

    sequences: list[BehaviorSequence]
    node_names: list[str]
    track_names: list[str]
    source_path: Path


def load_sleap_file(
    path: str | Path,
    *,
    fps: float = 30.0,
    skeleton_name: str = "sleap",
    instance_mode: str = "flatten",
    confidence_threshold: float | None = None,
) -> SLEAPImportResult:
    """Load SLEAP ``.slp`` or analysis ``.h5`` output.

    Args:
        path: SLEAP labels/predictions file. ``.slp`` uses optional
            ``sleap-io``. ``.h5``/``.hdf5`` uses ``h5py`` and supports SLEAP
            analysis exports containing ``tracks``.
        fps: Frames per second to store on returned sequences.
        skeleton_name: Skeleton registry name to record in metadata.
        instance_mode:
            ``"flatten"`` returns one sequence with instances concatenated as
            ``K_total = tracks * nodes``.
            ``"separate"`` returns one sequence per track.
        confidence_threshold: If point scores are present, coordinates below
            this threshold are replaced with ``NaN``.

    Returns:
        SLEAPImportResult containing one or more BehaviorSequence objects.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".slp":
        return _load_slp_with_sleap_io(
            path,
            fps=fps,
            skeleton_name=skeleton_name,
            instance_mode=instance_mode,
            confidence_threshold=confidence_threshold,
        )
    if suffix in {".h5", ".hdf5"}:
        return _load_analysis_h5(
            path,
            fps=fps,
            skeleton_name=skeleton_name,
            instance_mode=instance_mode,
            confidence_threshold=confidence_threshold,
        )
    raise ValueError(f"Unsupported SLEAP file extension: {path.suffix}")


def _load_analysis_h5(
    path: Path,
    *,
    fps: float,
    skeleton_name: str,
    instance_mode: str,
    confidence_threshold: float | None,
) -> SLEAPImportResult:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py or behavior-lab[loaders] to read SLEAP H5 files") from exc

    with h5py.File(path, "r") as h5:
        if "tracks" not in h5:
            raise KeyError(f"{path} has no 'tracks' dataset. Available keys: {list(h5.keys())}")
        tracks = np.asarray(h5["tracks"], dtype=np.float32)
        point_scores = np.asarray(h5["point_scores"], dtype=np.float32) if "point_scores" in h5 else None
        node_names = _decode_h5_strings(h5.get("node_names"))
        track_names = _decode_h5_strings(h5.get("track_names"))

    coords = _normalize_sleap_tracks(tracks)
    scores = _normalize_sleap_scores(point_scores, coords.shape) if point_scores is not None else None
    return _sequences_from_coords(
        coords,
        scores,
        source_path=path,
        fps=fps,
        skeleton_name=skeleton_name,
        node_names=node_names,
        track_names=track_names,
        instance_mode=instance_mode,
        confidence_threshold=confidence_threshold,
    )


def _load_slp_with_sleap_io(
    path: Path,
    *,
    fps: float,
    skeleton_name: str,
    instance_mode: str,
    confidence_threshold: float | None,
) -> SLEAPImportResult:
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise ImportError(
            "Install sleap-io to read .slp files, or export SLEAP analysis .h5 first"
        ) from exc

    labels = sio.load_slp(path)
    labeled_frames = list(getattr(labels, "labeled_frames", []))
    if not labeled_frames:
        raise ValueError(f"No labeled frames found in {path}")

    skeleton = _first_attr(labels, ("skeleton", "skeletons"), default=None)
    if isinstance(skeleton, (list, tuple)):
        skeleton = skeleton[0] if skeleton else None
    node_names = _node_names_from_skeleton(skeleton)

    frame_numbers = [int(_first_attr(lf, ("frame_idx", "frame_index"), default=i))
                     for i, lf in enumerate(labeled_frames)]
    T = max(frame_numbers) + 1
    max_instances = max(len(getattr(lf, "instances", [])) for lf in labeled_frames)
    if max_instances == 0:
        raise ValueError(f"No instances found in {path}")

    K = len(node_names) or _infer_num_nodes_from_slp(labeled_frames)
    coords = np.full((T, max_instances, K, 2), np.nan, dtype=np.float32)
    scores = np.full((T, max_instances, K), np.nan, dtype=np.float32)

    for lf, frame_idx in zip(labeled_frames, frame_numbers):
        for inst_idx, inst in enumerate(getattr(lf, "instances", [])):
            pts = _instance_points(inst)
            n = min(K, pts.shape[0])
            coords[frame_idx, inst_idx, :n, : pts.shape[1]] = pts[:n, :2]
            inst_scores = _instance_scores(inst)
            if inst_scores is not None:
                scores[frame_idx, inst_idx, : min(K, len(inst_scores))] = inst_scores[:K]

    track_names = [f"track_{i}" for i in range(max_instances)]
    return _sequences_from_coords(
        coords,
        scores,
        source_path=path,
        fps=fps,
        skeleton_name=skeleton_name,
        node_names=node_names,
        track_names=track_names,
        instance_mode=instance_mode,
        confidence_threshold=confidence_threshold,
    )


def _sequences_from_coords(
    coords: np.ndarray,
    scores: np.ndarray | None,
    *,
    source_path: Path,
    fps: float,
    skeleton_name: str,
    node_names: list[str],
    track_names: list[str],
    instance_mode: str,
    confidence_threshold: float | None,
) -> SLEAPImportResult:
    if confidence_threshold is not None and scores is not None:
        coords = coords.copy()
        coords[scores < confidence_threshold] = np.nan

    T, M, K, D = coords.shape
    if not node_names:
        node_names = [f"node_{i}" for i in range(K)]
    if not track_names:
        track_names = [f"track_{i}" for i in range(M)]

    base_meta: dict[str, Any] = {
        "source": "sleap",
        "source_file": str(source_path),
        "node_names": node_names,
        "track_names": track_names,
        "confidence_threshold": confidence_threshold,
    }

    if instance_mode == "flatten":
        keypoints = coords.reshape(T, M * K, D)
        meta = dict(base_meta, instance_mode="flatten", num_instances=M, joints_per_instance=K)
        if scores is not None:
            meta["point_scores"] = scores.reshape(T, M * K)
        sequences = [BehaviorSequence(
            keypoints=keypoints.astype(np.float32),
            skeleton_name=skeleton_name,
            sample_id=source_path.stem,
            fps=fps,
            metadata=meta,
        )]
    elif instance_mode == "separate":
        sequences = []
        for i in range(M):
            meta = dict(base_meta, instance_mode="separate", track_index=i, track_name=track_names[i])
            if scores is not None:
                meta["point_scores"] = scores[:, i]
            sequences.append(BehaviorSequence(
                keypoints=coords[:, i].astype(np.float32),
                skeleton_name=skeleton_name,
                sample_id=f"{source_path.stem}_{track_names[i]}",
                fps=fps,
                metadata=meta,
            ))
    else:
        raise ValueError("instance_mode must be 'flatten' or 'separate'")

    return SLEAPImportResult(sequences, node_names, track_names, source_path)


def _normalize_sleap_tracks(tracks: np.ndarray) -> np.ndarray:
    """Normalize common SLEAP tracks layouts to ``(T, tracks, nodes, xy)``."""
    arr = np.asarray(tracks, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected SLEAP tracks to be 4-D, got {arr.shape}")
    # The coordinate (xy/xyz) axis is the trailing axis in SLEAP analysis
    # exports (frames, nodes, tracks, xy). Search from the end so a nodes or
    # tracks count of 2/3 is not mistaken for the coordinate axis.
    dim_axis = next((i for i in reversed(range(arr.ndim)) if arr.shape[i] in (2, 3)), None)
    if dim_axis is None:
        raise ValueError(f"Could not locate coordinate dimension in tracks shape {arr.shape}")
    arr = np.moveaxis(arr, dim_axis, -1)
    if arr.shape[1] > arr.shape[2]:
        # Typical SLEAP analysis export is (frames, nodes, tracks, xy).
        arr = arr.transpose(0, 2, 1, 3)
    return arr[..., :2]


def _normalize_sleap_scores(scores: np.ndarray, coords_shape: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    T, M, K, _ = coords_shape
    if arr.shape == (T, K, M):
        return arr.transpose(0, 2, 1)
    if arr.shape == (T, M, K):
        return arr
    if arr.ndim == 3:
        axes = list(range(arr.ndim))
        t_axis = next((i for i, size in enumerate(arr.shape) if size == T), 0)
        axes.remove(t_axis)
        arr = np.moveaxis(arr, t_axis, 0)
        if arr.shape == (T, K, M):
            return arr.transpose(0, 2, 1)
    raise ValueError(f"Could not align point_scores shape {arr.shape} with coords {coords_shape}")


def _decode_h5_strings(dataset: Any) -> list[str]:
    if dataset is None:
        return []
    values = np.asarray(dataset)
    out = []
    for value in values.reshape(-1):
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _first_attr(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _node_names_from_skeleton(skeleton: Any) -> list[str]:
    if skeleton is None:
        return []
    nodes = _first_attr(skeleton, ("nodes", "node_names"), default=[])
    names = []
    for node in nodes:
        names.append(str(getattr(node, "name", node)))
    return names


def _infer_num_nodes_from_slp(labeled_frames: list[Any]) -> int:
    for lf in labeled_frames:
        for inst in getattr(lf, "instances", []):
            points = _instance_points(inst)
            if points.size:
                return points.shape[0]
    raise ValueError("Could not infer node count from .slp instances")


def _instance_points(instance: Any) -> np.ndarray:
    points = _first_attr(instance, ("points_array", "numpy"), default=None)
    if callable(points):
        points = points()
    if points is None:
        points = _first_attr(instance, ("points",), default=None)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.shape[-1] > 2:
        arr = arr[..., :2]
    return arr


def _instance_scores(instance: Any) -> np.ndarray | None:
    scores = _first_attr(instance, ("scores", "point_scores"), default=None)
    if scores is None:
        return None
    return np.asarray(scores, dtype=np.float32).reshape(-1)


__all__ = ["SLEAPImportResult", "load_sleap_file"]
