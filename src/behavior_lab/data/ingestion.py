"""Unified data ingestion — one consistent standard to COLLECT / ORGANIZE / MANAGE /
PREPROCESS diverse behavior data into the canonical ``BehaviorSequence``.

Design (MoA-deliberated, deliberately minimal — functions + files, no DB/ETL):
- One ``ingest(source, source_type="auto", ...)`` façade over generic FORMAT loaders
  (DeepLabCut CSV/H5, SLEAP, npz) so ANY new keypoint file is ingested identically —
  distinct from the dataset-specific ``LOADER_REGISTRY`` (which knows one dataset each).
- Every sequence carries a standard ``metadata["provenance"]`` (IngestProvenance).
  Unknown fields stay ``None`` — never guessed (data integrity).
- Preprocessing = the existing ``PreprocessingPipeline`` applied after load + recorded.
- Management = an append-only ``manifest.jsonl`` (diffable, reproducible, no database).

BOUNDARY: video -> keypoints (SLEAP/DeepLabCut/DANNCE pose estimation) stays OUT of
core. Run pose estimators externally, then ingest their exported keypoint files. A thin
subprocess adapter may do both and record the command in provenance (future, optional).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.types import BehaviorSequence

INGEST_VERSION = "behavior-lab-ingest-v1"


@dataclass
class IngestProvenance:
    """Standard provenance stamped on every ingested ``BehaviorSequence``.

    ``units`` / ``coordinate_frame`` are ``None`` when the raw file does not state
    them — they are never guessed.
    """

    source_uri: str
    source_format: str
    loader: str
    checksum_sha256: str | None = None
    skeleton_name: str | None = None
    fps: float | None = None
    units: str | None = None
    coordinate_frame: str | None = None
    preprocessing: list[str] = field(default_factory=list)
    ingest_version: str = INGEST_VERSION


# --------------------------------------------------------------------------- #
# format detection
# --------------------------------------------------------------------------- #
_EXT_FORMAT: dict[str, str] = {
    ".slp": "sleap",
    ".csv": "dlc_csv",
    ".npz": "npz",
    ".npy": "npz",
    ".h5": "_h5",
    ".hdf5": "_h5",
}


def detect_format(path: Path) -> str:
    """Infer the ingestion format from extension (and h5 content sniffing)."""
    fmt = _EXT_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"Cannot auto-detect format for '{path.name}'. Pass source_type explicitly; "
            f"known formats: {sorted(FORMAT_LOADERS)}"
        )
    if fmt == "_h5":
        fmt = _sniff_h5(path)
    return fmt


def _sniff_h5(path: Path) -> str:
    try:
        import h5py

        with h5py.File(path, "r") as h5:
            if "tracks" in h5:
                return "sleap"
    except Exception:
        pass
    return "dlc_h5"


# --------------------------------------------------------------------------- #
# generic format loaders  (source path -> list[BehaviorSequence])
# --------------------------------------------------------------------------- #
def _load_sleap(path: Path) -> list[BehaviorSequence]:
    from ..pose.sleap import load_sleap_file

    return list(load_sleap_file(path).sequences)


def _load_dlc(path: Path) -> list[BehaviorSequence]:
    """DeepLabCut single-animal CSV/H5 -> one BehaviorSequence (T, K, 2)."""
    import pandas as pd

    if path.suffix.lower() in {".h5", ".hdf5"}:
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    # Columns are a (scorer, bodypart, coord) MultiIndex.
    bodyparts = list(dict.fromkeys(df.columns.get_level_values(-2)))
    if any(lvl == "individuals" for lvl in (df.columns.names or [])):
        raise ValueError("Multi-animal DLC not supported here; export via SLEAP or split per individual.")
    T, K = len(df), len(bodyparts)
    arr = np.full((T, K, 2), np.nan, dtype=np.float32)
    for k, bp in enumerate(bodyparts):
        for d, coord in enumerate(("x", "y")):
            cols = [c for c in df.columns if c[-2] == bp and c[-1] == coord]
            if cols:
                arr[:, k, d] = df[cols[0]].to_numpy(dtype=np.float32)
    return [BehaviorSequence(keypoints=arr, sample_id=path.stem, metadata={"node_names": bodyparts})]


def _load_npz(path: Path) -> list[BehaviorSequence]:
    """Generic npz/npy with a keypoints array of shape (T, K, D) or (N, T, K, D)."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray):
        arrays = [obj]
    else:
        key = next((k for k in ("keypoints", "kp", "poses", "x") if k in obj), None)
        if key is None:
            raise ValueError(f"{path.name}: no keypoints array (looked for keypoints/kp/poses/x). Keys: {list(obj.keys())}")
        data = obj[key]
        arrays = list(data) if data.ndim == 4 else [data]
    seqs = []
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 3:
            raise ValueError(f"{path.name}: expected (T,K,D) keypoints, got {a.shape}")
        seqs.append(BehaviorSequence(keypoints=a, sample_id=f"{path.stem}_{i}" if len(arrays) > 1 else path.stem))
    return seqs


FORMAT_LOADERS: dict[str, Callable[[Path], list[BehaviorSequence]]] = {
    "sleap": _load_sleap,
    "dlc_csv": _load_dlc,
    "dlc_h5": _load_dlc,
    "npz": _load_npz,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def _pipeline_step_names(pipeline) -> list[str]:
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        return [str(pipeline)]
    return [getattr(s, "name", type(s).__name__) for s in steps]


def append_manifest(manifest_path, source: Path, fmt: str, checksum: str,
                    seqs: list[BehaviorSequence]) -> None:
    """Append one JSONL record describing this ingested source (management SSOT)."""
    rec = {
        "source_uri": str(source),
        "format": fmt,
        "checksum_sha256": checksum,
        "n_sequences": len(seqs),
        "total_frames": int(sum(s.num_frames for s in seqs)),
        "skeleton": (seqs[0].skeleton_name or None) if seqs else None,
        "fps": seqs[0].fps if seqs else None,
        "ingest_version": INGEST_VERSION,
    }
    mp = Path(manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# the façade
# --------------------------------------------------------------------------- #
def ingest(source, *, source_type: str = "auto", preprocess=None,
           skeleton_name: str = "", units: str | None = None,
           coordinate_frame: str | None = None, manifest=None) -> list[BehaviorSequence]:
    """Ingest one keypoint file into canonical ``BehaviorSequence`` list with provenance.

    Args:
        source: path to a keypoint file (SLEAP/DeepLabCut/npz).
        source_type: format key or "auto" (detect by extension/content).
        preprocess: optional callable pipeline ``(T,K,D)->(T,K,D)`` (e.g. PreprocessingPipeline).
        skeleton_name / units / coordinate_frame: recorded in provenance; ``None`` = unknown.
        manifest: optional path to a ``manifest.jsonl`` to append a management record.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = detect_format(path) if source_type == "auto" else source_type
    loader = FORMAT_LOADERS.get(fmt)
    if loader is None:
        raise ValueError(f"Unknown ingestion format '{fmt}'. Known: {sorted(FORMAT_LOADERS)}. "
                         f"(For a whole known dataset directory use data.loaders.get_loader.)")
    seqs = loader(path)

    applied: list[str] = []
    if preprocess is not None:
        applied = _pipeline_step_names(preprocess)
        for seq in seqs:
            seq.keypoints = np.asarray(preprocess(seq.keypoints), dtype=np.float32)

    checksum = sha256_file(path)
    for seq in seqs:
        if skeleton_name:
            seq.skeleton_name = skeleton_name
        prov = IngestProvenance(
            source_uri=str(path),
            source_format=fmt,
            loader=getattr(loader, "__name__", str(loader)),
            checksum_sha256=checksum,
            skeleton_name=seq.skeleton_name or None,
            fps=seq.fps,
            units=units,
            coordinate_frame=coordinate_frame,
            preprocessing=applied,
        )
        seq.metadata["provenance"] = asdict(prov)

    if manifest is not None:
        append_manifest(manifest, path, fmt, checksum, seqs)
    return seqs


__all__ = ["ingest", "IngestProvenance", "detect_format", "append_manifest",
           "sha256_file", "FORMAT_LOADERS"]
