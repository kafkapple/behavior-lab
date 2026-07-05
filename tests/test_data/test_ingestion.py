"""Tests for the unified data ingestion façade."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from behavior_lab.data.ingestion import (
    IngestProvenance,
    detect_format,
    ingest,
    sha256_file,
)


def _write_dlc_csv(path: Path, T: int, bodyparts: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer"] + ["net"] * (len(bodyparts) * 3))
        bp_row, coord_row = ["bodyparts"], ["coords"]
        for bp in bodyparts:
            bp_row += [bp, bp, bp]
            coord_row += ["x", "y", "likelihood"]
        w.writerow(bp_row)
        w.writerow(coord_row)
        for t in range(T):
            row = [t]
            for k in range(len(bodyparts)):
                row += [float(t), float(k), 0.9]
            w.writerow(row)


def test_detect_format():
    assert detect_format(Path("a.slp")) == "sleap"
    assert detect_format(Path("a.csv")) == "dlc_csv"
    assert detect_format(Path("a.npz")) == "npz"
    with pytest.raises(ValueError):
        detect_format(Path("a.txt"))


def test_ingest_dlc_csv_with_provenance_and_manifest(tmp_path):
    csv_path = tmp_path / "pred.csv"
    _write_dlc_csv(csv_path, T=10, bodyparts=["nose", "neck", "tail"])
    manifest = tmp_path / "manifest.jsonl"

    seqs = ingest(csv_path, skeleton_name="mouse3", units="pixels", manifest=manifest)

    assert len(seqs) == 1
    seq = seqs[0]
    assert seq.keypoints.shape == (10, 3, 2)
    assert seq.skeleton_name == "mouse3"
    prov = seq.metadata["provenance"]
    assert prov["source_format"] == "dlc_csv"
    assert prov["checksum_sha256"] == sha256_file(csv_path)
    assert prov["units"] == "pixels"
    assert prov["coordinate_frame"] is None  # unknown -> not guessed
    assert prov["ingest_version"] == "behavior-lab-ingest-v1"

    # manifest appended
    rec = json.loads(manifest.read_text().strip())
    assert rec["format"] == "dlc_csv"
    assert rec["total_frames"] == 10
    assert rec["skeleton"] == "mouse3"


def test_ingest_npz_multi_sequence(tmp_path):
    npz = tmp_path / "data.npz"
    np.savez(npz, keypoints=np.zeros((2, 8, 5, 3), dtype=np.float32))  # (N, T, K, D)
    seqs = ingest(npz)
    assert len(seqs) == 2
    assert seqs[0].keypoints.shape == (8, 5, 3)
    assert seqs[0].metadata["provenance"]["source_format"] == "npz"


def test_ingest_with_preprocess_records_steps(tmp_path):
    npz = tmp_path / "d.npz"
    np.savez(npz, keypoints=np.ones((6, 4, 2), dtype=np.float32))

    class _Scale:
        name = "scale2x"

        def __call__(self, kp, **kw):
            return kp * 2.0

    class _Pipe:
        steps = [_Scale()]

        def __call__(self, kp, **kw):
            for s in self.steps:
                kp = s(kp)
            return kp

    seqs = ingest(npz, preprocess=_Pipe())
    assert float(seqs[0].keypoints[0, 0, 0]) == 2.0
    assert seqs[0].metadata["provenance"]["preprocessing"] == ["scale2x"]


def test_ingest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "nope.npz")
