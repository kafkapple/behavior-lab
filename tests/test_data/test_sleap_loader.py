from pathlib import Path

import numpy as np
import pytest

from behavior_lab.data.loaders import get_loader
from behavior_lab.pose.sleap import load_sleap_file


def test_sleap_analysis_h5_flatten(tmp_path: Path):
    pytest.importorskip("h5py")
    h5py = __import__("h5py")
    path = tmp_path / "predictions.analysis.h5"
    tracks = np.zeros((5, 3, 2, 2), dtype=np.float32)  # frames, nodes, tracks, xy
    tracks[:, :, 0, 0] = 1
    tracks[:, :, 1, 1] = 2
    scores = np.ones((5, 3, 2), dtype=np.float32)
    scores[0, 1, 0] = 0.1

    with h5py.File(path, "w") as h5:
        h5.create_dataset("tracks", data=tracks)
        h5.create_dataset("point_scores", data=scores)
        h5.create_dataset("node_names", data=np.array([b"nose", b"neck", b"tail_base"]))
        h5.create_dataset("track_names", data=np.array([b"resident", b"intruder"]))

    result = load_sleap_file(path, confidence_threshold=0.5)

    assert len(result.sequences) == 1
    seq = result.sequences[0]
    assert seq.keypoints.shape == (5, 6, 2)
    assert np.isnan(seq.keypoints[0, 1]).all()
    assert seq.metadata["node_names"] == ["nose", "neck", "tail_base"]
    assert seq.metadata["track_names"] == ["resident", "intruder"]


def test_sleap_analysis_h5_frames_last_layout(tmp_path: Path):
    """Real SLEAP analysis.h5 stores tracks as (tracks, xy, nodes, frames) — frames last."""
    pytest.importorskip("h5py")
    h5py = __import__("h5py")
    path = tmp_path / "real.analysis.h5"
    # (tracks=2, xy=2, nodes=5, frames=20); frames is the largest axis.
    tracks = np.zeros((2, 2, 5, 20), dtype=np.float32)
    tracks[0, 0] = 1.0  # track 0, x
    tracks[1, 1] = 2.0  # track 1, y
    scores = np.ones((2, 5, 20), dtype=np.float32)  # (tracks, nodes, frames)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("tracks", data=tracks)
        h5.create_dataset("point_scores", data=scores)
        h5.create_dataset("node_names", data=np.array([f"n{i}".encode() for i in range(5)]))
        h5.create_dataset("track_names", data=np.array([b"a", b"b"]))

    seq = load_sleap_file(path, confidence_threshold=0.5).sequences[0]
    # 20 frames, 5 nodes x 2 tracks flattened = 10, xy
    assert seq.keypoints.shape == (20, 10, 2)


def test_sleap_loader_factory_separate(tmp_path: Path):
    pytest.importorskip("h5py")
    h5py = __import__("h5py")
    path = tmp_path / "predictions.h5"
    tracks = np.zeros((4, 2, 2, 2), dtype=np.float32)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tracks", data=tracks)

    loader = get_loader("sleap", data_dir=tmp_path, instance_mode="separate")
    sequences = loader.load_all()

    assert len(sequences) == 2
    assert sequences[0].keypoints.shape == (4, 2, 2)
    assert sequences[0].metadata["source"] == "sleap"
