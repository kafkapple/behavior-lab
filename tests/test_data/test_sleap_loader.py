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
