"""SLP label checks and the two report pages. Files are synthetic: hand-built hdf5 with the SLEAP layout, tiny images."""
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pd = pytest.importorskip("pandas")

from behavior_lab.pose import slp_labels, slp_report

NODES = ["nose1", "neck1", "earL1", "earR1", "forelegL1", "forelegR1", "tailstart1", "hindlegL1", "hindlegR1", "tail1", "tailend1"]


def _video(name, shape):
    return json.dumps({"backend": {"type": "HDF5Video", "shape": shape, "filename": "pkg.slp"}, "source_video": {"filename": f"D:/x/{name}"}})


def _write(path: Path, frames, videos, n_video_table=None):
    """frames: list of (video index, frame_idx, pts (11, 3))."""
    meta = {"nodes": [{"name": n} for n in NODES], "skeletons": [{"links": [{"source": 1, "target": 4}, {"source": 6, "target": 1}]}]}
    pts, inst, fr = [], [], []
    for k, (v, idx, p) in enumerate(frames):
        s = len(pts)
        pts += [(x, y, bool(vis), True) for x, y, vis in p]
        inst.append((k, 0, k, 0, -1, -1, 1.0, s, s + len(p), 1.0))
        fr.append((k, v, idx, k, k + 1))
    with h5py.File(path, "w") as f:
        f.create_dataset("metadata", data=0).attrs["json"] = json.dumps(meta)
        vt = videos if n_video_table is None else videos[:n_video_table]
        f.create_dataset("videos_json", data=np.array([x.encode() for x in vt]))
        f.create_dataset("frames", data=np.array(fr, dtype=[("frame_id", "i8"), ("video", "u4"), ("frame_idx", "i8"), ("instance_id_start", "i8"), ("instance_id_end", "i8")]))
        f.create_dataset("instances", data=np.array(inst, dtype=[("instance_id", "i8"), ("instance_type", "u1"), ("frame_id", "i8"), ("skeleton", "u4"), ("track", "i4"), ("from_predicted", "i8"), ("score", "f4"), ("point_id_start", "i8"), ("point_id_end", "i8"), ("tracking_score", "f4")]))
        f.create_dataset("points", data=np.array(pts, dtype=[("x", "f8"), ("y", "f8"), ("visible", "?"), ("complete", "?")]))


def _pose(left_side=+1, occlude_neck=False, x_shift=0.0):
    """Tail at (100,500), neck at (100,300): axis points up the image. L on the positive side of cross(axis, L - tail) = x > 100 (image y points down)."""
    p = np.zeros((11, 3))
    p[:, 2] = 1
    p[6] = (100 + x_shift, 500, 1)
    p[1] = (100 + x_shift, 300, 0 if occlude_neck else 1)
    p[0] = (100 + x_shift, 250, 1)
    p[4] = (100 + x_shift + 40 * left_side, 300, 1)
    p[5] = (100 + x_shift - 40 * left_side, 300, 1)
    p[7] = (100 + x_shift + 40 * left_side, 480, 1)
    p[8] = (100 + x_shift - 40 * left_side, 480, 1)
    p[2], p[3], p[9], p[10] = (80, 260, 1), (120, 260, 1), (100, 520, 1), (100, 560, 1)
    return p


def _files(tmp_path):
    videos = [_video("rec_bot.mp4", [50, 1200, 1200, 3]), _video("other.mp4", [50, 720, 960, 3])]
    train = [(0, 10, _pose()), (0, 20, _pose(occlude_neck=True)), (0, 30, _pose(left_side=-1)), (1, 5, _pose(x_shift=900))]
    val = [(0, 10, _pose()), (0, 400, _pose()), (2, 7, _pose())]  # video index 2 is missing from the table
    _write(tmp_path / "t.slp", train, videos)
    _write(tmp_path / "v.slp", val, videos)
    return tmp_path / "t.slp", tmp_path / "v.slp"


def test_load_skips_frames_with_missing_video(tmp_path):
    t, v = _files(tmp_path)
    val = slp_labels.load_slp(v)
    assert val["n_frames"] == 3 and len(val["frames"]) == 2 and val["skipped"] == 1
    assert val["nodes"] == NODES and val["edges"] == [[1, 4], [6, 1]]


def test_analyze_counts_leakage_and_outside(tmp_path):
    t, v = _files(tmp_path)
    a = slp_labels.analyze(slp_labels.load_slp(t), slp_labels.load_slp(v))
    assert a["avatar_counts"] == {"train": 3, "val": 2}
    assert a["val_with_train_within"] == {"0": 1, "5": 1, "30": 1, "100": 1}   # frame 10 is shared; frame 400 is far from every train frame
    inside, outside = a["oob_by_size"]["1200x1200"], a["oob_by_size"]["960x720"]
    assert inside[0] == 0 and outside[0] > 0                            # x_shift 900 pushes points past 960
    assert len(a["avatar_frames"]) == 5


def test_side_check_counts_occluded_points_only_when_asked(tmp_path):
    t, _ = _files(tmp_path)
    frames = [f for f in slp_labels.load_slp(t)["frames"] if f["src"] == "rec_bot.mp4"]
    allf = slp_labels.side_check(frames, NODES, ("tailstart1", "neck1"), "forelegL1")
    vis = slp_labels.side_check(frames, NODES, ("tailstart1", "neck1"), "forelegL1", visible_only=True)
    assert allf == {"n": 3, "left_positive": pytest.approx(0.67, abs=0.01)}   # two frames L left of axis, one swapped
    assert vis["n"] == 2                                                      # the occluded-neck frame is dropped


def test_val_error_matches_by_video_and_frame(tmp_path):
    t, v = _files(tmp_path)
    val = slp_labels.load_slp(v)
    pred = [{"src": f["src"], "frame_idx": f["frame_idx"], "pts": f["pts"] + np.array([3.0, 4.0, 0])} for f in val["frames"]]
    e = slp_labels.val_error(val["frames"], pred)
    assert e.shape == (2, 11) and np.allclose(e, 5.0)


def test_label_report_page_contains_numbers(tmp_path):
    t, v = _files(tmp_path)
    a = slp_labels.analyze(slp_labels.load_slp(t), slp_labels.load_slp(v))
    out = slp_report.build(a, tmp_path / "p.html", ["synthetic source"], [("route", "state")], {"mOKS": 0.5})
    html = out.read_text()
    assert "nose1" in html and "synthetic source" in html and "mOKS" in html and "<canvas" in html
    assert "1 val frames reference a missing video" in html


def test_predictor_report_builds_from_a_minimal_directory(tmp_path):
    from PIL import Image

    from behavior_lab.pose.predictors import report, tables

    for cam in ("1", "2"):
        (tmp_path / "gt_label_261002" / "images").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 100), (200, 200, 200)).save(tmp_path / "gt_label_261002" / "images" / f"f006_cam{cam}.jpg")
    rows = []
    for model, names in (("sleap_1423", ["nose1", "neck1"]), ("superanimal_quadruped", ["nose", "neck_base"])):
        for cam in ("cam_1", "cam_2"):
            rows += [(model, 6, cam, n, 10.0 + i, 20.0, 0.9) for i, n in enumerate(names)]
    out = tmp_path / "kp_gt_261003"
    out.mkdir()
    tables.write_table(pd.DataFrame(rows, columns=list(tables.COLUMNS)), out / "all.csv")
    (out / "results.json").write_text(json.dumps({
        "detection": {m: {"cells": 4, "detected": 1.0, "confident": 1.0} for m in ("sleap_1423", "superanimal_quadruped")},
        "reprojection": [{"model": m, "n_triangulated": 0, "n_candidates": 2, "median_px": None, "p90_px": None} for m in ("sleap_1423", "superanimal_quadruped")],
        "agreement": [{"a": "sleap_1423", "b": "superanimal_quadruped", "part": "nose", "n": 2, "median_px": 1.5, "p90_px": 2.0}]}))
    side = tmp_path / "side.html"
    side.write_text('<html><body>"quoted" <b>x</b></body></html>')
    one = report.build(tmp_path, tmp_path / "one.html", embed={"Training labels": side}).read_text()
    assert "Training labels" in one and "srcdoc=" in one and "&quot;quoted&quot;" in one and '<div class="tab" data-i="0">' in one
    html = report.build(tmp_path, tmp_path / "cmp.html").read_text()
    assert "SuperAnimal-Quadruped" in html and "Lightning Pose" in html and '"kn"' in html and "1.5" in html
