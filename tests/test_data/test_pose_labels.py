"""Label-table exporters. Inputs are synthetic by construction (unit-test tables), never real labels."""
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from behavior_lab.pose.labels import labeled_fraction, load_label_table, to_coco, to_dlc_csv, to_sleap

KEYPOINTS = ["nose", "neck", "tail"]


def _table() -> pd.DataFrame:
    rows = []
    for camera, width in (("cam_1", 100), ("cam_2", 120)):
        for frame in (3, 7):
            for k, name in enumerate(KEYPOINTS):
                rows.append({"frame": frame, "camera": camera, "image": f"images/f{frame:03d}_{camera}.png",
                             "image_w": width, "image_h": 80, "keypoint": name,
                             "x_px": 10.0 + 10 * k + frame, "y_px": 20.0 + k, "visible": 1.0, "note": ""})
    df = pd.DataFrame(rows)
    df.loc[(df.camera == "cam_1") & (df.frame == 3) & (df.keypoint == "neck"), ["x_px", "y_px", "visible"]] = [30.0, 40.0, 0.0]
    df.loc[(df.camera == "cam_1") & (df.frame == 3) & (df.keypoint == "tail"), ["x_px", "y_px", "visible"]] = np.nan
    return df


def test_load_validates_and_orders():
    df = load_label_table(_table(), keypoints=KEYPOINTS)
    assert list(df["keypoint"].cat.categories) == KEYPOINTS
    assert labeled_fraction(df) == pytest.approx(11 / 12)
    bad = _table()
    bad.loc[0, "visible"] = 2
    with pytest.raises(ValueError, match="visible must be"):
        load_label_table(bad)
    bad = _table()
    bad.loc[0, "x_px"] = 500.0
    with pytest.raises(ValueError, match="outside"):
        load_label_table(bad)
    bad = _table()
    bad.loc[0, "visible"] = np.nan
    with pytest.raises(ValueError, match="disagree"):
        load_label_table(bad)


def test_dlc_csv_layout_per_camera(tmp_path: Path):
    paths = to_dlc_csv(load_label_table(_table(), KEYPOINTS), tmp_path, scorer="joon")
    assert [p.name for p in paths] == ["CollectedData_joon_cam_1.csv", "CollectedData_joon_cam_2.csv"]
    table = pd.read_csv(paths[0], header=[0, 1, 2], index_col=0)
    assert table.shape == (2, 6)
    assert list(table.columns.get_level_values(1).unique()) == KEYPOINTS
    assert table.index.tolist() == ["images/f003_cam_1.png", "images/f007_cam_1.png"]
    assert np.isnan(table.loc["images/f003_cam_1.png", ("joon", "tail", "x")])
    assert table.loc["images/f003_cam_1.png", ("joon", "neck", "x")] == 30.0


def test_coco_visibility_and_bbox():
    coco = to_coco(load_label_table(_table(), KEYPOINTS), edges=[("nose", "neck"), ("neck", "tail")])
    assert coco["categories"][0]["skeleton"] == [[1, 2], [2, 3]]
    first = next(a for a, i in zip(coco["annotations"], coco["images"]) if i["file_name"] == "images/f003_cam_1.png")
    assert first["keypoints"][2::3] == [2, 1, 0]          # seen, occluded-located, unlabelled
    assert first["num_keypoints"] == 2
    x, y, w, h = first["bbox"]
    assert x >= 0 and y >= 0 and x + w <= 100 and y + h <= 80


def test_sleap_roundtrip(tmp_path: Path):
    sio = pytest.importorskip("sleap_io")
    imageio = pytest.importorskip("imageio.v3")
    df = load_label_table(_table(), KEYPOINTS)
    for image, w in df.groupby("image")["image_w"].first().items():
        (tmp_path / image).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(tmp_path / image, np.zeros((80, int(w), 3), dtype=np.uint8))
    out = to_sleap(df, tmp_path, tmp_path / "labels.slp", edges=[("nose", "neck")])
    labels = sio.load_slp(str(out))
    assert len(labels.labeled_frames) == 4
    assert [n.name for n in labels.skeletons[0].nodes] == KEYPOINTS
    points = labels.labeled_frames[0].instances[0].numpy()
    assert points.shape == (3, 2) and np.isnan(points[2]).all()
