"""Predictor tables and comparison. Inputs are synthetic by construction, never real predictions."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from behavior_lab.pose.predictors import compare, registry, tables
from behavior_lab.pose.predictors.joints import PART_MAP
from behavior_lab.pose.predictors.run_vitpose import boxes_from_table
from behavior_lab.rig.avatar import COMPOSITE, ORIGIN


def _subtle(tmp_path: Path) -> Path:
    comp = np.array(COMPOSITE, float)
    kp = {"0": {"1": {"0": list((np.array([1300.0, 100.0]) / comp).round(6)) + [0.9], "1": [0.0, 0.0, 0.0]}}}
    d = {"node": {"cnt": 2, "id": ["nose1", "neck1"]}, "keypoint": kp}
    p = tmp_path / "m.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump(d, fh)
    return p


def test_subtle_json_to_cell_pixels_and_missing(tmp_path):
    df = tables.from_subtle_json(_subtle(tmp_path), "sleap_1423", frames=[0])
    assert len(df) == 5 * 2
    hit = df[(df.camera == "cam_2") & (df.keypoint == "nose1")].iloc[0]
    assert hit.x_px == pytest.approx(1300 - ORIGIN["cam_2"][0], abs=0.01) and hit.y_px == pytest.approx(100, abs=0.01)
    assert df[(df.camera == "cam_2") & (df.keypoint == "neck1")].x_px.isna().all()
    assert df.x_px.notna().sum() == 1


def test_dlc_video_csv_missing_marker_and_row_count(tmp_path):
    p = tmp_path / "sa.csv"
    p.write_text(",nose_x,nose_y,nose_likelihood\n0,10.0,20.0,0.8\n1,-1.0,-1.0,-1.0\n")
    df = tables.from_dlc_video_csv(p, "superanimal_quadruped", "cam_1", frames=[6, 23])
    assert df.x_px.tolist()[0] == 10.0 and np.isnan(df.x_px.tolist()[1])
    with pytest.raises(ValueError, match="rows for"):
        tables.from_dlc_video_csv(p, "m", "cam_1", frames=[1])


def _two_models(swap_ears: bool, offset: float) -> "pd.DataFrame":
    rows = []
    for model, ears in (("sleap_1423", ("earL1", "earR1")), ("superanimal_quadruped", ("left_earend", "right_earend"))):
        nose = "nose1" if model == "sleap_1423" else "nose"
        shift = offset if model != "sleap_1423" else 0.0
        e = ears[::-1] if (swap_ears and model != "sleap_1423") else ears
        rows += [(model, 0, "cam_1", nose, 100.0 + shift, 100.0, 0.9),
                 (model, 0, "cam_1", e[0], 50.0, 60.0, 0.9), (model, 0, "cam_1", e[1], 150.0, 60.0, 0.9)]
    present = {(r[0], r[3]) for r in rows}
    for model in ("sleap_1423", "superanimal_quadruped"):  # every mapped name must exist in a real table, missing = NaN
        rows += [(model, 0, "cam_1", n, np.nan, np.nan, np.nan) for n in PART_MAP[model] if (model, n) not in present]
    return pd.DataFrame(rows, columns=list(tables.COLUMNS))


def test_agreement_is_left_right_agnostic_and_measures_offset():
    for swap in (False, True):
        out = compare.agreement(_two_models(swap, 3.0), ["sleap_1423", "superanimal_quadruped"])
        by = out.set_index("part")["median_px"]
        assert by["nose"] == pytest.approx(3.0)
        assert by["ear"] == pytest.approx(0.0)


def test_part_points_rejects_unknown_mapped_name():
    df = _two_models(False, 0.0)
    df = df[df.keypoint != "nose1"]
    with pytest.raises(KeyError, match="nose1"):
        compare.part_points(df, "sleap_1423")


def test_low_confidence_points_are_ignored():
    df = _two_models(False, 0.0)
    df.loc[df.keypoint == "nose", "conf"] = 0.1
    out = compare.agreement(df, ["sleap_1423", "superanimal_quadruped"])
    assert "nose" not in set(out.part)


def test_part_map_has_no_unknown_parts():
    from behavior_lab.pose.predictors.joints import PARTS
    assert all(v in PARTS for m in PART_MAP.values() for v in m.values())


def test_registry_statuses_cover_every_part_map_model():
    assert set(PART_MAP) <= {m.name for m in registry.MODELS}
    assert {m.name for m in registry.by_status(registry.NEEDS_LABELS)} >= {"lightning_pose", "dannce"}
    assert registry.get("dannce").n_keypoints is None


def test_boxes_from_table_pad_and_minimum_size():
    df = _two_models(False, 0.0)
    box = boxes_from_table(df, "sleap_1423", pad=0.25).iloc[0]
    assert (box.x0, box.x1) == (50 - 0.25 * 100, 150 + 0.25 * 100)
    assert box.y1 - box.y0 == pytest.approx(40 * 1.5)  # min size 20 per axis would not apply, extent is 40


def test_reprojection_of_consistent_points_is_near_zero():
    cv2 = pytest.importorskip("cv2")
    K = np.array([[500.0, 0, 300], [0, 500.0, 300], [0, 0, 1]])
    cams = []
    for i, tx in enumerate((-1.0, 0.0, 1.0)):
        R = cv2.Rodrigues(np.array([0.0, 0.1 * (i - 1), 0.0]))[0]
        t = np.array([tx, 0.0, 5.0])
        cams.append({"K": K, "D": np.zeros(5), "P": np.hstack([R, t[:, None]]), "size": (600, 600)})
    X = np.array([0.1, -0.2, 0.3])
    rows = []
    for i, c in enumerate(cams):
        p = c["P"] @ np.append(X, 1.0)
        uv = K @ (p / p[2])
        rows.append(("m", 0, f"cam_{i + 1}", "nose", float(uv[0]), float(uv[1]), 0.9))
    df = pd.DataFrame(rows, columns=list(tables.COLUMNS))
    out = compare.reprojection(df, "m", cams)
    assert out["n_triangulated"] == 1 and out["median_px"] < 1e-3
    df.loc[0, "x_px"] += 40
    assert compare.reprojection(df, "m", cams)["median_px"] > 5
