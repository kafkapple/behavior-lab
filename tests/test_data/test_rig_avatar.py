"""AVATAR rig module: layout cells, triangulation round trip, residual of exact and of shifted
detections."""

import cv2
import numpy as np
import pytest

from behavior_lab.rig import avatar
from behavior_lab.rig.multiview import load_calib, reproj_px, triangulate

SIZES = {
    "cam_1": (1200, 1000),
    "cam_2": (1200, 1000),
    "cam_3": (1200, 1200),
    "cam_4": (1200, 1000),
    "cam_5": (1200, 1000),
}
RVECS = [(0, 0, 0), (0, 0.6, 0), (0.6, 0, 0), (0, -0.6, 0), (-0.6, 0, 0)]
TVECS = [(0, 0, 1.0), (-0.5, 0, 0.9), (0, 0.5, 0.9), (0.5, 0, 0.9), (0, -0.5, 0.9)]


@pytest.fixture()
def calib(tmp_path):
    lines = ["camera_count = 5"]
    for (name, (w, h)), r, t in zip(SIZES.items(), RVECS, TVECS):
        lines += [
            f"[{name}]",
            f"size = [{w}, {h}]",
            f"matrix = [[1000.0, 0.0, {w / 2}], [0.0, 1000.0, {h / 2}], [0.0, 0.0, 1.0]]",
            "distortions = [0.0, 0.0, 0.0, 0.0, 0.0]",
            f"rotation = {list(r)}",
            f"translation = {list(t)}",
        ]
    p = tmp_path / "config.toml"
    p.write_text("\n".join(lines))
    return p


def _project(X, cam):
    rvec, _ = cv2.Rodrigues(cam["P"][:, :3])
    return cv2.projectPoints(X.reshape(1, 3), rvec, cam["P"][:, 3], cam["K"], cam["D"])[0].reshape(
        2
    )


def _kp(cams, shift_cam=None, shift=0.0, n=5):
    origin = np.array(list(avatar.ORIGIN.values()), float)
    rng = np.random.default_rng(0)
    frames = {}
    for f in range(n):
        pts = rng.uniform(-0.05, 0.05, (3, 3))
        frames[str(f)] = {
            str(c): {
                str(k): [
                    *(
                        (_project(pts[k], cam) + (shift if c == shift_cam else 0) + origin[c])
                        / avatar.COMPOSITE
                    ),
                    0.9,
                ]
                for k in range(3)
            }
            for c, cam in enumerate(cams)
        }
    return {"node": {"id": ["a", "b", "c"]}, "keypoint": frames}


def test_cells_match_calibrated_sizes(calib):
    cs = avatar.cells(calib)
    assert cs["cam_3"] == (2400, 0, 1200, 1200) and cs["cam_5"] == (1200, 1000, 1200, 1000)


def test_triangulation_round_trip(calib):
    cams = load_calib(calib)
    X = np.array([0.02, -0.03, 0.01])
    obs = [(c, _project(X, cam)) for c, cam in enumerate(cams)]
    Xh = triangulate(obs, cams)
    assert np.allclose(Xh, X, atol=1e-6)
    assert max(reproj_px(Xh, c, xy, cams) for c, xy in obs) < 1e-3


def test_residual_zero_when_exact_and_positive_when_one_camera_is_off(calib):
    cams = load_calib(calib)
    exact = avatar.residuals(_kp(cams), cams)
    assert exact["triangulated_frac"] == 1.0 and exact["median_px"] < 1e-3
    off = avatar.residuals(_kp(cams, shift_cam=2, shift=50.0), cams)
    assert off["median_px"] > 5 and int(np.argmax(off["per_cam_median_px"])) == 2
