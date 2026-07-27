"""Ground truth for the left/right relabelling, which the SBeA recordings lack.

A synthetic rig lets us swap known labels and check they come back. The real
sessions can only be judged by proxies (reprojection, bone rigidity), so the
correctness claim rests here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sbea_lr_resolve as lr  # noqa: E402

BODYPARTS = ["nose", "left_ear", "right_ear", "neck", "left_front_limb",
             "right_front_limb", "left_hind_limb", "right_hind_limb",
             "left_front_claw", "right_front_claw", "left_hind_claw",
             "right_hind_claw", "back", "root_tail", "mid_tail", "tip_tail"]
N_CAM, T = 4, 40


def triangulate(pts2d, probs, Ps, prob_min):
    rows = []
    for P, (x, y), q in zip(Ps, pts2d, probs):
        if np.isfinite([x, y]).all() and q >= prob_min:
            rows += [x * P[2] - P[0], y * P[2] - P[1]]
    if len(rows) < 4:
        return np.full(3, np.nan)
    X = np.linalg.svd(np.stack(rows))[2][-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-9 else np.full(3, np.nan)


def reproject(X, P):
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


def rig():
    """Four cameras on a ring looking at the origin."""
    K = np.array([[2000.0, 0, 644], [0, 2000.0, 482], [0, 0, 1]])
    Ps = []
    for ang in np.linspace(0, 2 * np.pi, N_CAM, endpoint=False):
        fwd = -np.array([np.cos(ang), np.sin(ang), -0.6])
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(fwd, right), fwd])
        C = np.array([np.cos(ang), np.sin(ang), 0.6]) * 400
        Ps.append(K @ np.hstack([R, (-R @ C)[:, None]]))
    return Ps


def template():
    """A fixed body: bilateral joints offset to either side of the midline."""
    kp = np.zeros((len(BODYPARTS), 3))
    for k, name in enumerate(BODYPARTS):
        side = 18.0 if name.startswith("left") else -18.0 if name.startswith("right") else 0.0
        along = 30.0 if "front" in name or name.endswith("ear") or name == "nose" else -30.0
        kp[k] = [along + (k % 4) * 2.0, side, 20.0 + k * 1.5]   # keep all 16 distinct
    return kp


def scene():
    """(T, K, 3) — the template carried by a rigid motion, so bones stay constant."""
    body = template()
    kp = np.zeros((T, len(BODYPARTS), 3))
    for t in range(T):
        th = t * 0.05
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        kp[t] = body @ R.T + [t * 1.5, 0.0, 0.0]
    return kp


def project(kp3d, Ps):
    """(cam, T, K, 3) — x, y, likelihood."""
    out = np.zeros((N_CAM, kp3d.shape[0], kp3d.shape[1], 3))
    for c, P in enumerate(Ps):
        for t in range(kp3d.shape[0]):
            for k in range(kp3d.shape[1]):
                out[c, t, k, :2] = reproject(kp3d[t, k], P)
        out[c, ..., 2] = 0.99
    return out


def corrupt(kp2d, cams, pairs, frames):
    """Swap the two names of `pairs` in `cams`, over `frames` — the DLC failure."""
    bad = kp2d.copy()
    for a, b in pairs:
        ia, ib = BODYPARTS.index(a), BODYPARTS.index(b)
        for c in cams:
            for t in frames:
                bad[c, t, [ia, ib]] = bad[c, t, [ib, ia]]
    return bad


def _resolve(kp2d):
    return lr.resolve(kp2d, rig(), 0.6, BODYPARTS, triangulate, reproject)[0]


def test_recovers_a_sustained_single_camera_swap():
    Ps, kp3d = rig(), scene()
    clean = project(kp3d, Ps)
    bad = corrupt(clean, [1], lr.SYM_PAIRS, range(10, 30))
    fixed = _resolve(bad)
    ia, ib = BODYPARTS.index("left_hind_claw"), BODYPARTS.index("right_hind_claw")
    # the naming is only defined up to a global left<->right exchange, so accept
    # either the original labelling or its mirror, as long as it is consistent
    got = fixed[:, :, [ia, ib], :2]
    ref = clean[:, :, [ia, ib], :2]
    mirrored = clean[:, :, [ib, ia], :2]
    assert min(np.abs(got - ref).max(), np.abs(got - mirrored).max()) < 1e-6


def test_leaves_uncorrupted_input_alone():
    clean = project(scene(), rig())
    assert np.abs(_resolve(clean) - clean).max() < 1e-6


def test_switch_penalty_suppresses_flicker():
    """lam=0 chases noise. Noise has to be large enough to make the sides genuinely
    ambiguous — that is the regime the real recordings are in."""
    Ps, clean = rig(), project(scene(), rig())
    noisy = clean.copy()
    noisy[..., :2] += np.random.default_rng(1).normal(0, 60.0, noisy[..., :2].shape)
    args = (Ps, 0.6, BODYPARTS, triangulate, reproject)
    loose = lr.resolve(noisy, *args, lam=0.0)[1]
    tight = lr.resolve(noisy, *args, lam=lr.DEFAULT_LAMBDA)[1]
    assert tight < loose


def test_midline_joints_are_never_touched():
    clean = project(scene(), rig())
    bad = corrupt(clean, [2], lr.SYM_PAIRS, range(T))
    fixed = _resolve(bad)
    mid = [BODYPARTS.index(n) for n in ("nose", "neck", "back", "root_tail",
                                        "mid_tail", "tip_tail")]
    assert np.abs(fixed[:, :, mid] - bad[:, :, mid]).max() == 0


@pytest.mark.parametrize("bones", [lr.BONES, lr.CTRL_BONES])
def test_bone_metric_reads_a_rigid_body_as_rigid(bones):
    assert lr.robust_cv(lr.bone_lengths(scene(), bones, BODYPARTS)) < 1e-9


def test_bone_metric_rises_on_a_partial_swap():
    """The metric must respond to the defect it is used to detect.

    Only a *partial* swap is detectable this way: mirroring the whole body leaves
    every bone length identical, which is exactly why the naming is ambiguous up
    to a global exchange. Swapping the claws but not the limbs is the real defect.
    """
    kp3d = scene()
    broken = kp3d.copy()
    for a, b in [("left_front_claw", "right_front_claw"),
                 ("left_hind_claw", "right_hind_claw")]:
        ia, ib = BODYPARTS.index(a), BODYPARTS.index(b)
        broken[:T // 2, [ia, ib]] = broken[:T // 2, [ib, ia]]
    assert lr.robust_cv(lr.bone_lengths(kp3d, lr.BONES, BODYPARTS)) < 1e-9
    assert lr.robust_cv(lr.bone_lengths(broken, lr.BONES, BODYPARTS)) > 0.1
