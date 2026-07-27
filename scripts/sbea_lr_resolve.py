"""Bilateral (left/right) joint relabelling for SBeA, and the checks that judge it.

The released SBeA model is a plain DLC ResNet-50: independent per-part heatmaps,
no skeleton constraint. It therefore names each side from single-view appearance
alone, and two cameras watching opposite flanks disagree about which claw is
"left". The 260727 diagnostic measured this (scripts/diag_sbea_residual.py): a
per-camera swap reprojects better on 73-88% of frames for bilateral pairs, versus
9-15% for non-swappable control pairs.

Fixing it is less simple than finding it, and the failures are worth stating:

  * Choosing the best swap per frame independently *does* cut reprojection error,
    but it flickers — 99% of frames get relabelled and the 3D trajectory becomes
    less coherent. Reprojection is the objective being minimised, so it cannot
    judge its own result.
  * Frame-to-frame 3D step size is not a fair judge either, in the opposite
    direction: a mislabelled joint triangulates to a midline compromise, which is
    smooth *because* it is wrong. The metric rewards the artifact.
  * Bone length is the honest check. Limb segments are near-rigid, so a wrong
    assignment stretches them however the point moved. Under Viterbi smoothing
    the robust CV of the bilateral segments falls 0.265 -> 0.183 (rec10-M2) and
    0.237 -> 0.199 (rec10-M7) at lam=20, while the midline control is unchanged.

So this helps, but partially: it does not bring the residual down to the ~19-20 px
floor the midline joints already sit at, and the left/right *length symmetry*
check does not improve consistently across the two sessions. Treat SBeA 3D as
usable for SAM2 prompting, not for quantitative limb kinematics.

lam=20 was chosen as the value that improves both sessions rather than either
session's own optimum (M2 prefers 160, M7 prefers 20).
"""
from __future__ import annotations

import numpy as np

SYM_PAIRS = [("left_ear", "right_ear"),
             ("left_front_limb", "right_front_limb"),
             ("left_hind_limb", "right_hind_limb"),
             ("left_front_claw", "right_front_claw"),
             ("left_hind_claw", "right_hind_claw")]

# near-rigid segments spanning the bilateral joints, and midline segments as control
BONES = [("left_front_limb", "left_front_claw"), ("right_front_limb", "right_front_claw"),
         ("left_hind_limb", "left_hind_claw"), ("right_hind_limb", "right_hind_claw"),
         ("neck", "left_ear"), ("neck", "right_ear")]
CTRL_BONES = [("nose", "neck"), ("neck", "back"), ("back", "root_tail"),
              ("root_tail", "mid_tail"), ("mid_tail", "tip_tail")]

DEFAULT_LAMBDA = 20.0


def _hamming(n_cam: int) -> np.ndarray:
    return np.array([[bin(p ^ q).count("1") for q in range(1 << n_cam)]
                     for p in range(1 << n_cam)], dtype=np.float64)


def pair_costs(kp2d, Ps, ia, ib, prob_min, triangulate, reproject) -> np.ndarray:
    """(T, 2**C) reprojection cost of each per-camera swap pattern. NaN row = skip."""
    C, T = kp2d.shape[0], kp2d.shape[1]
    cost = np.full((T, 1 << C), np.nan)
    for t in range(T):
        ok = (kp2d[:, t, ia, 2] >= prob_min) & (kp2d[:, t, ib, 2] >= prob_min)
        if ok.sum() < 3:            # under three views the search is unconstrained
            continue
        for p in range(1 << C):
            flip = [(p >> c) & 1 for c in range(C)]
            errs = []
            for src, dst in ((ia, ib), (ib, ia)):
                pts = np.stack([kp2d[c, t, dst if flip[c] else src, :2] for c in range(C)])
                X = triangulate(pts, ok.astype(np.float64), Ps, 0.5)
                if not np.isfinite(X).all():
                    errs = None
                    break
                errs += [float(np.linalg.norm(reproject(X, Ps[c]) - pts[c]))
                         for c in range(C) if ok[c]]
            if errs:
                cost[t, p] = float(np.median(errs))
    return cost


def viterbi(cost: np.ndarray, lam: float):
    """Cheapest pattern path; switching a camera's assignment costs `lam`."""
    T, S = cost.shape
    usable = ~np.isnan(cost).all(axis=1)
    if not usable.any():
        return np.zeros(T, dtype=int), usable
    fill = float(np.nanmax(cost[usable]))
    emit = np.where(np.isnan(cost), fill, cost)
    hamm = _hamming(int(np.log2(S)))
    dp = np.empty_like(emit)
    bk = np.zeros(emit.shape, dtype=int)
    dp[0] = emit[0]
    for t in range(1, T):
        tot = dp[t - 1][:, None] + lam * hamm
        bk[t] = tot.argmin(axis=0)
        dp[t] = tot.min(axis=0) + emit[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(dp[-1].argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bk[t, path[t]]
    return path, usable


def resolve(kp2d, Ps, prob_min, bodyparts, triangulate, reproject,
            lam: float = DEFAULT_LAMBDA):
    """Return relabelled copy of kp2d plus the number of pattern switches used."""
    out = kp2d.copy()
    C = out.shape[0]
    switches = 0
    for a, b in SYM_PAIRS:
        ia, ib = bodyparts.index(a), bodyparts.index(b)
        cost = pair_costs(kp2d, Ps, ia, ib, prob_min, triangulate, reproject)
        path, usable = viterbi(cost, lam)
        switches += int((np.diff(path) != 0).sum())
        for t in range(out.shape[1]):
            if not usable[t]:
                continue
            for c in range(C):
                if (path[t] >> c) & 1:
                    out[c, t, [ia, ib]] = out[c, t, [ib, ia]]
    return out, switches


def bone_lengths(kp3d: np.ndarray, bones, bodyparts) -> np.ndarray:
    idx = [(bodyparts.index(a), bodyparts.index(b)) for a, b in bones]
    return np.stack([np.linalg.norm(kp3d[:, a] - kp3d[:, b], axis=1) for a, b in idx])


def robust_cv(L: np.ndarray) -> float:
    """Median over segments of MAD/median — how rigid the segments actually are."""
    med = np.nanmedian(L, axis=1)
    mad = np.nanmedian(np.abs(L - med[:, None]), axis=1)
    return float(np.nanmedian(mad / med))


def report(kp3d_before, kp3d_after, bodyparts) -> str:
    """One line: does the relabelling make the segments more rigid, control flat?"""
    def cv(k, bones):
        return robust_cv(bone_lengths(k, bones, bodyparts))
    return (f"bone rCV {cv(kp3d_before, BONES):.3f} -> {cv(kp3d_after, BONES):.3f}"
            f" | midline control {cv(kp3d_before, CTRL_BONES):.3f} -> "
            f"{cv(kp3d_after, CTRL_BONES):.3f} (must not move)")
