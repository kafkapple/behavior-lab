"""Why is the SBeA reprojection residual ~25 px? Distortion, or 2D detector noise?

The 260727 handoff assumed lens distortion (SBeA ships P matrices only — no
distortion coefficients, verified: caliParas.mat holds cam_mat_all/rotation/
translation and nothing else). That assumption was never tested. It is testable
from the saved npz alone, because the two candidates leave different signatures:

  radial distortion  → residual grows with radius from the principal point AND
                       the residual vector points along the radial direction
  2D detector noise  → residual flat in radius, direction isotropic, and larger
                       on the joints DLC is least sure about

Run with no args to use the default rec10-M2 npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def decompose(P: np.ndarray):
    """P (3,4) → K, R, t. cv2 returns t in homogeneous camera-center form."""
    K, R, C = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2, 2]
    C = (C[:3] / C[3]).ravel()
    return K, R, -R @ C


def reproject(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


def triangulate(pts: np.ndarray, ok: np.ndarray, Ps) -> np.ndarray:
    rows = []
    for P, (x, y), good in zip(Ps, pts, ok):
        if good:
            rows += [x * P[2] - P[0], y * P[2] - P[1]]
    if len(rows) < 4:
        return np.full(3, np.nan)
    X = np.linalg.svd(np.stack(rows))[2][-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-9 else np.full(3, np.nan)


def collect(kp2d: np.ndarray, Ps, prob_min: float):
    """One row per (cam, frame, joint) observation that took part in a solve."""
    C, T, K = kp2d.shape[:3]
    obs = []
    for t in range(T):
        for k in range(K):
            ok = kp2d[:, t, k, 2] >= prob_min
            if ok.sum() < 2:
                continue
            X = triangulate(kp2d[:, t, k, :2], ok, Ps)
            if not np.isfinite(X).all():
                continue
            for c in range(C):
                if not ok[c]:
                    continue
                uv = kp2d[c, t, k, :2]
                obs.append((c, k, *uv, *(reproject(X, Ps[c]) - uv), kp2d[c, t, k, 2]))
    return np.array(obs)  # (N, 7): cam, joint, u, v, du, dv, prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=Path(
        "/mnt/d/data/derived/mac_backups_260813/gpu03_nonM5_260813/sbea_kp3d/rec10-M2-20221108-kp3d.npz"))
    ap.add_argument("--width", type=int, default=1288)
    ap.add_argument("--height", type=int, default=964)
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    kp2d, Ps = d["keypoints_2d"], d["P_matrices"]
    names = [str(n) for n in d["keypoint_names"]]
    prob_min = float(d["prob_min"])
    print(f"{a.npz.name}: kp2d {kp2d.shape}, prob_min {prob_min}, "
          f"stored median residual {np.nanmedian(d['reproj_residual_px']):.2f} px\n")

    print("== per-camera intrinsics (RQ decomposition of P) ==")
    for c, P in enumerate(Ps):
        K, _, _ = decompose(P)
        print(f"  cam{c}: f=({K[0,0]:8.1f}, {K[1,1]:8.1f})  "
              f"c=({K[0,2]:7.1f}, {K[1,2]:6.1f})  skew={K[0,1]:6.2f}")
    print(f"  image center for reference: ({a.width/2:.1f}, {a.height/2:.1f})\n")

    obs = collect(kp2d, Ps, prob_min)
    cam, joint = obs[:, 0].astype(int), obs[:, 1].astype(int)
    uv, duv, prob = obs[:, 2:4], obs[:, 4:6], obs[:, 6]
    err = np.linalg.norm(duv, axis=1)
    print(f"{len(obs)} observations, median residual {np.median(err):.2f} px\n")

    # -- test 1: does the residual grow with radius? -------------------------
    ctr = np.array([a.width / 2, a.height / 2])
    rad = np.linalg.norm(uv - ctr, axis=1)
    edges = np.percentile(rad, [0, 20, 40, 60, 80, 100])
    print("== test 1: residual vs radius from image center ==")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rad >= lo) & (rad <= hi)
        print(f"  r {lo:6.1f}-{hi:6.1f} px  n={m.sum():5d}  "
              f"median {np.median(err[m]):6.2f} px")
    print(f"  Spearman(r, residual) = {spearman(rad, err):+.3f}\n")

    # -- test 2: is the residual pointing radially? --------------------------
    # A radial distortion error is (anti)parallel to the radial direction; a
    # detector error has no preferred direction, so |cos| averages ~0.64 for a
    # uniform angle and the signed mean sits at 0.
    radial_unit = (uv - ctr) / np.maximum(rad, 1e-6)[:, None]
    cos = (duv * radial_unit).sum(axis=1) / np.maximum(err, 1e-6)
    print("== test 2: residual direction vs radial direction ==")
    print(f"  mean signed cos = {cos.mean():+.3f}  (distortion → |.| near 1)")
    print(f"  mean |cos|      = {np.abs(cos).mean():.3f}   (isotropic → 0.64)")
    for c in range(len(Ps)):
        m = cam == c
        print(f"    cam{c}: signed {cos[m].mean():+.3f}  |cos| {np.abs(cos[m]).mean():.3f}")
    print()

    # -- test 3: is the residual worst where DLC is least confident? ---------
    print("== test 3: residual per joint (sorted worst first) ==")
    rows = sorted(
        ((np.median(err[joint == k]), np.median(prob[joint == k]), (joint == k).sum(), names[k])
         for k in range(len(names)) if (joint == k).any()), reverse=True)
    for e, p, n, nm in rows:
        print(f"  {nm:<18} median {e:6.2f} px   likelihood {p:.3f}   n={n}")
    js = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    print(f"  Spearman(joint likelihood, joint residual) = {spearman(js[1], js[0]):+.3f}\n")

    swap_test(kp2d, Ps, names, prob_min)
    joint_vs_perpair(kp2d, Ps, names, prob_min)


# Test 3 leaves the limbs unexplained: DLC is *confident* and still inconsistent
# across views. The classic cause is left/right identity — a single-view detector
# labels the near limb "left" regardless of the animal's heading, so two cameras
# facing opposite sides disagree about which claw is which. That is a per-camera
# per-frame binary, so it is searchable: try both assignments for each camera and
# see whether a non-identity pattern reprojects better. NON_SYM is the control —
# the same 16-hypothesis search over a pair that cannot be swapped, which bounds
# how much of any gain is just the search overfitting.
SYM_PAIRS = [("left_ear", "right_ear"),
             ("left_front_limb", "right_front_limb"),
             ("left_hind_limb", "right_hind_limb"),
             ("left_front_claw", "right_front_claw"),
             ("left_hind_claw", "right_hind_claw")]
NON_SYM = [("nose", "root_tail"), ("neck", "back")]


def swap_test(kp2d: np.ndarray, Ps, names, prob_min: float) -> None:
    print("== test 4: does a per-camera left/right swap reprojct better? ==")
    print(f"  {'pair':<34} {'identity':>9} {'best-of-16':>11} {'swap wins':>10}")
    for group, label in ((SYM_PAIRS, "symmetric"), (NON_SYM, "CONTROL")):
        for a_name, b_name in group:
            ia, ib = names.index(a_name), names.index(b_name)
            base, best, wins, n = [], [], 0, 0
            for t in range(kp2d.shape[1]):
                r_id, r_best, pat = pair_residual(kp2d, Ps, t, ia, ib, prob_min)
                if r_id is None:
                    continue
                base.append(r_id)
                best.append(r_best)
                wins += pat != 0
                n += 1
            if not n:
                continue
            tag = f"{a_name}/{b_name}" + ("" if label == "symmetric" else "  [CONTROL]")
            print(f"  {tag:<34} {np.median(base):8.2f}p {np.median(best):10.2f}p "
                  f"{wins/n:9.0%}")


def joint_vs_perpair(kp2d, Ps, names, prob_min: float) -> None:
    """Whole-animal flip (one pattern for all 5 pairs) vs per-pair freedom.

    Physically the confusion is whole-animal: a single-view detector commits to
    one body orientation, so every symmetric pair should flip together. Per-pair
    freedom is 5x the hypotheses and can produce anatomically impossible mixes.
    If the constrained version captures most of the gain, prefer it.
    """
    C = kp2d.shape[0]
    idx = [(names.index(a), names.index(b)) for a, b in SYM_PAIRS]
    base, joint, perpair = [], [], []
    for t in range(kp2d.shape[1]):
        per_pattern = np.zeros(1 << C)
        pair_best, pair_id, usable = [], [], True
        for ia, ib in idx:
            errs = [pair_residual(kp2d, Ps, t, ia, ib, prob_min, pattern=p)
                    for p in range(1 << C)]
            if any(e is None for e in errs):
                usable = False
                break
            per_pattern += np.array(errs)
            pair_id.append(errs[0])
            pair_best.append(min(errs))
        if not usable:
            continue
        # all three must aggregate the same way — the joint score is a sum over
        # pairs, so compare means, not medians, or the outlier pairs skew it
        base.append(np.mean(pair_id))
        joint.append(per_pattern.min() / len(idx))
        perpair.append(np.mean(pair_best))
    print("\n== test 5: whole-animal flip vs per-pair flip (median over frames) ==")
    print(f"  identity            {np.median(base):6.2f} px")
    print(f"  whole-animal (16 h) {np.median(joint):6.2f} px   <- 4 DOF")
    print(f"  per-pair     (80 h) {np.median(perpair):6.2f} px   <- 20 DOF")


def pair_residual(kp2d, Ps, t, ia, ib, prob_min, pattern=None):
    """Residual of the (a, b) pair at frame t.

    With `pattern` given, returns that one flip pattern's median residual (or
    None if the frame is unusable). Without it, returns
    (identity, best-of-16, winning pattern).
    """
    C = kp2d.shape[0]
    ok = (kp2d[:, t, ia, 2] >= prob_min) & (kp2d[:, t, ib, 2] >= prob_min)
    if ok.sum() < 3:                       # need 3+ views or the search is trivial
        return None if pattern is not None else (None, None, None)
    results = []
    for p in range(1 << C) if pattern is None else (pattern,):
        flip = [(p >> c) & 1 for c in range(C)]
        errs = []
        for src, dst in ((ia, ib), (ib, ia)):
            pts = np.stack([kp2d[c, t, dst if flip[c] else src, :2] for c in range(C)])
            X = triangulate(pts, ok, Ps)
            if not np.isfinite(X).all():
                errs = None
                break
            errs += [np.linalg.norm(reproject(X, Ps[c]) - pts[c]) for c in range(C) if ok[c]]
        results.append(np.median(errs) if errs else np.inf)
    if pattern is not None:
        return results[0]
    best = int(np.argmin(results))
    # patterns 0 and all-ones are the same labelling, just with a/b names exchanged
    normalized = 0 if best in (0, (1 << C) - 1) else best
    return results[0], results[best], normalized


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v)
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v))
    return r


if __name__ == "__main__":
    main()
