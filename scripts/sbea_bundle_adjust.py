"""S2': does the missing bundle-adjustment step explain our 24.7 px reprojection residual?

The author pipeline ships `bundle_adjustment.pyd`; ours does not — we take the released
`cam_mat_all` as fixed constants and solve linear DLT. That gap is the leading candidate
for the residual floor (see vault 260816_behaviorlab_sbea_author_baseline_absent).

The trap this script is built around
------------------------------------
Reprojection error is the objective. Adding free camera parameters *will* lower it, and
that proves nothing — the same mistake was made three times in 260728 (docs
sbea_social_and_sdannce_status.md section 6). Two guards, both mandatory:

1. Independent metric — bone-length rCV (near-rigid segments) and a midline control.
   Real geometry improvement must carry the physics with it. Reprojection alone falling
   while rCV worsens is overfitting, and the verdict is FAIL.
2. Held-out sessions — the correction is fit on one subset and scored on sessions that
   never entered the fit. This is possible because the rig is shared: the 20 sessions
   form exactly two calibration groups by recording date (10 x 20221108, 10 x 20221109),
   verified by comparing the stored P_matrices.

Model
-----
Extrinsics only: P = K [R | t] by RQ decomposition, then optimise a rotation vector and
translation offset per camera with K fixed and camera 0 frozen (gauge). 18 free
parameters against ~10^5 observations per session. Intrinsics stay untouched because the
author calibrated distortion out with Zhang's method at the checkerboard stage, so the
projective part is not ours to refit — and letting P float freely admits a projective
homography that keeps reprojection low while destroying bone lengths.

3D points are never free parameters: they are re-solved by DLT from the current cameras,
so the optimiser cannot buy residual by moving points independently.

Usage
-----
    python scripts/sbea_bundle_adjust.py <kp3d_dir> [--fit-frac 0.5] [--max-frames 400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbea_lr_resolve as lr  # noqa: E402  (BONES / CTRL_BONES / rCV are the SSOT there)


def decompose(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P (3,4) -> K, R, t with positive diagonal K and det(R) = +1."""
    K, R = rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    if np.linalg.det(R) < 0:
        R = -R
        K = -K
    t = np.linalg.inv(K) @ P[:, 3]
    return K / K[2, 2], R, t / K[2, 2]


def compose(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t[:, None]])


def apply_offsets(base: list[tuple], theta: np.ndarray) -> np.ndarray:
    """Camera 0 frozen; cameras 1.. get a rotvec + translation offset from theta."""
    Ps = [compose(*base[0])]
    for c, (K, R, t) in enumerate(base[1:]):
        dr, dt = theta[6 * c:6 * c + 3], theta[6 * c + 3:6 * c + 6]
        Ps.append(compose(K, Rotation.from_rotvec(dr).as_matrix() @ R, t + dt))
    return np.stack(Ps)


def triangulate_batch(xy: np.ndarray, mask: np.ndarray, Ps: np.ndarray) -> np.ndarray:
    """Vectorised linear DLT. xy (N,C,2), mask (N,C) -> X (N,3), NaN where < 2 views.

    Masked-out views contribute all-zero rows, which is what dropping them does to the
    least-squares system — so the batch keeps a fixed shape without changing the answer.
    """
    N, C = mask.shape
    x, y = np.nan_to_num(xy[..., 0]), np.nan_to_num(xy[..., 1])
    rows = np.empty((N, 2 * C, 4))
    rows[:, 0::2] = x[..., None] * Ps[None, :, 2] - Ps[None, :, 0]
    rows[:, 1::2] = y[..., None] * Ps[None, :, 2] - Ps[None, :, 1]
    rows *= np.repeat(mask, 2, axis=1)[..., None]
    X = np.linalg.svd(rows)[2][:, -1]
    w = X[:, 3:]
    out = np.where(np.abs(w) > 1e-9, X[:, :3] / np.where(w == 0, 1, w), np.nan)
    return np.where((mask.sum(1) >= 2)[:, None], out, np.nan)


def residuals(xy: np.ndarray, mask: np.ndarray, Ps: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-view reprojection error (N,C); NaN where the view is masked or X is invalid."""
    h = np.einsum("cij,nj->nci", Ps, np.concatenate([X, np.ones((len(X), 1))], axis=1))
    proj = h[..., :2] / np.where(np.abs(h[..., 2:]) > 1e-9, h[..., 2:], np.nan)
    err = np.linalg.norm(proj - xy, axis=2)
    return np.where(mask, err, np.nan)


def load(path: Path, max_frames: int | None, seed: int = 0):
    d = np.load(path)
    kp2d, Ps = d["keypoints_2d"], d["P_matrices"]          # (C,T,K,3), (C,3,4)
    names = [str(s) for s in d["keypoint_names"]]
    pmin = float(d["prob_min"])
    T = kp2d.shape[1]
    if max_frames and T > max_frames:
        sel = np.random.default_rng(seed).choice(T, max_frames, replace=False)
        sel.sort()
        kp2d = kp2d[:, sel]
    C, T, K = kp2d.shape[:3]
    xy = kp2d[..., :2].transpose(1, 2, 0, 3).reshape(T * K, C, 2)
    mask = (kp2d[..., 2].transpose(1, 2, 0).reshape(T * K, C) >= pmin) & np.isfinite(xy).all(2)
    return xy, mask, Ps, names, (T, K)


def score(xy, mask, Ps, names, shape) -> dict:
    X = triangulate_batch(xy, mask, Ps)
    res = residuals(xy, mask, Ps, X)
    kp3d = X.reshape(*shape, 3)
    cv = lambda b: lr.robust_cv(lr.bone_lengths(kp3d, b, names))
    return {"reproj_px": float(np.nanmedian(res)),
            "bone_rcv": cv(lr.BONES), "ctrl_rcv": cv(lr.CTRL_BONES),
            "per_joint": np.nanmedian(res.reshape(*shape, -1), axis=(0, 2))}


def split_by_side(per_joint: np.ndarray, names: list[str]) -> tuple[float, float]:
    """Midline vs bilateral joint residual — the 19.9 / 27.8 px split from 260727."""
    side = np.array([n.startswith(("left_", "right_")) for n in names])
    return float(np.nanmedian(per_joint[~side])), float(np.nanmedian(per_joint[side]))


def fit(sessions: list[tuple], base: list[tuple]) -> np.ndarray:
    """Least-squares over the pooled reprojection residuals of the fit sessions."""
    def f(theta):
        Ps = apply_offsets(base, theta)
        out = []
        for xy, mask, *_ in sessions:
            X = triangulate_batch(xy, mask, Ps)
            out.append(np.nan_to_num(residuals(xy, mask, Ps, X)).ravel())
        return np.concatenate(out)

    r = least_squares(f, np.zeros(6 * (len(base) - 1)), loss="soft_l1", f_scale=10.0,
                      x_scale="jac", max_nfev=120, verbose=0)
    return r.x


def group_key(Ps: np.ndarray) -> str:
    return f"{np.round(Ps, 6).tobytes().__hash__():x}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kp3d_dir", type=Path)
    ap.add_argument("--max-frames", type=int, default=400,
                    help="frames sampled per session for fitting (eval uses the same set)")
    ap.add_argument("--fit-frac", type=float, default=0.5,
                    help="fraction of each calibration group used to fit; rest is held out")
    a = ap.parse_args()

    files = sorted(a.kp3d_dir.glob("*-kp3d.npz"))
    if not files:
        sys.exit(f"no *-kp3d.npz under {a.kp3d_dir}")

    loaded = {f: load(f, a.max_frames) for f in files}
    groups: dict[str, list[Path]] = {}
    for f, (_, _, Ps, _, _) in loaded.items():
        groups.setdefault(group_key(Ps), []).append(f)
    print(f"{len(files)} sessions -> {len(groups)} calibration group(s): "
          f"{[len(v) for v in groups.values()]}")

    for gi, (_, gfiles) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        n_fit = max(1, int(round(len(gfiles) * a.fit_frac)))
        fit_f, held_f = gfiles[:n_fit], gfiles[n_fit:]
        base = [decompose(P) for P in loaded[fit_f[0]][2]]
        print(f"\n=== group {gi}: fit {len(fit_f)} / held-out {len(held_f)} ===")
        print(f"    fit      = {[f.name.split('-kp3d')[0] for f in fit_f]}")
        print(f"    held-out = {[f.name.split('-kp3d')[0] for f in held_f]}")
        if not held_f:
            print("    SKIP — no held-out session, the test would be unfalsifiable")
            continue

        theta = fit([loaded[f] for f in fit_f], base)
        Ps_ba = apply_offsets(base, theta)
        print(f"    offsets: |rot| {np.linalg.norm(theta[0::6]):.4f} rad, "
              f"|trans| {np.linalg.norm(theta.reshape(-1, 6)[:, 3:]):.4f}")

        for tag, fs in (("FIT", fit_f), ("HELD-OUT", held_f)):
            print(f"    {tag}")
            pj = []
            for f in fs:
                xy, mask, Ps0, names, shape = loaded[f]
                b = score(xy, mask, Ps0, names, shape)
                c = score(xy, mask, Ps_ba, names, shape)
                pj.append((b["per_joint"], c["per_joint"]))
                verdict = ("PASS" if c["reproj_px"] < b["reproj_px"] and
                           c["bone_rcv"] <= b["bone_rcv"] + 1e-4 else "FAIL")
                print(f"      {f.name.split('-kp3d')[0]:<22} "
                      f"reproj {b['reproj_px']:6.2f} -> {c['reproj_px']:6.2f} px | "
                      f"bone rCV {b['bone_rcv']:.3f} -> {c['bone_rcv']:.3f} | "
                      f"ctrl {b['ctrl_rcv']:.3f} -> {c['ctrl_rcv']:.3f} | {verdict}")
            names = loaded[fs[0]][3]
            mb, sb = split_by_side(np.median([p[0] for p in pj], axis=0), names)
            ma, sa = split_by_side(np.median([p[1] for p in pj], axis=0), names)
            print(f"      {'-> midline / bilateral':<22} "
                  f"{mb:5.1f} / {sb:5.1f}  ->  {ma:5.1f} / {sa:5.1f} px")

    print("\nVerdict rule: reprojection down AND bone rCV not worse. Reprojection alone "
          "is the objective and cannot judge itself.")


if __name__ == "__main__":
    main()
