"""Do the single-session findings hold across every SBeA session at full length?

Built to check a conclusion drawn from two sessions x 200 frames. It did not fully
survive: see the 260728 results in `sbea_lr_resolve`'s docstring and
`docs/conventions.md`. Run this before trusting anything measured on a short clip.

Two traps this encodes, both hit during 260727-28:

  * A NaN 2D coordinate passes `prob < prob_min`, because every comparison against
    NaN is False. That silently poisoned the direction statistic. Everything is
    finite-filtered jointly here and the drop count is printed, so filtering can
    never quietly hide a problem.
  * Spearman alone says whether a radial trend exists but not whether it matters.
    The binned quintile medians are printed next to it for the effect size.

Usage: python scripts/sweep_sbea_sessions.py /node_data/joon/data/sbea_kp3d_full
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbea_dlc_triangulate import BODYPARTS, reproject, solve_all, triangulate  # noqa: E402
import sbea_lr_resolve as lr  # noqa: E402

CTR = np.array([1288 / 2, 964 / 2])
STEP = 5


def spearman(x, y):
    def rk(v):
        o = np.argsort(v)
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        return r
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


agg = {k: [] for k in ("rho", "acos", "scos", "res", "res_fix", "bone", "bone_fix", "q1", "q5")}
per_session = []
for npz in sorted(Path(sys.argv[1]).glob("*-kp3d.npz")):
    d = np.load(npz, allow_pickle=True)
    kp2d = d["keypoints_2d"][:, ::STEP]
    Ps, pm = list(d["P_matrices"]), float(d["prob_min"])
    kp3d, res = solve_all(kp2d, Ps, pm)

    R, E, CS = [], [], []
    raw = 0
    for t in range(kp2d.shape[1]):
        for k in range(kp2d.shape[2]):
            X = kp3d[t, k]
            if not np.isfinite(X).all():
                continue
            for c, P in enumerate(Ps):
                if not (kp2d[c, t, k, 2] >= pm):     # NaN-safe: NaN fails this
                    continue
                uv = kp2d[c, t, k, :2]
                if not np.isfinite(uv).all():
                    continue
                raw += 1
                dv = reproject(X, P) - uv
                e = float(np.linalg.norm(dv))
                r = float(np.linalg.norm(uv - CTR))
                if not np.isfinite(e) or not np.isfinite(r) or e < 1e-9 or r < 1e-9:
                    continue
                R.append(r); E.append(e); CS.append(float(dv @ ((uv - CTR) / r) / e))
    R, E, CS = np.array(R), np.array(E), np.array(CS)
    keep = np.isfinite(R) & np.isfinite(E) & np.isfinite(CS)
    R, E, CS = R[keep], E[keep], CS[keep]

    edges = np.percentile(R, [0, 20, 40, 60, 80, 100])
    q = [float(np.median(E[(R >= lo) & (R <= hi)])) for lo, hi in zip(edges[:-1], edges[1:])]

    kp2d_fix, _ = lr.resolve(kp2d, Ps, pm, BODYPARTS, triangulate, reproject)
    kp3d_fix, res_fix = solve_all(kp2d_fix, Ps, pm)

    def cv(k, b):
        return lr.robust_cv(lr.bone_lengths(k, b, BODYPARTS))

    row = dict(s=npz.name.replace("-kp3d.npz", ""), n=len(R), dropped=raw - len(R),
               rho=spearman(R, E), acos=float(np.abs(CS).mean()), scos=float(CS.mean()),
               res=float(np.nanmedian(res)), res_fix=float(np.nanmedian(res_fix)),
               bone=cv(kp3d, lr.BONES), bone_fix=cv(kp3d_fix, lr.BONES),
               ctrl=cv(kp3d, lr.CTRL_BONES), ctrl_fix=cv(kp3d_fix, lr.CTRL_BONES),
               q1=q[0], q5=q[-1])
    per_session.append(row)
    for k in agg:
        agg[k].append(row[k])
    print(f"{row['s']:<22} n{row['n']:6d} drop{row['dropped']:5d} rho{row['rho']:+.3f} "
          f"|cos|{row['acos']:.3f} scos{row['scos']:+.3f} q1{q[0]:5.1f} q5{q[-1]:5.1f} "
          f"res{row['res']:5.1f}->{row['res_fix']:5.1f} bone{row['bone']:.3f}->{row['bone_fix']:.3f}",
          flush=True)

print("\n=== 20-session summary (median [min, max]) ===")
for k, label in (("rho", "Spearman(radius, residual)"), ("acos", "mean |cos|  (isotropic 0.64)"),
                 ("scos", "mean signed cos (radial=+-1)"), ("q1", "residual, inner radius quintile"),
                 ("q5", "residual, outer radius quintile"), ("res", "reproj px"),
                 ("res_fix", "reproj px after L/R"), ("bone", "bone rCV"),
                 ("bone_fix", "bone rCV after L/R")):
    v = np.array(agg[k])
    print(f"  {label:<32} {np.median(v):+7.3f}  [{v.min():+.3f}, {v.max():+.3f}]")

imp = sum(r["bone_fix"] < r["bone"] for r in per_session)
worse = sum(r["bone_fix"] > r["bone"] for r in per_session)
moved = sum(abs(r["ctrl_fix"] - r["ctrl"]) > 1e-9 for r in per_session)
print(f"\n  bone rCV improved by L/R : {imp}/{len(per_session)}")
print(f"  bone rCV WORSENED by L/R : {worse}/{len(per_session)}")
print(f"  midline control moved    : {moved}/{len(per_session)} (must be 0)")
print(f"  reproj improved by L/R   : {sum(r['res_fix'] < r['res'] for r in per_session)}"
      f"/{len(per_session)}")
