"""R1: does the L/R relabelling work once bundle adjustment has removed the calibration error?

260728 judged the relabelling a failure (bone rCV -2%, 6/20 sessions worse). That verdict was
measured with calibration drift still in the residual. After BA the picture changed: midline
joints fell 19-21 -> 10.7 px while bilateral joints barely moved (25-26 -> 22 px), so L/R is now
the largest remaining component. This re-runs the relabelling on BA-corrected cameras.

Pre-registered before the first run (the 260728 failure was a violation of exactly this):

1. lam is selected on the FIT sessions ONLY, and selected by BONE rCV — never by reprojection.
   260728 picked lam by the objective on the very sessions it then reported.
2. The verdict is read on HELD-OUT sessions that took no part in the BA fit or the lam choice.
3. PASS requires bone rCV to improve on held-out AND the midline control not to degrade.
   Reprojection is reported but is NOT a criterion — it is the relabelling's own objective.

Usage:
    python scripts/sbea_lr_after_ba.py <kp3d_dir> [--max-frames 400] [--lams 0,5,10,20,40]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbea_bundle_adjust as ba  # noqa: E402
import sbea_lr_resolve as lr  # noqa: E402
from sbea_dlc_triangulate import reproject, triangulate  # noqa: E402


def rcv(kp2d, Ps, names, prob_min, bones) -> float:
    """Bone-length robust CV after triangulating kp2d with cameras Ps."""
    C, T, K = kp2d.shape[:3]
    xy = kp2d[..., :2].transpose(1, 2, 0, 3).reshape(T * K, C, 2)
    mask = (kp2d[..., 2].transpose(1, 2, 0).reshape(T * K, C) >= prob_min) & np.isfinite(xy).all(2)
    X = ba.triangulate_batch(xy, mask, Ps).reshape(T, K, 3)
    return lr.robust_cv(lr.bone_lengths(X, bones, names))


def costs_and_scores(f, max_frames, Ps, lams):
    """Pattern costs are lam-independent, so pay for them once and sweep lam for free."""
    d = np.load(f)
    kp2d, pmin = d["keypoints_2d"], float(d["prob_min"])
    names = [str(s) for s in d["keypoint_names"]]
    T = kp2d.shape[1]
    sel = np.random.default_rng(0).choice(T, min(T, max_frames), replace=False)
    sel.sort()
    kp2d = kp2d[:, sel]

    base = {b: rcv(kp2d, Ps, names, pmin, bones) for b, bones in
            (("bone", lr.BONES), ("ctrl", lr.CTRL_BONES))}
    pair_cost = {}
    for a, b in lr.SYM_PAIRS:
        ia, ib = names.index(a), names.index(b)
        pair_cost[(ia, ib)] = lr.pair_costs(kp2d, Ps, ia, ib, pmin, triangulate, reproject)

    out = {}
    for lam in lams:
        k = kp2d.copy()
        for (ia, ib), cost in pair_cost.items():
            path, usable = lr.viterbi(cost, lam)
            for t in np.nonzero(usable)[0]:
                for c in range(k.shape[0]):
                    if (path[t] >> c) & 1:
                        k[c, t, [ia, ib]] = k[c, t, [ib, ia]]
        out[lam] = {b: rcv(k, Ps, names, pmin, bones) for b, bones in
                    (("bone", lr.BONES), ("ctrl", lr.CTRL_BONES))}
    return base, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kp3d_dir", type=Path)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--lams", default="0,5,10,20,40")
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(",")]

    files = sorted(a.kp3d_dir.glob("*-kp3d.npz"))
    loaded = {f: ba.load(f, a.max_frames) for f in files}
    groups: dict[str, list[Path]] = {}
    for f, (_, _, Ps, _, _) in loaded.items():
        groups.setdefault(ba.group_key(Ps), []).append(f)

    for gi, (_, gfiles) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        n_fit = max(1, len(gfiles) // 2)
        fit_f, held_f = gfiles[:n_fit], gfiles[n_fit:]
        base_cams = [ba.decompose(P) for P in loaded[fit_f[0]][2]]
        Ps = ba.apply_offsets(base_cams, ba.fit([loaded[f] for f in fit_f], base_cams))
        print(f"\n=== group {gi}: fit {len(fit_f)} / held-out {len(held_f)}, BA applied ===")

        scored = {tag: [costs_and_scores(f, a.max_frames, Ps, lams) for f in fs]
                  for tag, fs in (("FIT", fit_f), ("HELD-OUT", held_f))}

        # step 1 — choose lam on FIT, by bone rCV only
        med = {lam: float(np.median([s[lam]["bone"] - b["bone"] for b, s in scored["FIT"]]))
               for lam in lams}
        best = min(med, key=med.get)
        print("    FIT bone rCV delta by lam: " +
              "  ".join(f"{lam:g}:{med[lam]:+.4f}" for lam in lams) + f"  -> lam={best:g}")
        if med[best] >= 0:
            # Every lam made the fit sessions worse. Picking the least-bad and then reading
            # held-out would be selection on noise, so stop before the verdict is even framed.
            print(f"    NO-GO — no lam improves bone rCV on FIT (best {med[best]:+.4f}). "
                  "Held-out not read.")
            continue

        # step 2 — read the verdict on HELD-OUT only
        wins = ctrl_bad = 0
        for f, (bse, sc) in zip(held_f, scored["HELD-OUT"]):
            db = sc[best]["bone"] - bse["bone"]
            dc = sc[best]["ctrl"] - bse["ctrl"]
            wins += db < 0
            ctrl_bad += dc > 0.005
            print(f"      {f.name.split('-kp3d')[0]:<22} bone {bse['bone']:.3f} -> "
                  f"{sc[best]['bone']:.3f} ({db:+.3f}) | ctrl {bse['ctrl']:.3f} -> "
                  f"{sc[best]['ctrl']:.3f} ({dc:+.3f})")
        dmed = float(np.median([sc[best]["bone"] - b["bone"] for b, sc in scored["HELD-OUT"]]))
        verdict = "PASS" if dmed < 0 and wins > len(held_f) / 2 and not ctrl_bad else "FAIL"
        print(f"    HELD-OUT lam={best:g}: bone rCV median {dmed:+.4f}, "
              f"improved {wins}/{len(held_f)}, control degraded {ctrl_bad} -> {verdict}")

        # Robustness to the lam choice. The FIT deltas that pick lam are ~1e-3, i.e. the same
        # order as noise, so a verdict that only holds at the selected lam is a selection
        # artefact. If every lam moves held-out the same way, the unstable pick does not matter.
        alllam = {lam: float(np.median([sc[lam]["bone"] - b["bone"]
                                        for b, sc in scored["HELD-OUT"]])) for lam in lams}
        print("    HELD-OUT by lam: " + "  ".join(f"{lam:g}:{alllam[lam]:+.4f}" for lam in lams) +
              ("  -> robust (전 lam 동일 방향)" if len({v < 0 for v in alllam.values()}) == 1
               else "  -> FRAGILE (lam 에 따라 부호가 바뀐다)"))

    print("\nPASS requires held-out bone rCV improvement with the midline control flat. "
          "Reprojection is the relabelling's objective and is not a criterion.")


if __name__ == "__main__":
    main()
