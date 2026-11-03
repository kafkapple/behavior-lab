"""Agreement of two keypoint results (model A, model B) on the same video, with no ground truth: it measures how much they differ, not which is right.
2D: per camera and node distance in rig-cell pixels with a block-bootstrap CI. 3D: distance after a similarity fit of B onto A. No web code."""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats as sps

import core

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THR, BLOCK, NBOOT = 0.25, 20, 1000        # recon3D keeps a 2D point above 0.25; blocks keep frame autocorrelation; seed fixed below
A_COL, B_COL = "#5ac8fa", "#ff9f0a"        # same blue/orange as the group plots
CAM_COL = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
ALIAS = {"nose": "nose1", "ass": "tailstart1", "rarm": "forelegR1", "larm": "forelegL1", "rleg": "hindlegR1", "lleg": "hindlegL1"}   # server YOLO node -> SLEAP node: an ASSUMPTION about body parts (box centre vs keypoint), see ALIAS_NOTE
ALIAS_NOTE = " YOLO 의 node 이름을 SLEAP 의 대응 부위로 가정해 비교했다(nose, ass↔tailstart, arm↔foreleg, leg↔hindleg). box 중심과 keypoint 는 같은 지점이 아닐 수 있고 좌우 규약도 가정이다."
CAVEAT = ("ground truth 없음: 거리는 두 결과의 차이이며 정확도가 아니다. 어느 모델이 맞는지 말하지 않는다. "
          "Wilcoxon p 는 프레임·카메라를 독립으로 놓아 낙관적이므로 참고용이다. confidence 는 모델마다 척도가 달라 비교가 제한적이다.")


def _boot_ci(d: np.ndarray, rng: np.random.Generator) -> list[float | None]:
    """95 % CI of the median of d (T, ...) with NaN = missing, resampling blocks of frames."""
    T = len(d)
    if np.isfinite(d).sum() < 10:
        return [None, None]
    nb = -(-T // BLOCK)
    idx = (rng.integers(0, max(1, T - BLOCK + 1), (NBOOT, nb))[..., None] + np.arange(BLOCK)).reshape(NBOOT, -1)[:, :T]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(d[idx].reshape(NBOOT, -1), axis=1)
    lo, hi = np.nanpercentile(m, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def _med(x) -> float | None:
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), 2) if len(x) else None


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) with dst ~ s R src + t, least squares (Umeyama 1991). src, dst: (N, 3)."""
    ms, md = src.mean(0), dst.mean(0)
    X, Y = src - ms, dst - md
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / X.var(0).sum())
    return s, R, md - s * R @ ms


def _bh(p: list[float | None]) -> list[float | None]:
    ix = [i for i, v in enumerate(p) if v is not None]
    out: list[float | None] = [None] * len(p)
    m, prev = len(ix), 1.0
    for rank, i in sorted(enumerate(sorted(ix, key=lambda i: p[i])), key=lambda t: -t[0]):
        prev = min(prev, p[i] * m / (rank + 1))
        out[i] = round(prev, 4)
    return out


def _lr_pairs(names: list[str]) -> list[tuple[int, int]]:
    return [(i, names.index(re.sub(r"L(\d*)$", r"R\1", n))) for i, n in enumerate(names) if re.search(r"L\d*$", n) and re.sub(r"L(\d*)$", r"R\1", n) in names]


def compute(A: dict, B: dict, names_a: str = "A", names_b: str = "B") -> dict:
    """A, B = core.load results of two sessions on the same video (same node list)."""
    if A["kp"] is None or B["kp"] is None:
        raise ValueError("both sessions need 2D keypoints")
    cA, cB = [ALIAS.get(n, n) for n in A["nodes"]], [ALIAS.get(n, n) for n in B["nodes"]]   # models may name nodes differently or have different sets: compare the shared ones
    names = [n for n in cA if n in cB]
    if len(names) < 3:
        raise ValueError(f"fewer than 3 shared nodes: {A['nodes']} vs {B['nodes']}")
    ia, ib = [cA.index(n) for n in names], [cB.index(n) for n in names]
    dropped = {"a_only": [n for n in cA if n not in names], "b_only": [n for n in cB if n not in names]}
    alias_used = any(x != y for x, y in zip(A["nodes"], cA) if y in names) or any(x != y for x, y in zip(B["nodes"], cB) if y in names)
    T = min(len(A["kp"]), len(B["kp"]))
    rng, K = np.random.default_rng(0), len(names)
    D = np.full((T, 5, K), np.nan)            # distance in rig-cell pixels where both are confident
    CA, CB = np.zeros((T, 5, K)), np.zeros((T, 5, K))
    both, aonly, bonly, swap = (np.zeros((T, 5, K), bool) for _ in range(4))
    for c in range(5):
        a, b = core.kp_cell(A, c)[:T][:, ia], core.kp_cell(B, c)[:T][:, ib]
        wh = np.array(core.CELLS[c][2:], float)
        va, vb = a[..., 2] > THR, b[..., 2] > THR
        both[:, c], aonly[:, c], bonly[:, c] = va & vb, va & ~vb, ~va & vb
        d = np.linalg.norm((a[..., :2] - b[..., :2]) * wh, axis=-1)
        D[:, c][both[:, c]] = d[both[:, c]]
        CA[:, c], CB[:, c] = a[..., 2], b[..., 2]
        bs = b.copy()
        for i, j in _lr_pairs(names):
            bs[:, [i, j]] = b[:, [j, i]]
        swap[:, c] = both[:, c] & (np.linalg.norm((a[..., :2] - bs[..., :2]) * wh, axis=-1) < d)
    cams = [{"cam": c + 1, "both": round(float(both[:, c].mean()), 3), "a_only": round(float(aonly[:, c].mean()), 3),
             "b_only": round(float(bonly[:, c].mean()), 3), "neither": round(float((~(both | aonly | bonly)[:, c]).mean()), 3),
             "median_px": _med(D[:, c]), "p90_px": round(float(np.nanpercentile(D[:, c], 90)), 2) if np.isfinite(D[:, c]).any() else None,
             "swap_better": round(float(swap[:, c].sum() / max(both[:, c].sum(), 1)), 3)} for c in range(5)]
    nodes, pw = [], []
    for k, n in enumerate(names):
        m = both[:, :, k]
        p = None
        if m.sum() >= 10:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    p = float(sps.wilcoxon(CA[:, :, k][m], CB[:, :, k][m]).pvalue)
                except ValueError:
                    p = None
        pw.append(p)
        nodes.append({"node": n, "n": int(m.sum()), "median_px": _med(D[:, :, k]), "ci": _boot_ci(D[:, :, k], rng),
                      "per_cam_median_px": [_med(D[:, c, k]) for c in range(5)],
                      "conf_a": round(float(CA[:, :, k][m].mean()), 3) if m.any() else None,
                      "conf_b": round(float(CB[:, :, k][m].mean()), 3) if m.any() else None,
                      "p_conf": None if p is None else round(p, 4)})
    for o, q in zip(nodes, _bh(pw)):
        o["p_conf_bh"] = q
    out = {"a": names_a, "b": names_b, "nodes_list": names, "frames": T, "thr": THR, "cams": cams, "nodes": nodes,
           "median_px_all": _med(D), "ci_all": _boot_ci(D.reshape(T, -1), rng), "xyz": None, "caveat": CAVEAT + (ALIAS_NOTE if alias_used else ""), "alias_used": alias_used,
           "both_mean": round(float(both.mean()), 3), "dropped": dropped}
    xa, xb = A["xyz"][:T][:, ia], B["xyz"][:T][:, ib]
    if xa.shape == xb.shape and xa.size:
        pa, pb = xa.reshape(-1, 3).astype(float), xb.reshape(-1, 3).astype(float)
        ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
        s, R, t = umeyama(pb[ok], pa[ok])
        dist = np.linalg.norm((s * (pb @ R.T) + t) - pa, axis=1).reshape(T, K)
        ext = float(np.median([np.linalg.norm(f[:, None] - f[None], axis=-1).max() for f in xa[:: max(1, T // 200)]]))
        out["xyz"] = {"scale_b_to_a": round(s, 4), "extent": round(ext, 4), "median_pct_extent": round(float(np.median(dist)) / ext * 100, 2),
                      "ci_pct": [round(v / ext * 100, 2) for v in _boot_ci(dist, rng)] if np.isfinite(dist).all() else [None, None],
                      "per_node_pct": [round(float(np.median(dist[:, k])) / ext * 100, 2) for k in range(K)],
                      "timeline_pct": [round(float(np.median(r)) / ext * 100, 2) for r in dist],
                      "note": "B 를 A 좌표계로 similarity fit 한 뒤의 거리, 단위 = A 의 몸 extent(관절 최대 거리 중앙값) 대비 %"}
    out["_D"], out["_both"] = D, (both, aonly, bonly)
    return out


def figure(R: dict):
    """One figure, 6 panels, built from the same dict the page shows."""
    D, (both, aonly, bonly) = R.pop("_D"), R.pop("_both")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2), dpi=110)
    n, T = R["nodes"], R["frames"]
    x = np.arange(len(n))
    a0 = ax[0, 0]
    med = np.array([o["median_px"] or 0 for o in n])
    lo = np.array([(o["ci"][0] if o["ci"][0] is not None else m) for o, m in zip(n, med)])
    hi = np.array([(o["ci"][1] if o["ci"][1] is not None else m) for o, m in zip(n, med)])
    a0.bar(x, med, color="#8e8e93", yerr=[med - lo, hi - med], capsize=3)
    a0.set_xticks(x, [o["node"] for o in n], rotation=60, ha="right")
    a0.set_ylabel("2D distance A–B, median (cell px)")
    a0.set_title("1 · Distance per node (95 % block-bootstrap CI)", loc="left", fontsize=10)
    a1 = ax[0, 1]
    for c in range(5):
        d = np.sort(D[:, c][np.isfinite(D[:, c])])
        if len(d):
            a1.plot(d, np.arange(1, len(d) + 1) / len(d), color=CAM_COL[c], label=f"cam_{c + 1} (n={len(d)})")
    a1.set_xscale("log")
    a1.set_xlabel("2D distance (cell px, log)")
    a1.set_ylabel("cumulative fraction")
    a1.legend(fontsize=8)
    a1.set_title("2 · Distance distribution per camera (ECDF)", loc="left", fontsize=10)
    a2 = ax[0, 2]
    left = np.zeros(5)
    for key, col, lab in (("both", "#59a14f", "both confident"), ("a_only", A_COL, "A only"), ("b_only", B_COL, "B only"), ("neither", "#d1d1d6", "neither")):
        v = np.array([c[key] for c in R["cams"]])
        a2.barh([f"cam_{c['cam']}" for c in R["cams"]], v, left=left, color=col, label=lab)
        left += v
    a2.invert_yaxis()
    a2.set_xlabel(f"fraction of node-frames (conf > {R['thr']})")
    a2.legend(fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.32))
    a2.set_title("3 · Detection agreement per camera", loc="left", fontsize=10)
    a3 = ax[1, 0]
    ca = np.array([o["conf_a"] or 0 for o in n])
    cb = np.array([o["conf_b"] or 0 for o in n])
    a3.vlines(x, np.minimum(ca, cb), np.maximum(ca, cb), color="#c7c7cc")
    a3.scatter(x, ca, color=A_COL, label="A", zorder=3)
    a3.scatter(x, cb, color=B_COL, label="B", zorder=3)
    for i, o in enumerate(n):
        q = o["p_conf_bh"]
        if q is not None and q < .05:
            a3.text(i, max(ca[i], cb[i]) + .01, "*" if q >= .01 else "**", ha="center")
    a3.set_xticks(x, [o["node"] for o in n], rotation=60, ha="right")
    a3.set_ylabel("mean confidence (both confident)")
    a3.legend(fontsize=8)
    a3.set_title("4 · Confidence per node (stars: Wilcoxon, BH)", loc="left", fontsize=10)
    a4, a5 = ax[1, 1], ax[1, 2]
    xy = R["xyz"]
    if xy:
        a4.plot(xy["timeline_pct"], color="#8e8e93", lw=1)
        a4.axhline(xy["median_pct_extent"], color="#e15759", ls="--", lw=1, label=f"median {xy['median_pct_extent']} %")
        a4.set_xlabel("frame")
        a4.set_ylabel("3D distance (% of body extent)")
        a4.legend(fontsize=8)
        a5.bar(x, xy["per_node_pct"], color="#8e8e93")
        a5.set_xticks(x, [o["node"] for o in n], rotation=60, ha="right")
        a5.set_ylabel("3D distance, median (% extent)")
    else:
        a4.text(.5, .5, "no comparable 3D", ha="center", transform=a4.transAxes)
        a5.axis("off")
    a4.set_title("5 · 3D distance per frame (after similarity fit)", loc="left", fontsize=10)
    a5.set_title("6 · 3D distance per node", loc="left", fontsize=10)
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"A = {R['a']}   vs   B = {R['b']}   ({T} frames, agreement only, no ground truth)", x=.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def aligned(root: Path, a: str, b: str, stride: int = 1) -> dict:
    """B's 3D skeleton in A's frame (similarity fitted on the shared nodes, applied to every B node) for drawing both in one 3D view."""
    A, B = core.load(root, a), core.load(root, b)
    cA, cB = [ALIAS.get(n, n) for n in A["nodes"]], [ALIAS.get(n, n) for n in B["nodes"]]
    names = [n for n in cA if n in cB]
    ia, ib = [cA.index(n) for n in names], [cB.index(n) for n in names]
    T = min(len(A["xyz"]), len(B["xyz"]))
    pa, pb = A["xyz"][:T][:, ia].reshape(-1, 3).astype(float), B["xyz"][:T][:, ib].reshape(-1, 3).astype(float)
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    s, R, t = umeyama(pb[ok], pa[ok])
    X = s * (B["xyz"][:T].astype(float) @ R.T) + t
    return {"nodes": B["nodes"], "edges": B["edges"], "stride": stride, "xyz": np.round(X[::stride], 4).tolist()}


def run_all(root: Path, a: str, bs: list[str]) -> list[dict]:
    """A against every other session: one summary row each (no figure); a pair that cannot be compared carries its error."""
    A, rows = core.load(root, a), []
    for b in bs:
        try:
            R = compute(A, core.load(root, b))
            x = R["xyz"]
            rows.append({"b": b, "frames": R["frames"], "nodes": len(R["nodes_list"]), "dropped": R["dropped"], "median_px": R["median_px_all"], "ci": R["ci_all"],
                         "both": R["both_mean"], "xyz_pct": x and x["median_pct_extent"], "xyz_ci": x and x["ci_pct"]})
        except ValueError as e:
            rows.append({"b": b, "error": str(e)})
    return rows


def run(root: Path, a: str, b: str, save: bool = False, out_root: Path | None = None) -> dict:
    R = compute(core.load(root, a), core.load(root, b), Path(a).name.removesuffix(".json.gz"), Path(b).name.removesuffix(".json.gz"))
    fig = figure(R)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    R["png"] = base64.b64encode(buf.getvalue()).decode()
    if save:
        d = (out_root or core.RESULTS_BASE or root / ".subtle_results") / "compare" / f"{R['a']}__vs__{R['b']}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "compare.png").write_bytes(buf.getvalue())
        pub = {k: v for k, v in R.items() if k != "png"}
        (d / "compare.json").write_text(json.dumps(pub, indent=1, ensure_ascii=False))
        with open(d / "node_stats.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node", "n_both", "median_px", "ci_lo", "ci_hi", *[f"cam{c}_median_px" for c in range(1, 6)], "conf_a", "conf_b", "p_wilcoxon", "p_bh", "xyz_median_pct_extent"])
            for i, o in enumerate(R["nodes"]):
                w.writerow([o["node"], o["n"], o["median_px"], *o["ci"], *o["per_cam_median_px"], o["conf_a"], o["conf_b"], o["p_conf"], o["p_conf_bh"],
                            R["xyz"]["per_node_pct"][i] if R["xyz"] else ""])
        R["saved"] = str(d)
    plt.close(fig)
    return R


def _selftest() -> None:
    rng = np.random.default_rng(1)
    T, K = 120, 11
    kp = np.zeros((T, 5, K, 3), np.float32)
    kp[..., 0], kp[..., 1], kp[..., 2] = rng.uniform(.3, .7, (T, 5, K)), rng.uniform(.3, .7, (T, 5, K)), .9
    xyz = rng.normal(size=(T, K, 3)).astype(np.float32)
    A = {"nodes": [f"n{i}" for i in range(K)], "kp": kp, "xyz": xyz}
    B = {"nodes": A["nodes"], "kp": kp.copy(), "xyz": xyz * 2 + 1}
    B["kp"][..., 0] += 0.01                                  # shift = 0.01 of the 3600 px composite = 36 px
    R = compute(A, B)
    assert abs(R["cams"][0]["median_px"] - 36) < 0.05, R["cams"][0]
    assert R["xyz"]["median_pct_extent"] < 1e-3 and abs(R["xyz"]["scale_b_to_a"] - .5) < 1e-3, R["xyz"]   # a similarity is undone exactly
    B["kp"][:, 1, :, 2] = 0                                  # camera 2 of B detects nothing
    R2 = compute(A, B)
    assert R2["cams"][1]["both"] == 0 and R2["cams"][1]["a_only"] == 1, R2["cams"][1]
    figure(R2)
    print("compare selftest ok")


if __name__ == "__main__":
    _selftest()
