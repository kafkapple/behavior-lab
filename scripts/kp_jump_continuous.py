#!/usr/bin/env python
"""Frame-to-frame 2D jumps of the SLEAP 1423 predictions on continuous SUBTLE/AVATAR sessions.

Premise: the labelled frames are sparse (see --gaps), so speed/acceleration exist only on continuous
predictions. This measures prediction INSTABILITY, not accuracy: real motion also gives large steps.

Input: SUBTLE session json.gz, keypoint[frame][camera][node] = [x, y, conf] (x, y = composite-normalised;
conf == 0 is the placeholder for "not detected"). Cell pixels via behavior_lab.rig.avatar.to_cell.
Per (session, camera, keypoint):
  speed = |p_t - p_(t-1)| / L,  accel = |p_(t+1) - 2 p_t + p_(t-1)| / L, both in % body length per frame,
  L = median neck1-tailstart1 distance of that session and camera (px). Only runs of detected points count.
IQR flag = Tukey fences (1.5 x IQR) on the pooled (camera, keypoint) sample, same rule as the viewer.
Confidence check: share of flagged steps whose lower confidence lies below the Q25 of all steps.

  uv run --with numpy --with pandas --with pyyaml --with matplotlib --with h5py python scripts/kp_jump_continuous.py \
     --out OUT SESSION.json.gz ...
  python scripts/kp_jump_continuous.py --selftest
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from kp_error_by_keypoint import draw_panel, iqr_mask, summarize  # noqa: E402
from pkg_groups import KP_COLORS  # noqa: E402

NECK, TAILSTART = 1, 6


def load_session(path: Path) -> tuple[list[str], np.ndarray]:
    """-> node names, (T, 5, nodes, 3) with x, y in cell pixels, conf unchanged."""
    from behavior_lab.rig.avatar import to_cell

    d = json.load(gzip.open(path, "rt"))
    nodes = d["node"]["id"]
    kp = d["keypoint"]
    assert sorted(map(int, kp)) == list(range(len(kp))), "frames must be contiguous"
    a = np.zeros((len(kp), 5, len(nodes), 3), np.float32)
    for f, cams in kp.items():
        for c, pts in cams.items():
            for k, v in pts.items():
                a[int(f), int(c), int(k)] = v
    for c in range(5):
        a[:, c, :, :2] = to_cell(a[:, c, :, :2], c)
    return nodes, a


def steps(xy: np.ndarray, conf: np.ndarray, body: float) -> dict:
    """xy (T,2) px, conf (T,). Returns speed/accel in % BL with their frame index and lower confidence."""
    ok = conf > 0
    t = np.nonzero(ok[1:] & ok[:-1])[0] + 1  # step t-1 -> t
    sp = 100 * np.linalg.norm(xy[t] - xy[t - 1], axis=1) / body
    t2 = np.nonzero(ok[2:] & ok[1:-1] & ok[:-2])[0] + 1  # centre frame
    ac = 100 * np.linalg.norm(xy[t2 + 1] - 2 * xy[t2] + xy[t2 - 1], axis=1) / body
    return dict(sp=sp, sp_t=t, sp_c=np.minimum(conf[t], conf[t - 1]), ac=ac, ac_t=t2,
                ac_c=np.minimum.reduce([conf[t2 - 1], conf[t2], conf[t2 + 1]]), n_trans=len(xy) - 1,
                n_trans2=len(xy) - 2)


def selftest() -> None:
    t = np.arange(50.0)
    xy = np.stack([t * 2.0, t * 0.0], 1)
    c = np.ones(50)
    r = steps(xy, c, 10.0)
    assert np.allclose(r["sp"], 20.0) and np.allclose(r["ac"], 0.0)
    xy[25, 0] += 100  # one teleport: two big steps, accel 200 at the centre frame and 100 next to it
    r = steps(xy, c, 10.0)
    assert np.isclose(r["sp"].max(), 1020.0) and np.isclose(np.sort(r["sp"])[-2], 980.0)
    assert int(r["ac_t"][np.argmax(r["ac"])]) == 25 and np.isclose(r["ac"].max(), 2000.0)
    c[10] = 0  # missing frame removes steps touching it
    r = steps(xy, c, 10.0)
    assert 10 not in r["sp_t"] and 11 not in r["sp_t"] and len(r["sp"]) == 47
    assert iqr_mask(r["sp"]).sum() == 2
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = KP_COLORS
    a.out.mkdir(parents=True, exist_ok=True)
    pool: dict = {}  # (cam, kp) -> lists
    nodes, excl, n_frames = None, 0, 0
    for p in a.sessions:
        nodes, arr = load_session(p)
        n_frames += len(arr)
        for c in range(5):
            w, h = {0: (1200, 1000), 1: (1200, 1000), 2: (1200, 1200), 3: (1200, 1000), 4: (1200, 1000)}[c]
            x, y, cf = arr[:, c, :, 0], arr[:, c, :, 1], arr[:, c, :, 2]
            inside = (x >= 0) & (x <= w) & (y >= 0) & (y <= h)
            excl += int(((cf > 0) & ~inside).sum())
            arr[:, c, :, 2] = np.where(inside, cf, 0)  # detections outside the camera cell = invalid
            ok = (arr[:, c, NECK, 2] > 0) & (arr[:, c, TAILSTART, 2] > 0)
            body = float(np.median(np.linalg.norm(arr[ok, c, NECK, :2] - arr[ok, c, TAILSTART, :2], axis=1)))
            assert body > 5, (p.name, c, body)
            for k in range(len(nodes)):
                r = steps(arr[:, c, k, :2], arr[:, c, k, 2], body)
                d = pool.setdefault((c, k), dict(sp=[], ac=[], sp_c=[], ac_c=[], sp_id=[], ac_id=[], nt=0, nt2=0))
                d["sp"].append(r["sp"]); d["ac"].append(r["ac"]); d["sp_c"].append(r["sp_c"]); d["ac_c"].append(r["ac_c"])
                d["sp_id"] += [(p.stem, int(t)) for t in r["sp_t"]]
                d["ac_id"] += [(p.stem, int(t)) for t in r["ac_t"]]
                d["nt"] += r["n_trans"]; d["nt2"] += r["n_trans2"]
    rows, top, conf_rows, panels = [], [], [], {}
    for (c, k), d in sorted(pool.items()):
        for kind, nt in (("speed", d["nt"]), ("accel", d["nt2"])):
            key = "sp" if kind == "speed" else "ac"
            v, cf, ids = np.concatenate(d[key]), np.concatenate(d[key + "_c"]), d[key + "_id"]
            m = iqr_mask(v)
            st = summarize(v, nt)
            rows.append(dict(kind=kind, camera=c + 1, node=nodes[k], n_possible=nt, n_valid=len(v),
                             missing_pct=st["missing_pct"], median=st["median"], mean=st["mean"], sd=st["sd"],
                             p90=st["p90"], q25=st["q25"], q75=st["q75"], iqr_flag_n=st["iqr_flag_n"],
                             iqr_flag_pct=st["iqr_flag_pct"]))
            panels[(kind, c, k)] = (v, st)
            lo = cf < np.quantile(cf, 0.25)
            conf_rows.append(dict(kind=kind, camera=c + 1, node=nodes[k], flagged=int(m.sum()),
                                  conf_med_flagged=float(np.median(cf[m])) if m.any() else np.nan,
                                  conf_med_unflagged=float(np.median(cf[~m])),
                                  flagged_in_lowest_conf_quartile_pct=float(100 * lo[m].mean()) if m.any() else np.nan))
            if kind == "speed":
                for i in np.argsort(v)[::-1][:40]:  # candidates; the per-keypoint top 5 is picked below
                    top.append(dict(node=nodes[k], camera=c + 1, session=ids[i][0], frame=ids[i][1],
                                    speed_pct_body_length_per_frame=round(float(v[i]), 1), min_conf=round(float(cf[i]), 3)))
    best = []  # top 5 per keypoint over all cameras/sessions; one event = one entry (skip +-3 frames)
    for nd in nodes:
        for r in sorted((t for t in top if t["node"] == nd), key=lambda t: -t["speed_pct_body_length_per_frame"]):
            if sum(b["node"] == nd for b in best) < 5 and not any(
                    b["node"] == nd and b["camera"] == r["camera"] and b["session"] == r["session"]
                    and abs(b["frame"] - r["frame"]) <= 3 for b in best):
                best.append(r)
    top = best
    assert len(top) == 5 * len(nodes)
    keys = list(rows[0])
    with (a.out / "kp_jump_by_keypoint.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        w.writerows({q: (round(x, 4) if isinstance(x, float) else x) for q, x in r.items()} for r in rows)
    with (a.out / "kp_jump_top5.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(top[0])); w.writeheader(); w.writerows(top)
    with (a.out / "kp_jump_confidence.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conf_rows[0])); w.writeheader()
        w.writerows({q: (round(x, 4) if isinstance(x, float) else x) for q, x in r.items()} for r in conf_rows)
    rng = np.random.default_rng(0)
    for kind in ("speed", "accel"):  # one figure per kind, 3 x 2 camera panels (readable at <= 1000 px)
        fig, axes = plt.subplots(2, 3, figsize=(7.2 * 3, 4.6 * 2), sharey=True, layout="constrained", squeeze=False)
        axes[1][2].axis("off")
        for c in range(5):
            ax = axes[c // 3][c % 3]
            draw_panel(ax, f"cam_{c + 1} · {kind}", nodes, [panels[(kind, c, k)][0] for k in range(len(nodes))],
                       [panels[(kind, c, k)][1] for k in range(len(nodes))], palette, rng, compact=True,
                       yticks=(0, 1, 10, 100, 1000), linear_width=1.0)
        for r in range(2):
            axes[r][0].set_ylabel(f"{kind} (% body length / frame) · asinh")
        fig.supxlabel(f"{len(a.sessions)} sessions, {n_frames} frames. Missing % = steps without two (speed) / three (accel) "
                      "consecutive detections; amber % = IQR-flagged / valid. Box = Q25-Q75, line = median, dot = mean, "
                      "bar = mean ± SD, whiskers = min-max.", fontsize=7)
        fig.savefig(a.out / f"kp_jump_{kind}.png", dpi=300)
        plt.close(fig)
    (a.out / "kp_jump_report.json").write_text(json.dumps(dict(
        sessions=[p.name for p in a.sessions], frames=n_frames, detections_outside_cell_dropped=excl), indent=2))
    print("done", n_frames, "frames; outside-cell detections dropped:", excl)


if __name__ == "__main__":
    main()
