#!/usr/bin/env python
"""Per-keypoint pixel error of a SLEAP model, same definitions as the SUBTLE viewer QC figure.

Definitions (mirrors tools/pose_runtime/evaluate_labels.py and keypoint_qc.py):
  - error = Euclidean pixel distance between GT and prediction at visible GT points
  - a visible GT point with no finite prediction is "missing" (separate %, denominator = visible GT)
  - IQR flag % = share of matched errors beyond Tukey fences (1.5 x IQR), denominator = matched
  - Mean, SD (ddof=0), Median, P90, Q25/Q75 over matched errors (np.quantile, linear)
Frames are matched by (source video path, frame_idx); nodes by name. Domain tag = AVATAR when
the source video path contains "AVATAR_split", else "other" (KHU/JHL labs).

Each --set is one GT/prediction pair. Prediction files for training labels do not exist in
SLEAP output (it only predicts val): run `sleap-nn track -i <pkg.slp>` and pass that file as the
prediction; frames absent from the GT file are ignored. Train error is IN-SAMPLE (fit + label
noise), not accuracy.

Example:
  uv run --with h5py --with numpy --with matplotlib python scripts/kp_error_by_keypoint.py \
    --set val labels_gt.val.0.slp labels_pr.val.0.slp \
    --set train labels_gt.train.0.slp pred_pkg.slp --out kp_error_train
  python scripts/kp_error_by_keypoint.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

AVATAR_TAG = "AVATAR_split"
DOMAINS = ("all", "avatar", "other")
PALETTE_FALLBACK = {  # viewer anatomical colors (tools/subtle_viewer/web/keypoint_palette.json)
    "nose1": "#ffff00", "neck1": "#ff00ff", "earL1": "#ffff00", "earR1": "#ffff00",
    "forelegL1": "#0000ff", "forelegR1": "#00ff00", "tailstart1": "#ffa500",
    "hindlegL1": "#00ffff", "hindlegR1": "#ff0000", "tail1": "#ffa500", "tailend1": "#ffa500",
}


# ---------------------------------------------------------------- SLEAP .slp reading (h5py)
def _video_name(video: dict) -> str:
    src = video.get("source_video")
    if isinstance(src, dict) and isinstance(src.get("filename"), str):
        return src["filename"]
    return str(video["backend"]["filename"])


def read_slp(path: Path, predicted: bool) -> tuple[list[str], dict]:
    """Return (node names, {(video, frame_idx): (N,2) xy, NaN where invisible/non-finite}).

    predicted=False reads user instances (GT); True reads predicted instances (best score).
    """
    import h5py

    with h5py.File(path, "r") as f:
        nodes = [n["name"] for n in json.loads(f["metadata"].attrs["json"])["nodes"]]
        vlist = [json.loads(v) for v in f["videos_json"][:]]
        # frames["video"] is the embedded dataset number ("videoN/video"), not the list index
        videos = {int(v["backend"]["dataset"].split("/")[0][5:]): v for v in vlist
                  if str(v["backend"].get("dataset", "")).startswith("video")}
        if len(videos) != len(vlist):
            videos = dict(enumerate(vlist))
        frames, inst = f["frames"][:], f["instances"][:]
        pts = f["pred_points" if predicted else "points"][:]
    out: dict = {}
    want = 1 if predicted else 0
    for fr in frames:
        rows = inst[(inst["frame_id"] == fr["frame_id"]) & (inst["instance_type"] == want)]
        if not len(rows):
            continue
        if predicted:
            row = rows[np.argmax(rows["score"])]
        elif len(rows) == 1:
            row = rows[0]
        else:
            raise ValueError(f"{path}: expected one GT instance per frame, got {len(rows)}")
        p = pts[row["point_id_start"]:row["point_id_end"]]
        xy = np.stack([p["x"], p["y"]], 1).astype(float)
        xy[~np.asarray(p["visible"], bool) | ~np.isfinite(xy).all(1)] = np.nan
        key = (_video_name(videos[fr["video"]]), int(fr["frame_idx"]))
        if key in out:
            raise ValueError(f"{path}: duplicate frame identity {key}")
        out[key] = xy
    return nodes, out


# ---------------------------------------------------------------- statistics
def iqr_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Strict Tukey fences on the given finite values."""
    if not len(values):
        return np.zeros(0, bool)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return (values < q1 - k * (q3 - q1)) | (values > q3 + k * (q3 - q1))


def summarize(errors: np.ndarray, n_visible: int) -> dict:
    """errors = matched errors (finite); n_visible = visible GT points (incl. missing)."""
    e = np.asarray(errors, float)
    n = len(e)
    row = dict(n_visible=n_visible, n_matched=n, n_missing=n_visible - n,
               missing_pct=100 * (n_visible - n) / n_visible if n_visible else np.nan)
    if n:
        flags = int(iqr_mask(e).sum())
        row.update(mean=e.mean(), sd=e.std(), median=np.median(e), p90=np.quantile(e, 0.9),
                   q25=np.quantile(e, 0.25), q75=np.quantile(e, 0.75), iqr_flag_n=flags,
                   iqr_flag_pct=100 * flags / n)
    else:
        row.update({k: np.nan for k in
                    ("mean", "sd", "median", "p90", "q25", "q75", "iqr_flag_n", "iqr_flag_pct")})
    return row


def pair_errors(gt: dict, pr: dict, n_nodes: int) -> list[tuple[str, np.ndarray]]:
    """Per GT frame: (video, error vector (n_nodes,)); NaN = invisible GT; inf = missing pred."""
    out = []
    for key, g in gt.items():
        p = pr.get(key, np.full_like(g, np.nan))
        err = np.linalg.norm(g - p, axis=1)
        err[np.isnan(g).any(1)] = np.nan  # not visible: excluded from both numerator and denominator
        err[np.isnan(p).any(1) & ~np.isnan(g).any(1)] = np.inf  # visible GT, no prediction
        out.append((key[0], err))
    assert all(len(e) == n_nodes for _, e in out)
    return out


def table(set_name: str, nodes: list[str], pairs: list[tuple[str, np.ndarray]]) -> list[dict]:
    rows = []
    for dom in DOMAINS:
        sel = [e for v, e in pairs
               if dom == "all" or (AVATAR_TAG in v) == (dom == "avatar")]
        mat = np.array(sel).reshape(len(sel), len(nodes))
        for i, node in enumerate(nodes):
            col = mat[:, i]
            vis = col[~np.isnan(col)]
            rows.append(dict(set=set_name, domain=dom, node=node, n_frames=len(sel),
                             **summarize(vis[np.isfinite(vis)], len(vis))))
    return rows


def samples(nodes: list[str], pairs: list[tuple[str, np.ndarray]]) -> list[np.ndarray]:
    mat = np.array([e for _, e in pairs])
    return [mat[:, i][np.isfinite(mat[:, i])] for i in range(len(nodes))]


# ---------------------------------------------------------------- figure
def figure(panels: list[tuple[str, list[str], list[np.ndarray], list[dict]]],
           out: Path, palette: dict) -> None:
    """Scatter + box (Q25-Q75, whiskers min-max) + Mean +- SD per keypoint, panels side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 4.8), sharey=True,
                             squeeze=False, layout="constrained")
    rng = np.random.default_rng(0)
    for ax, (title, nodes, samp, stats) in zip(axes[0], panels):
        for i, (node, s) in enumerate(zip(nodes, samp)):
            col = palette.get(node, "#7ee6b1")
            n = len(s)
            size, alpha = ((2.0, 0.65) if n <= 20 else (0.9, 0.35) if n <= 200
                           else (0.3, 0.12) if n <= 2000 else (0.1, 0.035))
            ax.scatter(i + rng.uniform(-0.12, 0.12, n), s, s=size * 4, color=col, alpha=alpha,
                       zorder=3, rasterized=True)
            st = stats[i]
            if n:
                ax.bxp([dict(med=st["median"], q1=st["q25"], q3=st["q75"], whislo=s.min(),
                             whishi=s.max(), fliers=[])], positions=[i], widths=0.25,
                       patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops=dict(facecolor=matplotlib.colors.to_rgba(col, 0.12),
                                     edgecolor="#424852"),
                       medianprops=dict(color="#171b22"), whiskerprops=dict(color="#69717b"),
                       capprops=dict(color="#69717b"))
                ax.errorbar(i, st["mean"], yerr=st["sd"], fmt="none", color="#515761",
                            linewidth=0.8, capsize=2, zorder=2)
                ax.scatter(i, st["mean"], c=col, s=22, edgecolors="#303640", linewidths=0.65,
                           zorder=4)
                ax.text(i, 1.01, f"{st['iqr_flag_pct']:.1f}%\n({st['iqr_flag_n']}/{n})",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=6.5, color="#c87518")
        ax.set_yscale("asinh", linear_width=1)
        ax.set_yticks([-100, -10, 0, 10, 100, 1000], labels=["-100", "-10", "0", "10", "100", "1000"])
        ax.set_xticks(range(len(nodes)), labels=[
            f"{nd}\n{stats[i]['missing_pct']:.1f}%\n({stats[i]['n_missing']}/{stats[i]['n_visible']})"
            for i, nd in enumerate(nodes)], fontsize=7)
        ax.set_xlim(-0.6, len(nodes) - 0.4)
        ax.grid(axis="y", color="#e7eaee", linewidth=0.65, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=10, pad=26)
    axes[0][0].set_ylabel("Position error (px) · asinh")
    fig.supxlabel("Missing % (missing / visible GT) under each keypoint; amber % = IQR-flagged / matched. "
                  "Box = Q25-Q75, line = median, dot = mean, bar = mean ± SD, whiskers = min-max.",
                  fontsize=7)
    fig.savefig(out, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- self-test
def selftest() -> None:
    nodes = ["a", "b"]
    gt = {("AVATAR_split/x.mp4", i): np.array([[0.0, 0.0], [10.0, 10.0]]) for i in range(4)}
    gt[("KHU/y.mp4", 0)] = np.array([[0.0, 0.0], [np.nan, np.nan]])  # b invisible
    pr = {("AVATAR_split/x.mp4", i): np.array([[3.0, 4.0], [10.0, 10.0 + i]]) for i in range(4)}
    pr[("AVATAR_split/x.mp4", 3)][0] = np.nan  # missing prediction for a at frame 3
    # KHU frame has no prediction at all -> a missing
    pairs = pair_errors(gt, pr, 2)
    rows = {(r["domain"], r["node"]): r for r in table("t", nodes, pairs)}
    a = rows[("all", "a")]
    assert (a["n_visible"], a["n_matched"], a["n_missing"]) == (5, 3, 2), a
    assert abs(a["missing_pct"] - 40.0) < 1e-9 and a["median"] == 5.0 and a["mean"] == 5.0
    b = rows[("all", "b")]  # invisible GT excluded from denominator: 4 visible, errors 0,1,2,3
    assert (b["n_visible"], b["n_matched"]) == (4, 4) and b["median"] == 1.5, b
    assert abs(b["p90"] - np.quantile([0, 1, 2, 3], 0.9)) < 1e-9
    assert rows[("avatar", "a")]["n_visible"] == 4 and rows[("other", "a")]["n_visible"] == 1
    assert rows[("other", "b")]["n_visible"] == 0 and np.isnan(rows[("other", "b")]["median"])
    assert iqr_mask(np.array([1.0, 1, 1, 1, 50])).tolist() == [False] * 4 + [True]
    assert iqr_mask(np.array([2.0, 2, 2])).sum() == 0
    print("selftest OK")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", nargs=3, action="append", metavar=("NAME", "GT_SLP", "PRED_SLP"),
                    help="one GT/prediction pair; repeat for train, val, ...")
    ap.add_argument("--out", type=Path, help="output directory (created)")
    ap.add_argument("--figure-sets", nargs="+", help="set names to draw (default: all sets)")
    ap.add_argument("--palette", type=Path, help="keypoint_palette.json (name -> [part, hex])")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.set or not a.out:
        ap.error("--set and --out are required (or --selftest)")
    palette = dict(PALETTE_FALLBACK)
    if a.palette:
        palette.update({k: v[1] for k, v in json.loads(a.palette.read_text()).items()})
    a.out.mkdir(parents=True, exist_ok=True)

    rows, panels, report = [], [], {}
    for name, gt_path, pr_path in a.set:
        nodes, gt = read_slp(Path(gt_path), predicted=False)
        pnodes, pr = read_slp(Path(pr_path), predicted=True)
        if pnodes != nodes:
            raise ValueError(f"{name}: GT/prediction node names differ")
        extra = len(pr.keys() - gt.keys())
        pairs = pair_errors(gt, pr, len(nodes))
        tab = table(name, nodes, pairs)
        rows += tab
        report[name] = dict(gt_frames=len(gt), pred_frames=len(pr), pred_frames_not_in_gt=extra,
                            gt_frames_without_pred=len(gt.keys() - pr.keys()),
                            avatar_frames=sum(AVATAR_TAG in k[0] for k in gt))
        allrows = [r for r in tab if r["domain"] == "all"]
        panels.append((name, f"{name} · {len(gt)} frames", nodes, samples(nodes, pairs), allrows))
    fields = ["set", "domain", "node", "n_frames", "n_visible", "n_matched", "n_missing",
              "missing_pct", "median", "mean", "sd", "p90", "q25", "q75", "iqr_flag_n",
              "iqr_flag_pct"]
    with (a.out / "kp_error_by_keypoint.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows)
    (a.out / "kp_error_report.json").write_text(json.dumps(report, indent=2))
    chosen = a.figure_sets or [p[0] for p in panels]
    figure([(p[1], p[2], p[3], p[4]) for p in panels if p[0] in chosen],
           a.out / "kp_error_scatter.png", palette)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
