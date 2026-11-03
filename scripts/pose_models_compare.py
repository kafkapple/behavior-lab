"""Per-keypoint differences between pose results of one clip, all in the same figure format.

Inputs: server-pipeline result json.gz files (keypoint[frame][cam][node] = [x, y, score], x/y normalized by
the composite size; (0, 0) = no output). No tolerance is applied anywhere: distances are reported as
distributions (n, median, p90, p99, max). Colour = keypoint (pkg_groups.KP_COLORS), one marker size (MARKER_R_REL).

    .venv-subtle/bin/python scripts/pose_models_compare.py --results outputs/pose_models_261103 \
        --server ~/data/subtle_outage_poc_261030/cort1_first240_new.json.gz --video <composite.mp4> --out <figdir>
"""
import argparse
import gzip
import itertools
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.colors
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from pkg_groups import KP_COLORS, MARKER_R_REL, check_palette  # noqa: E402

W, H = 3600, 2000  # composite size (ffprobe 261103); json coordinates are normalized by it
CELLS = [[0, 0, 1200, 1000], [1200, 0, 2400, 1000], [2400, 0, 3600, 1200], [0, 1000, 1200, 2000], [1200, 1000, 2400, 2000]]
REP = "sleap1423_mps_r1"  # representative SLEAP = the deployed default n=1423-1 (Mac run 1; run 2 is identical)
MARK = 1.5  # overlay marker = MARK x MARKER_R_REL x cell width (all overlays alike)
FRAME = 120  # fixed before looking at results: middle of the 240 frames
SAME_NODES = ["server_sleap1423", "sleap1423_mps_r1", "sleap1423_mps_r2", "sleap1501single_mps", "lp_reorganize_mps", "lp_khu_mps"]
LABEL = {"server_sleap1423": "server SLEAP 1423-1", "sleap1423_mps_r1": "Mac SLEAP 1423-1 r1", "sleap1423_mps_r2": "Mac SLEAP 1423-1 r2",
         "sleap1501single_mps": "Mac SLEAP 1501 single", "lp_reorganize_mps": "Mac LP reorganize", "lp_khu_mps": "Mac LP KHU", "yolo_avatar3d_mps": "Mac YOLO (AVATAR3D cfg, 9 pt)"}


def load(path: Path):
    d = json.load(gzip.open(path))
    names = d["node"]["id"]
    a = np.full((len(d["keypoint"]), 5, len(names), 3), np.nan)
    for f, cams in d["keypoint"].items():
        for c, nodes in cams.items():
            for k, (x, y, s) in nodes.items():
                if not (x == 0 and y == 0):
                    a[int(f), int(c), int(k)] = (x * W, y * H, s)
    own = set(names)
    if own <= set(KP_COLORS):  # node order and count differ between files: index by name, NaN for nodes the model lacks
        full = np.full((a.shape[0], 5, len(KP_COLORS), 3), np.nan)
        for i, n in enumerate(names):
            full[:, :, list(KP_COLORS).index(n)] = a[:, :, i]
        names, a = list(KP_COLORS), full
    return names, a, own


def pair_stats(a, b):
    dist = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)  # (F, cam, node), NaN where either side has no output
    only_a = (~np.isnan(a[..., 0]) & np.isnan(b[..., 0])).sum((0, 1))
    only_b = (np.isnan(a[..., 0]) & ~np.isnan(b[..., 0])).sum((0, 1))
    return dist, only_a, only_b


BINS = [0, 1, 10, 100, np.inf]  # decades, descriptive only (no tolerance)
BIN_LABEL = ["<= 1 px", "1-10 px", "10-100 px", "> 100 px"]
BIN_COLOR = ["#d9e8f5", "#8fbfe3", "#3b82c4", "#b2182b"]


def distance_figs(a, data, kp_names, pairs, dists, tags, title):
    """Fig A: share of pairs per distance bin and keypoint (median in px on top). Fig B: keypoint x frame heat map."""
    fig, axs = plt.subplots(len(pairs), 1, figsize=(11, 2.6 * len(pairs)), sharex=True, sharey=True, squeeze=False)
    for ax, (x, y) in zip(axs[:, 0], pairs):
        d = dists[(x, y)]
        for k, n in enumerate(kp_names):
            v = d[..., k][~np.isnan(d[..., k])]
            if not (v.size and n in data[x][2] and n in data[y][2]):
                ax.text(k, 50, "no\npair", ha="center", va="center", fontsize=7, color="gray")
                continue
            share = np.histogram(v, bins=BINS)[0] / v.size * 100
            bottom = 0
            for sh, col in zip(share, BIN_COLOR):
                ax.bar(k, sh, bottom=bottom, color=col, width=0.72, edgecolor="white", linewidth=0.4)
                bottom += sh
            ax.text(k, 101.5, f"{np.median(v):.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 112)
        ax.set_ylabel("share of pairs (%)")
        ax.set_title(f"{LABEL[x]}  vs  {LABEL[y]}", fontsize=9, loc="left")
    axs[-1, 0].set_xticks(range(len(kp_names)), kp_names, rotation=30, ha="right")
    axs[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in BIN_COLOR], labels=BIN_LABEL, ncol=4, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.12), title="distance between the two results (bins are descriptive, not a tolerance)", title_fontsize=8, frameon=False)
    fig.text(0.995, 0.005, "number above a bar = median distance (px)", ha="right", fontsize=8)
    fig.suptitle(f"Fig. {tags[0]} · {title}: share of keypoint pairs per distance bin (all 240 frames x 5 cameras, composite px)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(a.out / f"fig{tags[0]}_distance_by_keypoint.png", dpi=110)
    plt.close(fig)

    vmax = float(max(np.nanmax(d) for d in dists.values()))
    norm = matplotlib.colors.SymLogNorm(linthresh=1, vmin=0, vmax=vmax)
    fig, axs = plt.subplots(len(pairs), 1, figsize=(11, 2.3 * len(pairs)), sharex=True, squeeze=False, constrained_layout=True)
    for ax, (x, y) in zip(axs[:, 0], pairs):
        m = np.nanmax(np.where(np.isnan(dists[(x, y)]), -1, dists[(x, y)]), axis=1).T  # (node, frame), -1 = no pair
        im = ax.imshow(np.ma.masked_less(m, 0), aspect="auto", cmap="magma", norm=norm, interpolation="nearest")
        ax.set_yticks(range(len(kp_names)), kp_names, fontsize=7)
        ax.set_title(f"{LABEL[x]}  vs  {LABEL[y]}", fontsize=9, loc="left")
    axs[-1, 0].set_xlabel("frame")
    cb = fig.colorbar(im, ax=axs[:, 0].tolist(), shrink=0.85, pad=0.02)
    cb.set_label("largest distance over 5 cameras (px, log-like scale)", fontsize=8)
    fig.suptitle(f"Fig. {tags[1]} · {title}: distance by frame and keypoint (white = no pair, same colour scale in every row)", fontsize=10)
    fig.savefig(a.out / f"fig{tags[1]}_distance_by_frame.png", dpi=110)
    plt.close(fig)
    return vmax


def overlay(a, data, keys, img, no, title, fname):
    """Same frame, same zoomed crop per camera for every model; dot colour = keypoint, black = other node names."""
    keys = [k for k in keys if k in data]
    ref = data[REP][1][FRAME]  # (cam, node, 3): crop window follows the representative SLEAP animal
    fig, axs = plt.subplots(len(keys), 5, figsize=(15, len(keys) * 3.0 + 0.8), squeeze=False)
    for c, (x0, y0, x1, y1) in enumerate(CELLS):
        pts = ref[c, :, :2][~np.isnan(ref[c, :, 0])]
        edge = (abs(pts[:, 0] - x0) <= 2) | (abs(pts[:, 0] - x1) <= 2) | (abs(pts[:, 1] - y0) <= 2) | (abs(pts[:, 1] - y1) <= 2)
        pts = pts[~edge] if (~edge).any() else pts  # points clipped to the cell border must not stretch the window
        cx, cy = pts.mean(0) if len(pts) else ((x0 + x1) / 2, (y0 + y1) / 2)
        half = max(180.0, (np.ptp(pts, axis=0).max() if len(pts) else 0) * 0.75)
        half = min(half, (x1 - x0) / 2, (y1 - y0) / 2)
        wx0 = min(max(cx - half, x0), x1 - 2 * half); wy0 = min(max(cy - half, y0), y1 - 2 * half)
        win = (int(wx0), int(wy0), int(2 * half), int(2 * half))
        crop = cv2.cvtColor(img[win[1]:win[1] + win[3], win[0]:win[0] + win[2]], cv2.COLOR_BGR2RGB)
        rad = win[2] * 0.016 * MARK
        for r, key in enumerate(keys):
            names, arr, _own = data[key]
            ax = axs[r, c]
            ax.imshow(crop, extent=(win[0], win[0] + win[2], win[1] + win[3], win[1]))
            for k, n in enumerate(names):
                p = arr[FRAME, c, k]
                if not np.isnan(p[0]):
                    ax.add_patch(plt.Circle((p[0], p[1]), rad, facecolor=KP_COLORS.get(n, "#111111"), edgecolor="white", linewidth=0.9))
            ax.set_xlim(win[0], win[0] + win[2]); ax.set_ylim(win[1] + win[3], win[1])
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(LABEL.get(key, key), fontsize=8)
            if r == 0:
                ax.set_title(f"cam {c}  (zoom {win[2]} px)", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=KP_COLORS[n], markeredgecolor="white", markersize=8, label=n) for n in KP_COLORS]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="#111111", markeredgecolor="white", markersize=8, label="other node names (YOLO)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"Fig. {no} · {title}. Frame {FRAME}; every row shows the same zoom window per camera (centred on the representative SLEAP animal)", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(a.out / fname, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    check_palette()
    a.out.mkdir(parents=True, exist_ok=True)

    paths = {"server_sleap1423": a.server}
    paths.update({p.name[:-8]: p for p in sorted(a.results.glob("*.json.gz"))})
    data = {k: load(p) for k, p in paths.items()}
    kp_names = list(KP_COLORS)
    same = [k for k in SAME_NODES if k in data and data[k][0] == kp_names]  # models whose node names are a subset of the 11
    assert "server_sleap1423" in same and REP in same, "server reference or representative SLEAP missing"
    r1_r2 = [(REP, "sleap1423_mps_r2")]
    sleap_pairs = [p for p in [("server_sleap1423", REP), (REP, "sleap1501single_mps")] if p[0] in same and p[1] in same]
    cross_pairs = [p for p in [(REP, "lp_reorganize_mps"), (REP, "lp_khu_mps"), ("lp_reorganize_mps", "lp_khu_mps")] if p[0] in same and p[1] in same]
    pairs = sleap_pairs + [p for p in r1_r2 if p[1] in same] + cross_pairs

    rows, dists = [], {}
    for x, y in pairs:
        dist, only_x, only_y = pair_stats(data[x][1], data[y][1])
        dists[(x, y)] = dist
        for k, n in enumerate(kp_names):
            if n not in data[x][2] or n not in data[y][2]:
                continue  # node missing in one model: no pair, not a zero
            v = dist[..., k][~np.isnan(dist[..., k])]
            assert (v >= 0).all()
            q = np.percentile(v, [50, 90, 99]) if v.size else [np.nan] * 3
            rows.append((x, y, n, v.size, q[0], q[1], q[2], v.max() if v.size else np.nan, int(only_x[k]), int(only_y[k])))
    with open(a.out / "per_keypoint_table.md", "w") as f:
        f.write("| A | B | keypoint | n | median px | p90 px | p99 px | max px | only A | only B |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {LABEL[r[0]]} | {LABEL[r[1]]} | {r[2]} | {r[3]} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]} | {r[9]} |\n")

    vmax = {}
    vmax["sleap"] = distance_figs(a, data, kp_names, sleap_pairs, {p: dists[p] for p in sleap_pairs}, (1, 2), "SLEAP variants")
    if cross_pairs:
        vmax["cross"] = distance_figs(a, data, kp_names, cross_pairs, {p: dists[p] for p in cross_pairs}, (5, 6), "Across model families")

    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, img = cap.read()
    assert ok, "frame not readable"
    overlay(a, data, [REP, "lp_reorganize_mps", "lp_khu_mps", "yolo_avatar3d_mps"], img, 3, "Model families (SLEAP = representative 1423-1)", "fig3_overlay_families.png")
    overlay(a, data, ["server_sleap1423", REP, "sleap1501single_mps"], img, 4, "SLEAP variants", "fig4_overlay_sleap.png")

    with open(a.out / "present_rate_table.md", "w") as f:  # output / 240 frames x 5 cams, per model and keypoint
        f.write("| model | " + " | ".join(kp_names) + " |\n|---|" + "---:|" * len(kp_names) + "\n")
        for key in same:
            arr = data[key][1]
            f.write(f"| {LABEL[key]} | " + " | ".join(f"{100 * np.mean(~np.isnan(arr[..., k, 0])):.1f}" if kp_names[k] in data[key][2] else "-" for k in range(len(kp_names))) + " |\n")
    s = data[REP][1]  # self-check: a result against itself is exactly zero
    z, _, _ = pair_stats(s, s)
    assert np.nanmax(z) == 0.0
    print(json.dumps({"pairs": [[LABEL[x], LABEL[y]] for x, y in pairs], "models": same, "other_nodes": [k for k in paths if k in data and k not in same], "vmax_p99": vmax}))


if __name__ == "__main__":
    main()
