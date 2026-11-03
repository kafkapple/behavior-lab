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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from pkg_groups import KP_COLORS, MARKER_R_REL, check_palette  # noqa: E402

W, H = 3600, 2000  # composite size (ffprobe 261103); json coordinates are normalized by it
CELLS = [[0, 0, 1200, 1000], [1200, 0, 2400, 1000], [2400, 0, 3600, 1200], [0, 1000, 1200, 2000], [1200, 1000, 2400, 2000]]
FRAME = 120  # fixed before looking at results: middle of the 240 frames
SAME_NODES = ["server_sleap1423", "sleap1423_mps_r1", "sleap1423_mps_r2", "sleap1501_mps", "lp_reorganize_cpu", "lp_khu_cpu"]
LABEL = {"server_sleap1423": "server SLEAP 1423-1", "sleap1423_mps_r1": "Mac SLEAP 1423-1 r1", "sleap1423_mps_r2": "Mac SLEAP 1423-1 r2",
         "sleap1501_mps": "Mac SLEAP 1501", "lp_reorganize_cpu": "Mac LP reorganize", "lp_khu_cpu": "Mac LP KHU", "yolo_mps": "Mac YOLO KHU (9 pt)"}


def load(path: Path):
    d = json.load(gzip.open(path))
    names = d["node"]["id"]
    a = np.full((len(d["keypoint"]), 5, len(names), 3), np.nan)
    for f, cams in d["keypoint"].items():
        for c, nodes in cams.items():
            for k, (x, y, s) in nodes.items():
                if not (x == 0 and y == 0):
                    a[int(f), int(c), int(k)] = (x * W, y * H, s)
    if set(names) == set(KP_COLORS):  # node order differs between files: index by name
        order = [names.index(n) for n in KP_COLORS]
        names, a = list(KP_COLORS), a[:, :, order]
    return names, a


def pair_stats(a, b):
    dist = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)  # (F, cam, node), NaN where either side has no output
    only_a = (~np.isnan(a[..., 0]) & np.isnan(b[..., 0])).sum((0, 1))
    only_b = (np.isnan(a[..., 0]) & ~np.isnan(b[..., 0])).sum((0, 1))
    return dist, only_a, only_b


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
    same = [k for k in SAME_NODES if k in data and data[k][0] == kp_names]
    assert "server_sleap1423" in same, "server reference missing or node names differ"
    pairs = [("server_sleap1423", "sleap1423_mps_r1"), ("sleap1423_mps_r1", "sleap1423_mps_r2")]
    pairs += [p for p in itertools.combinations([k for k in same if k not in ("server_sleap1423", "sleap1423_mps_r2")], 2)]
    pairs = [p for p in pairs if p[0] in same and p[1] in same]

    rows, dists = [], {}
    for x, y in pairs:
        dist, only_x, only_y = pair_stats(data[x][1], data[y][1])
        dists[(x, y)] = dist
        for k, n in enumerate(kp_names):
            v = dist[..., k][~np.isnan(dist[..., k])]
            assert (v >= 0).all()
            q = np.percentile(v, [50, 90, 99]) if v.size else [np.nan] * 3
            rows.append((x, y, n, v.size, q[0], q[1], q[2], v.max() if v.size else np.nan, int(only_x[k]), int(only_y[k])))
    with open(a.out / "per_keypoint_table.md", "w") as f:
        f.write("| A | B | keypoint | n | median px | p90 px | p99 px | max px | only A | only B |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {LABEL[r[0]]} | {LABEL[r[1]]} | {r[2]} | {r[3]} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]} | {r[9]} |\n")

    # Fig 1: distance distribution per keypoint, one row per pair, shared y axis
    allv = np.concatenate([d[~np.isnan(d)] for d in dists.values()])
    fig, axs = plt.subplots(len(pairs), 1, figsize=(11, 2.1 * len(pairs)), sharex=True, sharey=True, squeeze=False)
    for ax, (x, y) in zip(axs[:, 0], pairs):
        d = dists[(x, y)]
        for k, n in enumerate(kp_names):
            v = d[..., k][~np.isnan(d[..., k])]
            if v.size:
                bp = ax.boxplot(v, positions=[k], widths=0.6, showfliers=True, patch_artist=True, flierprops=dict(markersize=2, alpha=0.4, markeredgecolor=KP_COLORS[n]),
                                medianprops=dict(color="black"))
                bp["boxes"][0].set(facecolor=KP_COLORS[n], alpha=0.85)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, allv.max() * 1.2)
        ax.set_title(f"{LABEL[x]}  vs  {LABEL[y]}", fontsize=9, loc="left")
        ax.set_ylabel("distance (px)")
    axs[-1, 0].set_xticks(range(len(kp_names)), kp_names, rotation=30, ha="right")
    fig.suptitle("Fig. 1 · Per-keypoint distance, all frames x 5 cameras (composite px, symlog y, shared axes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "fig1_distance_by_keypoint.png", dpi=110)
    plt.close(fig)

    # Fig 2: keypoint x frame heatmap of the largest distance over cameras, shared colour scale
    vmax = float(np.nanpercentile(allv, 99))
    fig, axs = plt.subplots(len(pairs), 1, figsize=(11, 1.9 * len(pairs)), sharex=True, squeeze=False)
    for ax, (x, y) in zip(axs[:, 0], pairs):
        m = np.nanmax(np.where(np.isnan(dists[(x, y)]), -1, dists[(x, y)]), axis=1).T  # (node, frame), -1 = no pair
        im = ax.imshow(np.ma.masked_less(m, 0), aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(kp_names)), kp_names, fontsize=6)
        ax.set_title(f"{LABEL[x]}  vs  {LABEL[y]}", fontsize=9, loc="left")
    axs[-1, 0].set_xlabel("frame")
    fig.colorbar(im, ax=axs[:, 0].tolist(), label=f"max over cameras (px), colour limit = global p99 ({vmax:.1f})", shrink=0.6)
    fig.suptitle("Fig. 2 · Distance by frame (grey = no pair)", fontsize=10)
    fig.savefig(a.out / "fig2_distance_by_frame.png", dpi=110)
    plt.close(fig)

    # Fig 3: same frame, same crops, every model; dots = KP_COLORS (neutral white for models with other node names)
    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, img = cap.read()
    assert ok, "frame not readable"
    tile = 420
    keys = [k for k in paths if k in data]
    fig, axs = plt.subplots(len(keys), 5, figsize=(5 * 3.0, len(keys) * 2.6), squeeze=False)
    for r, key in enumerate(keys):
        names, arr = data[key]
        for c, (x0, y0, x1, y1) in enumerate(CELLS):
            ax = axs[r, c]
            ax.imshow(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            rad = MARKER_R_REL * (x1 - x0) * 1.0
            for k, n in enumerate(names):
                p = arr[FRAME, c, k]
                if np.isnan(p[0]):
                    continue
                col = KP_COLORS.get(n, "#FFFFFF")
                ax.add_patch(plt.Circle((p[0] - x0, p[1] - y0), rad, facecolor=col, edgecolor="white", linewidth=0.8))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(LABEL.get(key, key), fontsize=7)
            if r == 0:
                ax.set_title(f"cam {c}", fontsize=8)
    fig.suptitle(f"Fig. 3 · Frame {FRAME}, same crops, one colour per keypoint (white = other node names)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "fig3_overlay_frame.png", dpi=110)
    plt.close(fig)

    # present-rate table (output / 240 frames x 5 cams) per model and keypoint
    with open(a.out / "present_rate_table.md", "w") as f:
        f.write("| model | " + " | ".join(kp_names) + " |\n|---|" + "---:|" * len(kp_names) + "\n")
        for key in same:
            arr = data[key][1]
            f.write(f"| {LABEL[key]} | " + " | ".join(f"{100 * np.mean(~np.isnan(arr[..., k, 0])):.1f}" for k in range(len(kp_names))) + " |\n")
    # self-check: a result against itself is exactly zero
    s = data[same[0]][1]
    z, _, _ = pair_stats(s, s)
    assert np.nanmax(z) == 0.0
    print(json.dumps({"pairs": len(pairs), "models": same, "other_nodes": [k for k in keys if k not in same], "vmax_p99": vmax}))


if __name__ == "__main__":
    main()
