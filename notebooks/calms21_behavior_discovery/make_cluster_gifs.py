"""Per-cluster GIF animations — decoupled from modeling.

Reads labels_v2.parquet, picks N representative clusters per method,
extracts a contiguous bout from each, and animates both mice's skeletons.

Per MoA Layer 2 consensus: keep this completely separate from the modeling
pipeline (modeling outputs (label, timestamp); GIF script is read-only).

Usage:
    python notebooks/calms21_behavior_discovery/make_cluster_gifs.py \
        --method subtle --top-k 5 --bout-frames 90

Output:
    data/calms21_behavior_discovery/results_v2/<run>/gifs/<method>_cluster_<id>.gif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ROOT = REPO / "data" / "calms21_behavior_discovery"
RAW_JSON = REPO / "data" / "calms21" / "task1_classic_classification" / "calms21_task1_train.json"

# 7-keypoint mouse skeleton edges (CalMS21 layout)
EDGES = [(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
JOINT_NAMES = ["nose", "L_ear", "R_ear", "neck", "L_hip", "R_hip", "tail"]


def load_pooled_raw_keypoints(sessions: list[str]) -> np.ndarray:
    """Stitch all sessions back into one (T_total, 2_mice, 7_kp, 2_xy) array."""
    with open(RAW_JSON) as f:
        bundle = json.load(f)
    block = bundle[next(iter(bundle))]
    name_to_key = {key.split("/")[-1]: key for key in block.keys()}

    chunks = []
    for s in sessions:
        kp = np.array(block[name_to_key[s]]["keypoints"], dtype=np.float32)
        # original layout: (T, 2_mice, 2_xy, 7_kp) → (T, 2_mice, 7_kp, 2_xy)
        kp = kp.transpose(0, 1, 3, 2)
        chunks.append(kp)
    return np.concatenate(chunks, axis=0)


def find_bouts(labels: np.ndarray, cluster: int, min_len: int = 60):
    """Return list of (start, end) contiguous runs where labels == cluster."""
    mask = labels == cluster
    bouts = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                bouts.append((i, j))
            i = j
        else:
            i += 1
    return bouts


def animate_bout(kp_slice: np.ndarray, label_id: int, method: str,
                 frame_offset: int, save_path: Path, fps: int = 20):
    """kp_slice shape: (T, 2_mice, 7_kp, 2_xy)."""
    T = kp_slice.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    ax.set_facecolor("#fafafa")
    all_xy = kp_slice.reshape(-1, 2)
    pad = 20
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].max() + pad, all_xy[:, 1].min() - pad)  # image-down y
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    colors = ["#d62728", "#1f77b4"]  # mouse 0 red, mouse 1 blue
    lines = [
        [ax.plot([], [], "-", lw=2, color=colors[m], alpha=0.85)[0] for _ in EDGES]
        for m in range(2)
    ]
    scats = [ax.scatter([], [], s=35, color=colors[m], zorder=3,
                        edgecolor="white", linewidth=0.8) for m in range(2)]
    title = ax.set_title("")

    def update(t):
        for m in range(2):
            kp = kp_slice[t, m]   # (7, 2)
            scats[m].set_offsets(kp)
            for ln, (a, b) in zip(lines[m], EDGES):
                ln.set_data([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]])
        title.set_text(f"{method}  cluster {label_id}  "
                       f"frame {frame_offset + t}/{frame_offset + T}")
        return [title, *scats] + sum(lines, [])

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="results_v2/<run> dir; defaults to the latest")
    ap.add_argument("--method", default="subtle",
                    choices=["subtle", "bsoid", "pca_hmm"])
    ap.add_argument("--top-k", type=int, default=5,
                    help="number of largest clusters to render")
    ap.add_argument("--bout-frames", type=int, default=90,
                    help="frames per GIF (~4.5 s at 20 fps)")
    args = ap.parse_args()

    runs = sorted([p for p in (REPO / "outputs" / "calms21_behavior_discovery" / "results_v2").iterdir() if p.is_dir()])
    assert runs, "no results_v2 run — execute compare_methods_v2.py first"
    run_dir = args.run_dir or runs[-1]
    print(f"run: {run_dir.name}")

    pq = run_dir / "labels_v2.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(run_dir / "labels_v2.csv")

    # rebuild full keypoints from the same session order used in v2 prep
    info = pd.read_csv(ROOT / "data_info_v2.csv")
    sessions = info["session"].tolist()
    kp_pooled = load_pooled_raw_keypoints(sessions)[: len(df)]
    print(f"pooled keypoints: {kp_pooled.shape}    labels rows: {len(df)}")

    labels = df[args.method].values
    valid = labels != -1
    uniq, counts = np.unique(labels[valid], return_counts=True)
    order = uniq[np.argsort(-counts)][: args.top_k]
    print(f"top-{args.top_k} {args.method} clusters: {order.tolist()}  "
          f"(sizes: {counts[np.argsort(-counts)][:args.top_k].tolist()})")

    out = run_dir / "gifs"
    out.mkdir(exist_ok=True)
    for cluster in order:
        bouts = find_bouts(labels, int(cluster), min_len=args.bout_frames)
        if not bouts:
            print(f"  cluster {cluster}: no contiguous bout >= {args.bout_frames}f — skip")
            continue
        # pick the median-length bout (avoids edge cases of very short flashes)
        bouts.sort(key=lambda b: b[1] - b[0])
        start, end = bouts[len(bouts) // 2]
        end = min(start + args.bout_frames, end)
        gif_path = out / f"{args.method}_cluster_{int(cluster):02d}.gif"
        animate_bout(kp_pooled[start:end], int(cluster), args.method, start, gif_path)
        print(f"  cluster {cluster:>3}: bout [{start}, {end})  saved {gif_path.name}")


if __name__ == "__main__":
    main()
