"""One-off fix — regenerate the UMAP per-method plot for the latest v2 run.

The first compare_methods_v2.py run crashed on a length mismatch
(HMM embedding length != pooled length). This script reads what is on disk
(labels_v2.parquet) and re-renders just the 3-panel UMAP plot, plus a v1/v2
side-by-side metric chart for the notebook.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(k, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ROOT = REPO / "data" / "calms21_behavior_discovery"


def latest_v2() -> Path:
    runs = sorted([p for p in (REPO / "outputs" / "calms21_behavior_discovery" / "results_v2").iterdir() if p.is_dir()])
    return runs[-1]


def main():
    run = latest_v2()
    print(f"run: {run.name}")
    df = pd.read_parquet(run / "labels_v2.parquet")

    # Build 2-D embeddings on demand via UMAP from the dyadic feature matrix.
    # (we didn't persist the SUBTLE/B-SOiD/HMM embeddings separately).
    info = pd.read_csv(ROOT / "data_info_v2.csv")
    sessions = info["session"].tolist()
    feats = np.concatenate([
        pd.read_csv(ROOT / "raw_csv_v2" / f"{s}.csv",
                    header=None, dtype=np.float32).values
        for s in sessions
    ])[: len(df)]

    from umap import UMAP
    print(f"UMAP on {feats.shape} ...")
    emb = UMAP(n_components=2, random_state=0, n_jobs=1,
               n_neighbors=30, min_dist=0.1).fit_transform(feats)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    rng = np.random.default_rng(0)
    n = min(20000, len(df))
    idx = rng.choice(len(df), n, replace=False)

    for ax, name in zip(axes, ["subtle", "bsoid", "pca_hmm"]):
        lab = df[name].values[idx]
        sns.scatterplot(x=emb[idx, 0], y=emb[idx, 1], hue=lab,
                        palette="tab20", s=2, ax=ax, legend=False)
        ax.set_title(f"{name.upper()}  (K={len(np.unique(lab[lab >= 0]))})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("v2 — shared UMAP of 24-dim dyadic features, coloured by each method's clusters")
    fig.tight_layout()
    fig.savefig(run / "compare_umap_by_method_v2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {run / 'compare_umap_by_method_v2.png'}")

    # GT colored
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    gt = df["gt"].values[idx]
    palette = {0: "#d62728", 1: "#1f77b4", 2: "#2ca02c", 3: "#bbbbbb"}
    names = ["attack", "investigation", "mount", "other"]
    for cls in range(4):
        m = gt == cls
        ax.scatter(emb[idx][m, 0], emb[idx][m, 1], s=2, c=palette[cls],
                   label=names[cls], alpha=0.5 if cls == 3 else 0.95)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("v2 dyadic UMAP coloured by CalMS21 GT")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=3)
    fig.tight_layout(); fig.savefig(run / "compare_umap_by_gt_v2.png",
                                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {run / 'compare_umap_by_gt_v2.png'}")

    # v1 vs v2 metric comparison bar
    v2 = pd.read_csv(run / "metrics_v2.csv")
    v1_runs = sorted([p for p in (REPO / "outputs" / "calms21_behavior_discovery" / "results").iterdir() if p.is_dir()])
    if v1_runs:
        v1 = pd.read_csv(v1_runs[-1] / "compare_metrics.csv")
        merged = v1.set_index("method")[["ari", "nmi"]].rename(
                    columns={"ari": "ari_v1", "nmi": "nmi_v1"})
        v2_idx = v2.set_index("method")[["ari", "nmi"]]
        v2_idx.index = v2_idx.index.str.replace("_best", "", regex=False)
        merged = merged.join(v2_idx.rename(columns={"ari": "ari_v2", "nmi": "nmi_v2"}))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
        merged[["ari_v1", "ari_v2"]].plot.bar(ax=axes[0], color=["#bbbbbb", "#1f77b4"])
        axes[0].set_title("ARI: v1 (resident pixel) → v2 (ego-centric dyadic)")
        axes[0].set_ylabel("Adjusted Rand Index"); axes[0].axhline(0, color="k", lw=0.4)
        merged[["nmi_v1", "nmi_v2"]].plot.bar(ax=axes[1], color=["#bbbbbb", "#2ca02c"])
        axes[1].set_title("NMI: v1 → v2")
        axes[1].set_ylabel("Normalised Mutual Info")
        for ax in axes:
            for c in ax.containers:
                ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
            ax.tick_params(axis="x", rotation=0)
        fig.suptitle("Effect of social/dyadic feature engineering")
        fig.tight_layout(); fig.savefig(run / "v1_vs_v2.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {run / 'v1_vs_v2.png'}")


if __name__ == "__main__":
    main()
