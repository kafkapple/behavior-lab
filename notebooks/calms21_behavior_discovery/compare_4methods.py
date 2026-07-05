"""Final 4-method comparison including real kp-MoSeq.

Combines v2 metrics (SUBTLE / B-SOiD / PCA-HMM fallback) with kpms_real_metrics
(real keypoint-MoSeq AR-HMM) into one figure and table. Also computes
median bout-length per method — kp-MoSeq's sticky HDP should produce
the longest dwell times by construction.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "calms21_behavior_discovery"


def median_bout(labels: np.ndarray) -> float:
    """Median contiguous-run length (in frames). Drop noise (-1)."""
    valid = labels != -1
    if not valid.any():
        return 0.0
    runs = []
    cur = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1] and labels[i] != -1:
            cur += 1
        else:
            if labels[i - 1] != -1:
                runs.append(cur)
            cur = 1
    return float(np.median(runs)) if runs else 0.0


def main():
    runs = sorted([p for p in (REPO / "outputs" / "calms21_behavior_discovery" / "results_v2").iterdir() if p.is_dir()])
    run = runs[-1]
    print(f"v2 run: {run.name}")

    v2_metrics = pd.read_csv(run / "metrics_v2.csv")
    kpms_metrics = pd.read_csv(run / "kpms_real_metrics.csv")
    merged = pd.concat([v2_metrics, kpms_metrics], ignore_index=True)

    # Add median bout length — requires run_kpms_real.py to have produced labels_v3.parquet
    labels_path = run / "labels_v3.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} missing — run notebooks/calms21_behavior_discovery/"
            "run_kpms_real.py first")
    df = pd.read_parquet(labels_path)
    bout_map = {
        "SUBTLE": median_bout(df["subtle"].values),
        "B-SOiD_best": median_bout(df["bsoid"].values),
        "PCA-HMM": median_bout(df["pca_hmm"].values),
        "kp-MoSeq_real": median_bout(df["kpms_real"].values),
    }
    merged["median_bout_frames"] = merged["method"].map(bout_map)
    merged.to_csv(run / "metrics_v3.csv", index=False)
    print(merged.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    metrics_df = merged.set_index("method")
    metrics_df["ari"].plot.bar(ax=axes[0], color="#1f77b4")
    axes[0].set_title("ARI (vs CalMS21 GT)")
    axes[0].axhline(0, color="grey", lw=0.5)

    metrics_df["nmi"].plot.bar(ax=axes[1], color="#2ca02c")
    axes[1].set_title("NMI (vs CalMS21 GT)")

    metrics_df["median_bout_frames"].plot.bar(ax=axes[2], color="#d62728")
    axes[2].set_title("Median bout length (frames, @30 fps)")
    axes[2].set_ylabel("frames")

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)

    fig.suptitle("4-method comparison on CalMS21 dyadic features (v2 + real kp-MoSeq)",
                 y=1.02)
    fig.tight_layout(); fig.savefig(run / "metrics_v3.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {run / 'metrics_v3.png'}")
    print(f"saved: {run / 'metrics_v3.csv'}")


if __name__ == "__main__":
    main()
