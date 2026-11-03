"""Plot the human-label AMI of each grid method per recording (reads human_label_matched_k.json).

Two panels: stored cluster counts (k = 0 in the json) and k = 5. One dot = the median over 3 seeds of
one recording, the bar = the seed min to max of that recording. The line = median over recordings.

Usage: ``uv run python scripts/plot_human_label_score.py [--out PATH.png]``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "behavior_analysis_workbench" / "level_rescore" / "human_label_matched_k.json"
ORDER = [("subtle", "SUBTLE"), ("b_soid", "B-SOiD"), ("keypoint_moseq", "keypoint-MoSeq"),
         ("pca_hmm_moseq_fallback", "PCA+HMM"), ("kmeans_pca_umap", "k-means")]
COLORS = ["#c0392b", "#2e6da4", "#2e8b57", "#8e7cc3", "#7f8c8d"]
PANELS = [("0", "stored cluster counts (SUBTLE 4, B-SOiD 41, kpMoSeq 20, HMM 12, k-means 8)"),
          ("5", "matched k = 5 (largest 4 clusters kept, rest merged)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "behavior_analysis_workbench" / "level_rescore" / "human_label_ami.png"))
    cells = json.loads(SRC.read_text())["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    slices = sorted({c["slice"] for c in cells["0"]["subtle"]})
    for ax, (k, title) in zip(axes, PANELS):
        for i, ((m, name), col) in enumerate(zip(ORDER, COLORS)):
            per = {sl: [c["ami"] for c in cells[k][m] if c["slice"] == sl] for sl in slices}
            assert all(len(v) == 3 for v in per.values()), (m, per)
            med = np.array([np.median(v) for v in per.values()])
            lo = np.array([min(v) for v in per.values()])
            hi = np.array([max(v) for v in per.values()])
            x = i + np.linspace(-0.18, 0.18, len(slices))
            ax.vlines(x, lo, hi, color=col, alpha=0.35, lw=2)
            ax.scatter(x, med, color=col, s=28, zorder=3)
            ax.hlines(np.median(med), i - 0.3, i + 0.3, color=col, lw=2.5)
        ax.set_xticks(range(len(ORDER)), [n for _, n in ORDER], rotation=20)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AMI vs human label (annotated frames)")
    fig.suptitle("Human-label AMI, 5 SUBTLE y5a5 recordings: dot = recording (seed median), bar = seed range, line = median", fontsize=9)
    fig.tight_layout()
    out = Path(ap.parse_args().out)
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
