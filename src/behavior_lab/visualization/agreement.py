"""Label-sequence agreement between discovery runs (methods, repeats, dataset slices).

One module for the notebook and the HTML report, so both show the same numbers.
ARI alone is misleading when cluster counts differ a lot (a clean split of one label into
finer ones lowers it), so AMI, homogeneity and a circular-shift null are reported with it.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
)


def stretch_labels(labels: np.ndarray, T: int) -> np.ndarray:
    """Nearest-neighbour resample to length T (methods that emit bins, e.g. B-SOiD 10 Hz)."""
    labels = np.asarray(labels)
    return labels if len(labels) == T else labels[np.linspace(0, len(labels) - 1, T).astype(int)]


def label_agreement(seqs: dict[str, np.ndarray], *, n_shifts: int = 20,
                    seed: int = 0) -> dict[str, object]:
    """Pairwise agreement between label sequences that cover the same time span.

    Returns ``names`` and square matrices: ``ari``, ``ami``, ``homogeneity`` (entry [i, j] = 1
    when every label of column j falls inside a single label of row i, so a high [i, j] with
    a low [j, i] means j is a finer split of i; asymmetric) and ``null_ari`` (mean ARI after circularly shifting one sequence by a random
    offset of at least 10% of its length, i.e. what unrelated sequences with the same
    label statistics give).
    """
    names = list(seqs)
    T = max(len(v) for v in seqs.values())
    lab = [stretch_labels(seqs[n], T) for n in names]
    n = len(names)
    out = {k: np.eye(n) for k in ("ari", "ami", "homogeneity")}
    null = np.zeros((n, n))
    rng = np.random.default_rng(seed)
    shifts = rng.integers(max(1, T // 10), max(2, T - T // 10), size=n_shifts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out["homogeneity"][i, j] = homogeneity_score(lab[i], lab[j])
            if i < j:
                out["ari"][i, j] = out["ari"][j, i] = adjusted_rand_score(lab[i], lab[j])
                out["ami"][i, j] = out["ami"][j, i] = adjusted_mutual_info_score(lab[i], lab[j])
                null[i, j] = null[j, i] = float(np.mean(
                    [adjusted_rand_score(lab[i], np.roll(lab[j], int(s))) for s in shifts]))
    assert np.allclose(out["ari"], out["ari"].T)
    return {"names": names, "n_frames": T, **out, "null_ari": null}


def _heatmap(ax, names: list[str], M: np.ndarray, title: str) -> None:
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if M[i, j] > 0.6 else "white")
    ax.set_title(title, fontsize=10)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def plot_label_agreement(seqs: dict[str, np.ndarray], title: str = "", *, fps: float | None = None):
    """Ethogram per sequence plus ARI and AMI heatmaps. Returns ``(fig, agreement_dict)``."""
    import matplotlib.pyplot as plt

    agr = label_agreement(seqs)
    names, T = agr["names"], agr["n_frames"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 0.5 * len(names) + 3.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.2, 1, 1]})
    # colors are per-sequence ids: the same color in two rows does not mean the same behavior
    axes[0].imshow(np.stack([stretch_labels(seqs[n], T) % 20 for n in names]), aspect="auto",
                   cmap="tab20", interpolation="nearest",
                   extent=(0, T / fps if fps else T, len(names), 0))
    axes[0].set_yticks(np.arange(len(names)) + 0.5)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("time (s)" if fps else "frame")
    axes[0].set_title(f"{title}: label sequences", fontsize=10)
    _heatmap(axes[1], names, agr["ari"], "ARI")
    _heatmap(axes[2], names, agr["ami"], "AMI")
    return fig, agr


__all__ = ["label_agreement", "plot_label_agreement", "stretch_labels"]
