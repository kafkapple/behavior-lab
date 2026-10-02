"""Cluster map: a 2D embedding colored by cluster, with centroids and transition arrows.

The layout follows the behavior maps of SUBTLE (Kwon et al., IJCV 2024): frames embedded in
2D, clusters as colored islands, and temporal links between clusters. Works for any method
that returns an embedding and per-frame labels in time order.
"""
from __future__ import annotations

import numpy as np


def transition_matrix(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalized transition probabilities between different consecutive labels.

    Returns ``(ids, P)``; self-transitions are dropped, so ``P[i, j]`` is the probability
    that a bout of ``ids[i]`` is followed by a bout of ``ids[j]``. Noise (-1) is ignored.
    """
    labels = np.asarray(labels)
    ids = np.array(sorted(set(labels.tolist()) - {-1}))
    index = {c: k for k, c in enumerate(ids)}
    counts = np.zeros((len(ids), len(ids)))
    for a, b in zip(labels[:-1], labels[1:]):
        if a != b and a in index and b in index:
            counts[index[a], index[b]] += 1
    rows = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)
    assert np.all((P.sum(axis=1) < 1 + 1e-9)) and np.all(np.diag(P) == 0)
    return ids, P


def plot_cluster_map(embedding: np.ndarray, labels: np.ndarray, title: str = "", *, ax=None,
                     min_transition: float = 0.15, max_points: int = 20000):
    """Scatter of ``embedding[:, :2]`` colored by label, centroid ids, transition arrows.

    Arrows connect cluster centroids for transitions with probability >= ``min_transition``;
    width scales with probability. Distances on a UMAP plane are not metric, so the arrows
    show which clusters follow each other, not how far apart they are.
    """
    import matplotlib.pyplot as plt

    emb, labels = np.asarray(embedding)[:, :2], np.asarray(labels)
    assert len(emb) == len(labels), f"embedding {len(emb)} vs labels {len(labels)}"
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    step = max(1, len(emb) // max_points)
    cmap = plt.get_cmap("tab20")
    ids, P = transition_matrix(labels)
    color = {c: cmap(k % 20) for k, c in enumerate(ids)}
    noise = labels[::step] == -1
    ax.scatter(*emb[::step][noise].T, s=3, color="0.8", linewidths=0)
    ax.scatter(*emb[::step][~noise].T, s=4, linewidths=0, alpha=0.6,
               c=[color[c] for c in labels[::step][~noise]])
    cent = {c: emb[labels == c].mean(axis=0) for c in ids}
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if P[i, j] >= min_transition:
                ax.annotate("", xy=cent[b], xytext=cent[a], zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color="0.25", alpha=0.7,
                                            lw=0.5 + 3 * P[i, j], shrinkA=7, shrinkB=7,
                                            connectionstyle="arc3,rad=0.15"))
    for c in ids:
        ax.text(*cent[c], str(c), ha="center", va="center", fontsize=8, weight="bold", zorder=4,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=color[c], lw=1.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title} ({len(ids)} clusters)", fontsize=10)
    return ax


def plot_subtle_cluster_map(embedding: np.ndarray, subclusters: np.ndarray,
                            superclusters: np.ndarray, title: str = ""):
    """SUBTLE's two levels side by side on one embedding: subclusters and superclusters.

    ``superclusters`` may be ``(T,)`` or ``(T, n_levels)``; the finest level is drawn.
    """
    import matplotlib.pyplot as plt

    sup = np.asarray(superclusters)
    if sup.ndim == 2:
        sup = sup[:, int(np.argmax([len(np.unique(sup[:, j])) for j in range(sup.shape[1])]))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    plot_cluster_map(embedding, subclusters, f"{title}: subclusters", ax=axes[0])
    plot_cluster_map(embedding, sup, f"{title}: superclusters", ax=axes[1])
    return fig


__all__ = ["plot_cluster_map", "plot_subtle_cluster_map", "transition_matrix"]
