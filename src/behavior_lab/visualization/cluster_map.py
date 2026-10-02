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


def pose_embedding(keypoints: np.ndarray) -> np.ndarray:
    """Method-neutral 2D axes for one recording: PCA of posture and speed.

    Features need no alignment and no skeleton: all pairwise joint distances, joint heights
    (third coordinate, when present) and centroid speed, each standardized. No discovery
    method in the grid clusters on these axes, so no method is favored when all are drawn on
    them. Linear, so distances on the plane are meaningful up to the variance it keeps.
    Returns ``(T, 2)``; ``pose_embedding.explained`` holds the last call's variance ratio.
    """
    from sklearn.decomposition import PCA

    kp = np.asarray(keypoints, dtype=float)
    i, j = np.triu_indices(kp.shape[1], k=1)
    feats = [np.linalg.norm(kp[:, i] - kp[:, j], axis=-1)]
    if kp.shape[2] >= 3:
        feats.append(kp[:, :, 2])
    speed = np.linalg.norm(np.diff(kp.mean(axis=1), axis=0, prepend=kp[:1].mean(axis=1)), axis=1)
    feats.append(speed[:, None])
    X = np.concatenate(feats, axis=1)
    X = (X - np.median(X, axis=0)) / (X.std(axis=0) + 1e-9)
    X = np.clip(X, -5, 5)  # tracking-loss frames would otherwise set the axes
    pca = PCA(n_components=2, random_state=0).fit(X)
    pose_embedding.explained = float(pca.explained_variance_ratio_.sum())
    return pca.transform(X)


def plot_method_maps(embedding: np.ndarray, seqs: dict[str, np.ndarray], title: str = ""):
    """The same embedding once per method, colored by that method's labels (small multiples)."""
    import matplotlib.pyplot as plt

    from .agreement import stretch_labels

    n = len(seqs)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows), squeeze=False,
                             constrained_layout=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (name, lab) in zip(axes.ravel(), seqs.items()):
        plot_cluster_map(embedding, stretch_labels(lab, len(embedding)), name, ax=ax)
    fig.suptitle(title, fontsize=10)
    return fig


def cluster_sizes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``(ids, share of labelled steps, noise share)``, largest cluster first."""
    labels = np.asarray(labels)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts)
    return ids[order], counts[order] / len(labels), float((labels < 0).mean())


def plot_cluster_sizes(seqs: dict[str, np.ndarray], title: str = ""):
    """Per method: share of time in each cluster, largest first, with count and top-3 share."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(seqs), figsize=(3.3 * len(seqs), 2.8), squeeze=False,
                             constrained_layout=True, sharey=True)
    cmap = plt.get_cmap("tab20")
    for ax, (name, lab) in zip(axes[0], seqs.items()):
        ids, share, noise = cluster_sizes(lab)
        all_ids = sorted(ids.tolist())  # same color per id as plot_cluster_map
        ax.bar(range(len(ids)), share, color=[cmap(all_ids.index(c) % 20) for c in ids])
        if len(ids) <= 20:
            ax.set_xticks(range(len(ids)))
            ax.set_xticklabels(ids, fontsize=6)
        else:
            ax.set_xticks([])
        note = f", noise {noise:.0%}" if noise else ""
        ax.set_title(f"{name}\n{len(ids)} clusters, top 3 = {share[:3].sum():.0%}{note}",
                     fontsize=8)
        ax.set_xlabel("cluster (by size)", fontsize=8)
    axes[0][0].set_ylabel("share of time")
    fig.suptitle(title, fontsize=10)
    return fig


__all__ = ["cluster_sizes", "plot_cluster_map", "plot_cluster_sizes", "plot_method_maps",
           "plot_subtle_cluster_map", "pose_embedding", "transition_matrix"]
