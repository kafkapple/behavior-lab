"""Embedding visualization for behavior discovery results."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def plot_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "2D Embedding",
    figsize: tuple[int, int] = (8, 6),
    alpha: float = 0.5,
    s: float = 1.0,
    cmap: str = "tab20",
    class_names: Sequence[str] | None = None,
    ax=None,
    save_path: str | None = None,
):
    """Plot 2D embedding scatter colored by labels.

    Args:
        embeddings: (N, 2) or (N, D) — first 2 dims used
        labels: (N,) integer labels for coloring
        title: Plot title
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        cmap: Colormap name
        class_names: Optional label names for legend
        ax: Matplotlib axes (created if None)
        save_path: If set, save figure to this path
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    x, y = embeddings[:, 0], embeddings[:, 1]

    if labels is not None:
        scatter = ax.scatter(x, y, c=labels, cmap=cmap, alpha=alpha, s=s, rasterized=True)
        if class_names is not None:
            unique = sorted(set(labels))
            handles = []
            for i, u in enumerate(unique):
                name = class_names[u] if u < len(class_names) else str(u)
                h = ax.scatter([], [], c=[scatter.cmap(scatter.norm(u))], label=name, s=10)
                handles.append(h)
            ax.legend(handles=handles, loc="best", fontsize=7, markerscale=2)
    else:
        ax.scatter(x, y, alpha=alpha, s=s, rasterized=True)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_embedding_3d(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "3D Embedding",
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    s: float = 1.0,
    cmap: str = "tab20",
    save_path: str | None = None,
):
    """Plot 3D embedding scatter.

    Args:
        embeddings: (N, 3+) — first 3 dims used
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]

    if labels is not None:
        ax.scatter(x, y, z, c=labels, cmap=cmap, alpha=alpha, s=s)
    else:
        ax.scatter(x, y, z, alpha=alpha, s=s)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
