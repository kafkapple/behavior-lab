"""Analysis visualization — transition matrices, bout durations, temporal rasters."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    class_names: Sequence[str] | None = None,
    title: str = "Behavior Transition Matrix",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    ax=None,
    save_path: str | None = None,
):
    """Plot transition probability matrix as a heatmap.

    Args:
        transition_matrix: (C, C) transition probability matrix
        class_names: Labels for rows/columns
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        ax: Matplotlib axes
        save_path: If set, save figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(transition_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Transition Probability")

    n = transition_matrix.shape[0]
    labels = class_names or [str(i) for i in range(n)]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = transition_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_bout_duration(
    bout_durations: dict[int, float],
    class_names: Sequence[str] | None = None,
    title: str = "Mean Bout Duration",
    figsize: tuple[int, int] = (8, 4),
    ax=None,
    save_path: str | None = None,
):
    """Bar plot of mean bout durations per behavior class.

    Args:
        bout_durations: {class_id: mean_duration_seconds}
        class_names: Optional label names
        title: Plot title
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    keys = sorted(bout_durations.keys())
    values = [bout_durations[k] for k in keys]
    labels = [class_names[k] if class_names and k < len(class_names) else str(k) for k in keys]

    bars = ax.bar(range(len(keys)), values, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title(title)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2f}s", ha="center", va="bottom", fontsize=8,
        )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_temporal_raster(
    labels: np.ndarray,
    fps: float = 30.0,
    class_names: Sequence[str] | None = None,
    title: str = "Behavioral Ethogram",
    figsize: tuple[int, int] = (14, 2),
    cmap: str = "tab20",
    ax=None,
    save_path: str | None = None,
):
    """Plot temporal raster / ethogram of behavior labels.

    Args:
        labels: (T,) per-frame behavior labels
        fps: Frames per second for x-axis in seconds
        class_names: Optional label names for legend
        title: Plot title
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    T = len(labels)
    time_sec = np.arange(T) / fps

    # Create a color-coded strip
    unique = sorted(set(labels))
    cmap_obj = plt.cm.get_cmap(cmap, len(unique))
    label_to_color = {l: i for i, l in enumerate(unique)}
    colors = np.array([label_to_color[l] for l in labels])

    ax.imshow(
        colors.reshape(1, -1),
        aspect="auto",
        cmap=cmap_obj,
        extent=[time_sec[0], time_sec[-1], 0, 1],
        interpolation="nearest",
    )

    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title(title)

    # Legend
    if class_names is not None:
        import matplotlib.patches as mpatches
        patches = []
        for l in unique:
            name = class_names[l] if l < len(class_names) else str(l)
            patches.append(mpatches.Patch(
                color=cmap_obj(label_to_color[l]), label=name
            ))
        ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=min(len(unique), 5))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
