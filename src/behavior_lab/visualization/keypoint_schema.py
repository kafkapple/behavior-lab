"""Keypoint schema figure: one real frame, joints as colored points, bones as lines.

A recording's mean pose is meaningless while the animal moves and turns, so the figure shows
the frame whose pairwise joint distances are closest to the recording's median.
"""
from __future__ import annotations

import numpy as np


def joint_color(k: int) -> str:
    """Fixed color of joint ``k`` (tab20 by index): the schema figure, its table and the player
    use the same one, so joint names are written once, in the table."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    return to_hex(colormaps["tab20"](k % 20))


def representative_frame(keypoints: np.ndarray) -> int:
    kp = np.asarray(keypoints, dtype=float)
    i, j = np.triu_indices(kp.shape[1], k=1)
    d = np.linalg.norm(kp[:, i] - kp[:, j], axis=-1)
    return int(np.argmin(np.abs(d - np.median(d, axis=0)).sum(axis=1)))


def plot_keypoint_schema(keypoints: np.ndarray, joint_names: list[str],
                         edges: list[tuple[int, int]], title: str = ""):
    """Top view (first two coordinates) and, for 3D data, a side view (long horizontal axis
    of that frame against the third coordinate). Without ``edges`` only points are drawn."""
    import matplotlib.pyplot as plt

    kp = np.asarray(keypoints, dtype=float)
    f = representative_frame(kp)
    pose = kp[f]
    assert len(joint_names) == pose.shape[0], "joint names do not match the keypoint count"
    views = [("top view (coordinates 0, 1)", pose[:, 0], pose[:, 1])]
    if pose.shape[1] >= 3:
        xy = pose[:, :2] - pose[:, :2].mean(axis=0)
        axis = np.linalg.svd(xy, full_matrices=False)[2][0]
        views.append(("side view (long axis, coordinate 2)", xy @ axis, pose[:, 2]))
    fig, axes = plt.subplots(1, len(views), figsize=(5.2 * len(views), 4.2), squeeze=False,
                             constrained_layout=True)
    for ax, (name, x, y) in zip(axes[0], views):
        for a, b in edges:
            ax.plot([x[a], x[b]], [y[a], y[b]], color="0.5", lw=1.5, zorder=1)
        ax.scatter(x, y, s=60, c=[joint_color(k) for k in range(len(x))], zorder=2,
                   edgecolors="white", linewidths=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{title}: frame {f}", fontsize=10)
    return fig


__all__ = ["joint_color", "plot_keypoint_schema", "representative_frame"]
