"""Skeleton visualization — static plots and animations."""
from __future__ import annotations

from typing import Optional

import numpy as np


def plot_skeleton(
    keypoints: np.ndarray,
    skeleton=None,
    frame: int = 0,
    figsize: tuple[int, int] = (6, 6),
    title: str = "",
    joint_size: float = 30,
    limb_width: float = 1.5,
    ax=None,
    save_path: str | None = None,
):
    """Plot a single skeleton frame.

    Args:
        keypoints: (T, K, D) or (K, D) keypoint array
        skeleton: SkeletonDefinition for edges and names
        frame: Frame index to plot (if keypoints is 3D)
        figsize: Figure size
        title: Plot title
        joint_size: Scatter point size for joints
        limb_width: Line width for limbs
        ax: Matplotlib axes (created if None)
        save_path: If set, save figure
    """
    import matplotlib.pyplot as plt

    if keypoints.ndim == 3:
        kp = keypoints[frame]
    else:
        kp = keypoints

    K, D = kp.shape
    is_3d = D >= 3

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot joints
    if is_3d:
        ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], s=joint_size, zorder=5)
    else:
        ax.scatter(kp[:, 0], kp[:, 1], s=joint_size, zorder=5)

    # Plot edges
    if skeleton is not None:
        for i, j in skeleton.edges:
            if i < K and j < K:
                if is_3d:
                    ax.plot(
                        [kp[i, 0], kp[j, 0]],
                        [kp[i, 1], kp[j, 1]],
                        [kp[i, 2], kp[j, 2]],
                        linewidth=limb_width, color="gray",
                    )
                else:
                    ax.plot(
                        [kp[i, 0], kp[j, 0]],
                        [kp[i, 1], kp[j, 1]],
                        linewidth=limb_width, color="gray",
                    )

    ax.set_title(title or f"Frame {frame}")
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def animate_skeleton(
    keypoints: np.ndarray,
    skeleton=None,
    fps: float = 30.0,
    figsize: tuple[int, int] = (6, 6),
    title: str = "Skeleton Animation",
    save_path: str | None = None,
):
    """Create an animation of skeleton keypoints over time.

    Args:
        keypoints: (T, K, D) keypoint sequence
        skeleton: SkeletonDefinition for edges
        fps: Frames per second for animation
        figsize: Figure size
        title: Animation title
        save_path: If set, save as .mp4 or .gif

    Returns:
        matplotlib.animation.FuncAnimation object
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    T, K, D = keypoints.shape
    is_3d = D >= 3

    fig = plt.figure(figsize=figsize)
    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Compute bounds
    margin = 0.1
    mins = np.nanmin(keypoints, axis=(0, 1))
    maxs = np.nanmax(keypoints, axis=(0, 1))
    ranges = maxs - mins
    mins -= ranges * margin
    maxs += ranges * margin

    def update(frame):
        ax.clear()
        kp = keypoints[frame]

        if is_3d:
            ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], s=30, zorder=5)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
        else:
            ax.scatter(kp[:, 0], kp[:, 1], s=30, zorder=5)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])

        if skeleton is not None:
            for i, j in skeleton.edges:
                if i < K and j < K:
                    if is_3d:
                        ax.plot(
                            [kp[i, 0], kp[j, 0]],
                            [kp[i, 1], kp[j, 1]],
                            [kp[i, 2], kp[j, 2]],
                            linewidth=1.5, color="gray",
                        )
                    else:
                        ax.plot(
                            [kp[i, 0], kp[j, 0]],
                            [kp[i, 1], kp[j, 1]],
                            linewidth=1.5, color="gray",
                        )

        ax.set_title(f"{title} — Frame {frame}/{T}")
        ax.set_aspect("equal")

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)

    return anim
