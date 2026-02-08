"""Skeleton visualization — static plots and animations with color support."""
from __future__ import annotations

import numpy as np

from .colors import (
    get_joint_colors,
    get_limb_colors,
    get_person_colors,
    get_joint_labels,
    _FALLBACK_COLOR,
)


def _adjust_color(hex_color: str, factor: float = 0.7) -> str:
    """Darken or lighten a hex color. factor < 1 darkens, > 1 lightens."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, max(0, int(r * factor)))
    g = min(255, max(0, int(g * factor)))
    b = min(255, max(0, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_skeleton(
    keypoints: np.ndarray,
    skeleton=None,
    frame: int = 0,
    figsize: tuple[int, int] = (6, 6),
    title: str = "",
    joint_size: float = 30,
    limb_width: float = 1.5,
    joint_colors: list[str] | None = None,
    limb_colors: list[str] | None = None,
    show_labels: bool = False,
    ax=None,
    save_path: str | None = None,
):
    """Plot a single skeleton frame with body-part coloring.

    Args:
        keypoints: (T, K, D) or (K, D) keypoint array.
            For multi-person skeletons, K = num_persons * joints_per_person.
        skeleton: SkeletonDefinition for edges and body_parts
        frame: Frame index to plot (if keypoints is 3D)
        figsize: Figure size
        title: Plot title
        joint_size: Scatter point size for joints
        limb_width: Line width for limbs
        joint_colors: Per-joint colors (auto-derived from skeleton if None)
        limb_colors: Per-edge colors (auto-derived from skeleton if None)
        show_labels: Overlay joint name abbreviations
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

    # Multi-person detection
    num_persons = 1
    jpn = K  # joints per person
    if skeleton is not None and skeleton.num_persons > 1:
        num_persons = skeleton.num_persons
        jpn = skeleton.num_joints

    if num_persons > 1 and K >= jpn * num_persons:
        # Multi-person: render each person with distinct color scheme
        person_base = get_person_colors(num_persons)

        for p in range(num_persons):
            start = p * jpn
            end = start + jpn
            pkp = kp[start:end]
            base_color = person_base[p]

            # Derive per-joint colors tinted by person color
            if skeleton is not None and skeleton.body_parts:
                pj_colors = get_joint_colors(skeleton)
            else:
                pj_colors = [base_color] * jpn

            # Tint: blend body-part color toward person base
            blended_jc = []
            for c in pj_colors:
                if c == _FALLBACK_COLOR:
                    blended_jc.append(base_color)
                else:
                    blended_jc.append(c)
            # For person 1+, darken/lighten to distinguish
            if p > 0:
                blended_jc = [_adjust_color(c, 0.7 + 0.15 * p) for c in blended_jc]

            if is_3d:
                ax.scatter(pkp[:, 0], pkp[:, 1], pkp[:, 2],
                           c=blended_jc, s=joint_size, zorder=5,
                           edgecolors="white", linewidths=0.5)
            else:
                ax.scatter(pkp[:, 0], pkp[:, 1],
                           c=blended_jc, s=joint_size, zorder=5,
                           edgecolors="white", linewidths=0.5)

            # Edges per person (offset indices)
            if skeleton is not None:
                pl_colors = get_limb_colors(skeleton)
                if p > 0:
                    pl_colors = [_adjust_color(c, 0.7 + 0.15 * p) for c in pl_colors]
                for ei, (i, j) in enumerate(skeleton.edges):
                    if i < jpn and j < jpn:
                        lc = pl_colors[ei] if ei < len(pl_colors) else base_color
                        if is_3d:
                            ax.plot(
                                [pkp[i, 0], pkp[j, 0]],
                                [pkp[i, 1], pkp[j, 1]],
                                [pkp[i, 2], pkp[j, 2]],
                                linewidth=limb_width, color=lc, alpha=0.8,
                            )
                        else:
                            ax.plot(
                                [pkp[i, 0], pkp[j, 0]],
                                [pkp[i, 1], pkp[j, 1]],
                                linewidth=limb_width, color=lc, alpha=0.8,
                            )

            # Labels
            if show_labels and skeleton is not None:
                labels = get_joint_labels(skeleton)
                for li in range(min(jpn, len(labels))):
                    if is_3d:
                        ax.text(pkp[li, 0], pkp[li, 1], pkp[li, 2],
                                f" {labels[li]}", fontsize=6, alpha=0.7)
                    else:
                        ax.text(pkp[li, 0], pkp[li, 1],
                                f" {labels[li]}", fontsize=6, alpha=0.7)
    else:
        # Single-person rendering
        jc = joint_colors
        if jc is None and skeleton is not None:
            jc = get_joint_colors(skeleton)

        if is_3d:
            if jc is not None:
                ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2],
                           c=jc[:K], s=joint_size, zorder=5,
                           edgecolors="white", linewidths=0.5)
            else:
                ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2],
                           s=joint_size, zorder=5)
        else:
            if jc is not None:
                ax.scatter(kp[:, 0], kp[:, 1],
                           c=jc[:K], s=joint_size, zorder=5,
                           edgecolors="white", linewidths=0.5)
            else:
                ax.scatter(kp[:, 0], kp[:, 1],
                           s=joint_size, zorder=5)

        # Edges
        if skeleton is not None:
            lc = limb_colors
            if lc is None:
                lc = get_limb_colors(skeleton)

            for ei, (i, j) in enumerate(skeleton.edges):
                if i < K and j < K:
                    ec = lc[ei] if lc and ei < len(lc) else "gray"
                    if is_3d:
                        ax.plot(
                            [kp[i, 0], kp[j, 0]],
                            [kp[i, 1], kp[j, 1]],
                            [kp[i, 2], kp[j, 2]],
                            linewidth=limb_width, color=ec, alpha=0.8,
                        )
                    else:
                        ax.plot(
                            [kp[i, 0], kp[j, 0]],
                            [kp[i, 1], kp[j, 1]],
                            linewidth=limb_width, color=ec, alpha=0.8,
                        )

        # Labels
        if show_labels and skeleton is not None:
            labels = get_joint_labels(skeleton)
            for li in range(min(K, len(labels))):
                if is_3d:
                    ax.text(kp[li, 0], kp[li, 1], kp[li, 2],
                            f" {labels[li]}", fontsize=6, alpha=0.7)
                else:
                    ax.text(kp[li, 0], kp[li, 1],
                            f" {labels[li]}", fontsize=6, alpha=0.7)

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
    joint_colors: list[str] | None = None,
    limb_colors: list[str] | None = None,
    show_labels: bool = False,
    save_path: str | None = None,
):
    """Create an animation of skeleton keypoints over time with colors.

    Args:
        keypoints: (T, K, D) keypoint sequence
        skeleton: SkeletonDefinition for edges
        fps: Frames per second for animation
        figsize: Figure size
        title: Animation title
        joint_colors: Per-joint colors (auto-derived if None)
        limb_colors: Per-edge colors (auto-derived if None)
        show_labels: Overlay joint name abbreviations
        save_path: If set, save as .mp4 or .gif

    Returns:
        matplotlib.animation.FuncAnimation object
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    T, K, D = keypoints.shape
    is_3d = D >= 3

    # Pre-compute colors
    num_persons = 1
    jpn = K
    if skeleton is not None and skeleton.num_persons > 1:
        num_persons = skeleton.num_persons
        jpn = skeleton.num_joints

    multi = num_persons > 1 and K >= jpn * num_persons

    # Pre-compute color arrays for efficiency
    if multi:
        person_base = get_person_colors(num_persons)
        person_jc = []
        person_lc = []
        for p in range(num_persons):
            if skeleton is not None and skeleton.body_parts:
                pjc = get_joint_colors(skeleton)
            else:
                pjc = [person_base[p]] * jpn
            blended = [person_base[p] if c == _FALLBACK_COLOR else c for c in pjc]
            if p > 0:
                blended = [_adjust_color(c, 0.7 + 0.15 * p) for c in blended]
            person_jc.append(blended)

            if skeleton is not None:
                plc = get_limb_colors(skeleton)
                if p > 0:
                    plc = [_adjust_color(c, 0.7 + 0.15 * p) for c in plc]
                person_lc.append(plc)
            else:
                person_lc.append([])
    else:
        jc = joint_colors
        if jc is None and skeleton is not None:
            jc = get_joint_colors(skeleton)
        lc = limb_colors
        if lc is None and skeleton is not None:
            lc = get_limb_colors(skeleton)
        labels = get_joint_labels(skeleton) if show_labels and skeleton else None

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

        if multi:
            for p in range(num_persons):
                start = p * jpn
                end = start + jpn
                pkp = kp[start:end]

                if is_3d:
                    ax.scatter(pkp[:, 0], pkp[:, 1], pkp[:, 2],
                               c=person_jc[p], s=30, zorder=5,
                               edgecolors="white", linewidths=0.3)
                else:
                    ax.scatter(pkp[:, 0], pkp[:, 1],
                               c=person_jc[p], s=30, zorder=5,
                               edgecolors="white", linewidths=0.3)

                if skeleton is not None:
                    plc = person_lc[p]
                    for ei, (i, j) in enumerate(skeleton.edges):
                        if i < jpn and j < jpn:
                            ec = plc[ei] if plc and ei < len(plc) else person_base[p]
                            if is_3d:
                                ax.plot([pkp[i, 0], pkp[j, 0]],
                                        [pkp[i, 1], pkp[j, 1]],
                                        [pkp[i, 2], pkp[j, 2]],
                                        linewidth=1.5, color=ec, alpha=0.8)
                            else:
                                ax.plot([pkp[i, 0], pkp[j, 0]],
                                        [pkp[i, 1], pkp[j, 1]],
                                        linewidth=1.5, color=ec, alpha=0.8)
        else:
            if is_3d:
                if jc is not None:
                    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2],
                               c=jc[:K], s=30, zorder=5,
                               edgecolors="white", linewidths=0.3)
                else:
                    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], s=30, zorder=5)
            else:
                if jc is not None:
                    ax.scatter(kp[:, 0], kp[:, 1],
                               c=jc[:K], s=30, zorder=5,
                               edgecolors="white", linewidths=0.3)
                else:
                    ax.scatter(kp[:, 0], kp[:, 1], s=30, zorder=5)

            if skeleton is not None:
                for ei, (i, j) in enumerate(skeleton.edges):
                    if i < K and j < K:
                        ec = lc[ei] if lc and ei < len(lc) else "gray"
                        if is_3d:
                            ax.plot([kp[i, 0], kp[j, 0]],
                                    [kp[i, 1], kp[j, 1]],
                                    [kp[i, 2], kp[j, 2]],
                                    linewidth=1.5, color=ec, alpha=0.8)
                        else:
                            ax.plot([kp[i, 0], kp[j, 0]],
                                    [kp[i, 1], kp[j, 1]],
                                    linewidth=1.5, color=ec, alpha=0.8)

            if show_labels and labels:
                for li in range(min(K, len(labels))):
                    if is_3d:
                        ax.text(kp[li, 0], kp[li, 1], kp[li, 2],
                                f" {labels[li]}", fontsize=5, alpha=0.6)
                    else:
                        ax.text(kp[li, 0], kp[li, 1],
                                f" {labels[li]}", fontsize=5, alpha=0.6)

        if is_3d:
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
        else:
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])

        ax.set_title(f"{title} — Frame {frame}/{T}")
        ax.set_aspect("equal")

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)

    return anim


def plot_skeleton_comparison(
    keypoints_list: list[np.ndarray],
    titles: list[str],
    skeleton=None,
    frame: int = 0,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
):
    """Side-by-side skeleton comparison (e.g. raw vs preprocessed).

    Args:
        keypoints_list: List of (T, K, D) or (K, D) arrays
        titles: Title for each subplot
        skeleton: SkeletonDefinition for edges
        frame: Frame index to plot
        figsize: Figure size (auto-computed if None)
        save_path: If set, save figure

    Returns:
        (fig, axes) tuple
    """
    import matplotlib.pyplot as plt

    n = len(keypoints_list)
    if figsize is None:
        figsize = (5 * n, 5)

    # Detect 3D
    sample = keypoints_list[0]
    if sample.ndim == 3:
        D = sample.shape[2]
    else:
        D = sample.shape[1]
    is_3d = D >= 3

    if is_3d:
        fig = plt.figure(figsize=figsize)
        axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]
    else:
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

    for i, (kp, t) in enumerate(zip(keypoints_list, titles)):
        plot_skeleton(
            kp, skeleton=skeleton, frame=frame,
            title=t, ax=axes[i], show_labels=(i == 0),
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes
