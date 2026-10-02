"""Temporal structure of label sequences: occupancy over time, transitions, bout lengths.

Bout lengths and transition rates depend on the label rate (a per-frame method can flicker
faster than a method that labels 100 ms bins), so every sequence is first put on one common
grid (``common_rate``, majority label per bin). That grid is the floor of every bout length.
"""
from __future__ import annotations

import numpy as np

from .agreement import stretch_labels
from .cluster_map import OTHER_COLOR, rank_colors, transition_matrix

COMMON_HZ, WINDOW_SEC, TOP = 10.0, 30.0, 8


def common_rate(labels: np.ndarray, n_frames: int, fps: float, hz: float = COMMON_HZ) -> np.ndarray:
    """Labels on a ``hz`` grid: stretched to frames, then the majority label of each bin."""
    lab = stretch_labels(labels, n_frames)
    b = max(1, round(fps / hz))
    lab = lab[: len(lab) // b * b].reshape(-1, b)
    out = np.empty(len(lab), dtype=lab.dtype)
    for k, row in enumerate(lab):
        vals, counts = np.unique(row, return_counts=True)
        out[k] = vals[np.argmax(counts)] if counts.max() > 1 or b == 1 else row[0]
    return out


def bout_lengths(labels: np.ndarray) -> np.ndarray:
    """Run lengths (in label steps) of the non-noise bouts."""
    labels = np.asarray(labels)
    cuts = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, cuts]
    lengths = np.diff(np.r_[starts, len(labels)])
    assert lengths.sum() == len(labels)
    return lengths[labels[starts] >= 0]


def plot_dynamics(seqs: dict[str, np.ndarray], n_frames: int, fps: float, title: str = ""):
    """One row per method: occupancy over time, transition probabilities, bout lengths."""
    import matplotlib.pyplot as plt

    hz = fps / max(1, round(fps / COMMON_HZ))
    fig, axes = plt.subplots(len(seqs), 3, figsize=(15, 2.5 * len(seqs)), squeeze=False,
                             constrained_layout=True, gridspec_kw={"width_ratios": [2.4, 1, 1]})
    for row, (name, raw) in zip(axes, seqs.items()):
        lab = common_rate(raw, n_frames, fps)
        color = rank_colors(lab)
        ids = [c for c in color if c >= 0][:TOP]  # dict order = size rank
        w = max(1, int(WINDOW_SEC * hz))
        n_win = max(1, len(lab) // w)
        win = lab[: n_win * w].reshape(n_win, w)
        share = np.stack([(win == c).mean(axis=1) for c in ids], axis=1)
        t = (np.arange(n_win) + 0.5) * w / hz / 60
        row[0].stackplot(t, share.T, 1 - share.sum(axis=1), colors=[color[c] for c in ids]
                         + [OTHER_COLOR], labels=[str(c) for c in ids] + ["other"])
        rate = (np.diff(win, axis=1) != 0).sum(axis=1) / (w / hz / 60)
        ax2 = row[0].twinx()
        ax2.plot(t, rate, color="black", lw=1)
        ax2.set_ylabel("transitions / min", fontsize=7)
        ax2.tick_params(labelsize=7)
        row[0].set_ylim(0, 1)
        row[0].set_xlim(t[0], t[-1] if n_win > 1 else t[0] + 1e-6)
        row[0].set_ylabel(f"{name}\nshare of window", fontsize=8)
        row[0].legend(fontsize=6, ncol=TOP + 1, loc="upper center", frameon=False)
        tids, P = transition_matrix(lab)
        keep = [int(np.flatnonzero(tids == c)[0]) for c in ids if c in tids]
        im = row[1].imshow(P[np.ix_(keep, keep)], vmin=0, vmax=1, cmap="magma")
        row[1].set_xticks(range(len(keep)))
        row[1].set_xticklabels([tids[k] for k in keep], fontsize=7)
        row[1].set_yticks(range(len(keep)))
        row[1].set_yticklabels([tids[k] for k in keep], fontsize=7)
        fig.colorbar(im, ax=row[1], fraction=0.046)
        sec = bout_lengths(lab) / hz
        bins = np.logspace(np.log10(1 / hz), np.log10(max(sec.max(), 2 / hz)), 30)
        row[2].hist(sec, bins=bins, color="0.35")
        row[2].set_xscale("log")
        row[2].axvline(1 / hz, color="tab:red", lw=1, ls="--")
        row[2].axvline(np.median(sec), color="black", lw=1)
        row[2].set_title(f"{len(sec)} bouts, median {np.median(sec):.2f} s", fontsize=8)
    axes[0][0].set_title(f"{title}: share of each {WINDOW_SEC:g} s window ({TOP} largest "
                         "clusters); black = transitions per minute", fontsize=9)
    axes[0][1].set_title("P(next cluster | cluster), rows = from", fontsize=9)
    axes[-1][0].set_xlabel("time (min)")
    axes[-1][2].set_xlabel(f"bout length (s); dashed = {1 / hz:.1f} s floor")
    return fig


__all__ = ["bout_lengths", "common_rate", "plot_dynamics"]
