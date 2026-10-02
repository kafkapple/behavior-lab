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
    L = len(labels)
    if L == T:
        return labels
    b = round(T / L)
    if b >= 1 and 0 <= T - L * b < 2 * b:
        # whole-frame bins that start at frame 0 (B-SOiD drops a short tail): bin k covers
        # frames [k*b, (k+1)*b); proportional stretching would drift by one bin at the end
        return labels[np.minimum(np.arange(T) // b, L - 1)]
    # frame i belongs to bin floor(i * L / T); linspace over (0, L-1) would shift bin edges
    return labels[np.arange(T) * L // T]


def shift_within(labels: np.ndarray, frac: float, lengths: list[int] | None = None) -> np.ndarray:
    """Circularly shift by ``frac`` of the length; with ``lengths``, inside each recording.

    A pooled sequence must not be rolled as a whole: labels would land on another animal, the
    null overlap would collapse and every comparison would look significant."""
    if not lengths:
        return np.roll(labels, int(frac * len(labels)))
    assert sum(lengths) == len(labels), "lengths do not add up to the sequence"
    parts = np.split(labels, np.cumsum(lengths)[:-1])
    return np.concatenate([np.roll(p, int(frac * len(p))) for p in parts])


def label_agreement(seqs: dict[str, np.ndarray], *, n_shifts: int = 20, seed: int = 0,
                    lengths: list[int] | None = None) -> dict[str, object]:
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
    shifts = rng.uniform(0.1, 0.9, size=n_shifts)  # fraction of the (recording) length
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out["homogeneity"][i, j] = homogeneity_score(lab[i], lab[j])
            if i < j:
                out["ari"][i, j] = out["ari"][j, i] = adjusted_rand_score(lab[i], lab[j])
                out["ami"][i, j] = out["ami"][j, i] = adjusted_mutual_info_score(lab[i], lab[j])
                null[i, j] = null[j, i] = float(np.mean(
                    [adjusted_rand_score(lab[i], shift_within(lab[j], s, lengths))
                     for s in shifts]))
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


def plot_label_agreement(seqs: dict[str, np.ndarray], title: str = "", *, fps: float | None = None,
                         lengths: list[int] | None = None):
    """Ethogram per sequence plus ARI and AMI heatmaps. Returns ``(fig, agreement_dict)``."""
    import matplotlib.pyplot as plt

    agr = label_agreement(seqs, lengths=lengths)
    names, T = agr["names"], agr["n_frames"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 0.5 * len(names) + 3.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.2, 1, 1]})
    # colors are per sequence (size rank): the same color in two rows is not the same behavior
    from matplotlib.colors import to_rgb

    from .cluster_map import rank_colors

    rows = []
    for n in names:
        lab = stretch_labels(seqs[n], T)
        color = {c: to_rgb(v) for c, v in rank_colors(lab).items()}
        rows.append(np.array([color[int(c)] for c in lab]))
    axes[0].imshow(np.stack(rows), aspect="auto", interpolation="nearest",
                   extent=(0, T / fps if fps else T, len(names), 0))
    axes[0].set_yticks(np.arange(len(names)) + 0.5)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("time (s)" if fps else "frame")
    axes[0].set_title(f"{title}: label sequences", fontsize=10)
    _heatmap(axes[1], names, agr["ari"], "ARI")
    _heatmap(axes[2], names, agr["ami"], "AMI")
    return fig, agr


def _jaccard(a: np.ndarray, b: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray):
    """Jaccard matrix and contingency counts over frames where neither label is noise (-1)."""
    keep = (a >= 0) & (b >= 0)
    C = np.zeros((len(ids_a), len(ids_b)))
    np.add.at(C, (np.searchsorted(ids_a, a[keep]), np.searchsorted(ids_b, b[keep])), 1)
    union = C.sum(axis=1, keepdims=True) + C.sum(axis=0, keepdims=True) - C
    return np.divide(C, union, out=np.zeros_like(C), where=union > 0), C


def match_clusters(a: np.ndarray, b: np.ndarray, *, n_shifts: int = 200, seed: int = 0,
                   lengths: list[int] | None = None) -> dict[str, object]:
    """One-to-one correspondence between the clusters of two label sequences.

    Similarity = Jaccard overlap of the frame sets (frames in both / frames in either), noise
    frames (-1) dropped on both sides. The assignment maximizes total Jaccard (Hungarian).
    Jaccard grows with cluster size, so each pair also carries ``expected``: the Jaccard two
    independent clusters of those sizes would have. The null repeats the whole procedure after
    circularly shifting ``b``: ``p`` of a pair = share of shifts whose best Jaccard over all
    cluster pairs reaches that pair's value (corrects for picking the best of many pairs),
    ``p_mean`` = the same for the mean Jaccard of the assignment.
    """
    from scipy.optimize import linear_sum_assignment

    T = max(len(a), len(b))
    a, b = stretch_labels(a, T), stretch_labels(b, T)
    ids_a, ids_b = np.unique(a[a >= 0]), np.unique(b[b >= 0])
    J, C = _jaccard(a, b, ids_a, ids_b)
    rows, cols = linear_sum_assignment(-J)
    n = C.sum()
    rng = np.random.default_rng(seed)
    null_best, null_mean = np.zeros(n_shifts), np.zeros(n_shifts)
    for k, s in enumerate(rng.uniform(0.1, 0.9, size=n_shifts)):
        Jn, _ = _jaccard(a, shift_within(b, s, lengths), ids_a, ids_b)
        r, c = linear_sum_assignment(-Jn)
        null_best[k], null_mean[k] = Jn.max(), Jn[r, c].mean()
    pairs = []
    for i, j in zip(rows, cols):
        if C[i, j] == 0:
            continue
        na, nb = C[i].sum(), C[:, j].sum()
        e = na * nb / n  # frames shared by independent clusters of these sizes
        pairs.append({"a": int(ids_a[i]), "b": int(ids_b[j]), "n_a": int(na), "n_b": int(nb),
                      "n_both": int(C[i, j]), "jaccard": float(J[i, j]),
                      "expected": float(e / (na + nb - e)),
                      "p": float((1 + (null_best >= J[i, j]).sum()) / (1 + n_shifts)),
                      "largest": bool(na == C.sum(axis=1).max() and nb == C.sum(axis=0).max())})
    pairs.sort(key=lambda d: -d["jaccard"])
    mean = float(J[rows, cols].mean())
    assert all(0 <= d["jaccard"] <= 1 and 0 <= d["expected"] <= 1 for d in pairs)
    return {"pairs": pairs, "jaccard": J, "ids_a": ids_a, "ids_b": ids_b,
            "share_a": C.sum(axis=1) / n, "share_b": C.sum(axis=0) / n, "mean_matched": mean,
            "null_mean_matched": float(null_mean.mean()),
            "p_mean": float((1 + (null_mean >= mean).sum()) / (1 + n_shifts)),
            "n_frames": int(n), "n_a": len(ids_a), "n_b": len(ids_b)}


__all__ = ["label_agreement", "match_clusters", "plot_label_agreement", "shift_within",
           "stretch_labels"]
