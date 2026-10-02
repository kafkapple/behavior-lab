"""Which clusters of different methods correspond: a meta-graph, and shares per recording.

The meta-graph follows the cluster-ensemble construction of Strehl & Ghosh (2002, MCLA): every
cluster of every method is a vertex and vertices of different methods are joined by an edge
weighted with the Jaccard overlap of their frame sets. Here the edges are the one-to-one
matched pairs that clear the shifted null, and the groups are the connected components of
those edges (the original partitions the graph with METIS). Positions carry no meaning: one
column per method, clusters ordered by group. Read the edges and the colors.
"""
from __future__ import annotations

import numpy as np

from .cluster_map import OTHER_COLOR, PALETTE, rank_colors

MIN_SHARE, MATCH_P = 0.01, 0.05


def meta_groups(matches: dict[tuple[str, str], dict]) -> tuple[dict, list]:
    """``(node -> share, groups)``; a node is ``(method, cluster id)``, a group is a list of
    nodes joined by significant matched pairs, largest first (singletons excluded)."""
    share: dict[tuple[str, int], float] = {}
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), m in matches.items():
        for name, ids, sh in ((a, m["ids_a"], m["share_a"]), (b, m["ids_b"], m["share_b"])):
            for c, v in zip(ids, sh):
                share[(name, int(c))] = max(share.get((name, int(c)), 0.0), float(v))
                parent.setdefault((name, int(c)), (name, int(c)))
    for (a, b), m in matches.items():
        for d in m["pairs"]:
            if d["p"] < MATCH_P:
                parent[find((a, d["a"]))] = find((b, d["b"]))
    comp: dict = {}
    for node in parent:
        comp.setdefault(find(node), []).append(node)
    groups = sorted((g for g in comp.values() if len(g) > 1),
                    key=lambda g: -sum(share[n] for n in g))
    return share, groups


def plot_meta_graph(matches: dict[tuple[str, str], dict], title: str = ""):
    import matplotlib.pyplot as plt

    share, groups = meta_groups(matches)
    methods = list(dict.fromkeys(m for pair in matches for m in pair))
    color = {n: (PALETTE[k] if k < len(PALETTE) else OTHER_COLOR)
             for k, g in enumerate(groups) for n in g}
    rank = {n: k for k, g in enumerate(groups) for n in g}
    fig, ax = plt.subplots(figsize=(2.6 * len(methods), 6), constrained_layout=True)
    pos = {}
    for x, m in enumerate(methods):
        nodes = sorted((n for n in share if n[0] == m and share[n] >= MIN_SHARE),
                       key=lambda n: (rank.get(n, len(groups)), -share[n]))
        for y, n in enumerate(nodes):
            pos[n] = (x, -y / max(1, len(nodes) - 1) if len(nodes) > 1 else -0.5)
    for (a, b), mm in matches.items():
        for d in mm["pairs"]:
            p, q = (a, d["a"]), (b, d["b"])
            if d["p"] < MATCH_P and p in pos and q in pos:
                ax.plot(*zip(pos[p], pos[q]), color=color.get(p, OTHER_COLOR), alpha=0.6,
                        lw=0.5 + 6 * d["jaccard"], zorder=1)
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=60 + 1500 * share[n], color=color.get(n, "white"), edgecolors="0.3",
                   zorder=2)
        ax.text(x, y, str(n[1]), ha="center", va="center", fontsize=7, zorder=3)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_title(f"{title}: {len(groups)} groups of corresponding clusters", fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return fig


def recording_purity(labels: np.ndarray, lengths: list[int]) -> tuple[float, float]:
    """``(NMI between cluster and recording, mean share of a cluster from its top recording)``.

    Near 0 and near 1/len(lengths): clusters are shared by the animals. Near 1: the clusters
    are the animals (position, size or session), not behavior."""
    from sklearn.metrics import normalized_mutual_info_score

    labels = np.asarray(labels)
    rec = np.repeat(np.arange(len(lengths)), lengths)
    assert len(rec) == len(labels)
    keep = labels >= 0
    ids = np.unique(labels[keep])
    top = [np.bincount(rec[labels == c], minlength=len(lengths)).max() / (labels == c).sum()
           for c in ids]
    weights = [(labels == c).sum() for c in ids]
    return (float(normalized_mutual_info_score(rec[keep], labels[keep])),
            float(np.average(top, weights=weights)))


def plot_recording_shares(seqs: dict[str, np.ndarray], lengths: list[int], names: list[str],
                          title: str = ""):
    """Per method: share of time per cluster in each recording (same ids in every recording)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(seqs), figsize=(3.3 * len(seqs), 3.2), squeeze=False,
                             constrained_layout=True, sharey=True)
    bounds = np.cumsum([0] + list(lengths))
    for ax, (name, lab) in zip(axes[0], seqs.items()):
        color = rank_colors(lab)
        ids = [c for c in color if c >= 0]
        bottom = np.zeros(len(lengths))
        for c in ids:
            sh = np.array([(lab[bounds[k]:bounds[k + 1]] == c).mean() for k in range(len(lengths))])
            ax.bar(range(len(lengths)), sh, bottom=bottom, color=color[c], width=0.85)
            bottom += sh
        nmi, top = recording_purity(lab, lengths)
        ax.set_title(f"{name}\nNMI(cluster, recording) {nmi:.2f}, top-recording share {top:.2f}",
                     fontsize=8)
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([n.split("_", 1)[-1] for n in names], rotation=45, ha="right",
                           fontsize=7)
    axes[0][0].set_ylabel("share of time")
    fig.suptitle(title, fontsize=10)
    return fig


__all__ = ["meta_groups", "plot_meta_graph", "plot_recording_shares", "recording_purity"]
