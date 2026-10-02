"""Per-slice blocks of the grid report: agreement, cluster sizes, shared maps, matching."""
from __future__ import annotations

import html
from itertools import combinations
from pathlib import Path

import numpy as np

from ._grid_common import (
    PAIR_HEADER,
    RawHtml,
    _details,
    _img,
    _labels,
    _pairs,
    _skeleton,
    _subtle_map,
    _table,
)
from .agreement import label_agreement, match_clusters, plot_label_agreement
from .cluster_map import (
    plot_cluster_sizes,
    plot_method_maps,
    plot_subtle_cluster_map,
    pose_embedding,
)
from .dynamics import COMMON_HZ, plot_dynamics
from .keypoint_schema import joint_color, plot_keypoint_schema

MATCH_ROWS, MATCH_P = 12, 0.05
SHORT_FRAMES = 2000  # below this, matching and bout statistics are reported as low confidence


def schema_block(name: str, kp: np.ndarray, node_names: list[str] | None) -> tuple[str, list]:
    """Figure plus the table cells ``[keypoints, dims, bones, names]`` of one keypoint layout."""
    skel, note = _skeleton(name, node_names, kp.shape[1])
    fig = plot_keypoint_schema(kp, skel.joint_names, skel.edges, name)
    chips = " ".join(
        f'<span style="white-space:nowrap"><span style="display:inline-block;width:.8em;'
        f'height:.8em;border-radius:50%;background:{joint_color(i)}"></span> {i} '
        f"{html.escape(n)}</span>" for i, n in enumerate(skel.joint_names))
    return _img(fig), [kp.shape[1], kp.shape[2], f"{len(skel.edges)} ({note})", RawHtml(chips)]


def slice_matches(seqs: dict[str, np.ndarray]) -> dict[tuple[str, str], dict]:
    return {(a, b): match_clusters(seqs[a], seqs[b]) for a, b in combinations(seqs, 2)}


def _supported(matches: dict[tuple[str, str], dict]) -> list[list[object]]:
    """Clusters whose one-to-one partner clears the null in two or more other methods."""
    partners: dict[tuple[str, int], list[tuple[str, int, float, bool]]] = {}
    for (a, b), m in matches.items():
        for d in m["pairs"]:
            if d["p"] < MATCH_P:
                partners.setdefault((a, d["a"]), []).append((b, d["b"], d["jaccard"], d["largest"]))
                partners.setdefault((b, d["b"]), []).append((a, d["a"], d["jaccard"], d["largest"]))
    rows, seen = [], set()
    for node, ps in sorted(partners.items(), key=lambda kv: -np.mean([p[2] for p in kv[1]])):
        if len(ps) < 2 or node in seen:
            continue
        seen.update((p[0], p[1]) for p in ps)  # the same group is listed once
        rows.append([f"{node[0]} {node[1]}", "; ".join(f"{p[0]} {p[1]} ({p[2]:.2f})" for p in ps),
                     len(ps) + 1, float(np.mean([p[2] for p in ps])),
                     "largest cluster of each method" if all(p[3] for p in ps) else ""])
    return rows


def _matching(matches: dict[tuple[str, str], dict]) -> str:
    summary = [[a, b, m["n_a"], m["n_b"], m["mean_matched"], m["null_mean_matched"], m["p_mean"]]
               for (a, b), m in matches.items()]
    allp = sorted(((d, a, b, m["n_frames"]) for (a, b), m in matches.items() for d in m["pairs"]),
                  key=lambda x: -x[0]["jaccard"])[:MATCH_ROWS]
    top = [[f"{a} {d['a']}", f"{b} {d['b']}", d["jaccard"], d["expected"],
            d["jaccard"] / max(d["expected"], 1e-9), d["p"], d["n_both"] / n,
            "largest cluster of both" if d["largest"] else ""] for d, a, b, n in allp]
    sup = _supported(matches)
    out = ("<p>Clusters of two methods are paired one to one by the Jaccard overlap of their "
           "frames (frames in both / frames in either); noise frames are left out. Jaccard grows "
           "with cluster size, so read it against <i>expected</i> (two independent clusters of "
           "those sizes). <i>p</i> comes from 200 circular shifts of one sequence and is "
           "corrected for picking the best of all cluster pairs.</p>"
           + _table(["A", "B", "clusters A", "clusters B", "mean Jaccard of pairs",
                     "same, shifted null", "p"], summary)
           + f"<p>The {len(top)} pairs with the highest Jaccard over all method pairs:</p>"
           + _table(["cluster A", "cluster B", "Jaccard", "expected", "ratio", "p",
                     "share of frames in both", "note"], top))
    if sup:
        out += ("<p>Clusters paired with p &lt; 0.05 in two or more other methods. A group "
                "made of each method's largest cluster is expected from size alone.</p>"
                + _table(["cluster", "partners (Jaccard)", "methods", "mean Jaccard", "note"], sup))
    else:
        out += "<p>No cluster is paired with p &lt; 0.05 in two or more other methods.</p>"
    return out


def plot_grid_overview(rows: list[dict], flags: dict, families: list[str]):
    """Clusters, median bout and repeat ARI of every finished cell: dot = slice, x = method."""
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    metrics = [("n_clusters", "clusters", "log"), ("median_bout_sec", "median bout (s)", "log"),
               ("repeat_ari_mean", "repeat ARI", "linear")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
    for ax, (key, name, scale) in zip(axes, metrics):
        for r in rows:
            if r.get(key) is None:
                continue
            f = families.index(r["dataset"].split("_")[0])
            x = methods.index(r["method"]) + (f - (len(families) - 1) / 2) * 0.22
            flagged = bool(flags[(r["dataset"], r["method"])])
            ax.scatter(x, r[key], s=36, facecolors="none" if flagged else f"C{f}",
                       edgecolors=f"C{f}", linewidths=1.2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_yscale(scale)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for f, fam in enumerate(families):
        axes[0].scatter([], [], color=f"C{f}", label=fam)
    axes[0].legend(fontsize=8, frameon=False)
    return fig


def slice_blocks(batch_dir: Path, ds: str, kp: np.ndarray | None, fps: float) -> str:
    seqs = _labels(batch_dir, ds)
    if len(seqs) < 2:
        return "<p>Fewer than two methods finished on this slice.</p>"
    T = max(len(v) for v in seqs.values())
    out = []
    if T < SHORT_FRAMES:
        out.append(f"<p>{T} frames: too short for stable clusters; cluster sizes and matches "
                   "below are low confidence.</p>")
    fig, agr = plot_label_agreement(seqs, ds, fps=fps)
    out.append(_details("Label sequences and agreement",
                        _img(fig) + _table(PAIR_HEADER, _pairs(agr)), open_=True))
    out.append(_details("Cluster sizes", _img(plot_cluster_sizes(seqs, ds))))
    out.append(_details(
        "Dynamics: occupancy over time, transitions, bout lengths",
        f"<p>Every method is first put on one {COMMON_HZ:g} Hz grid (majority label per bin), "
        "because bout lengths and transition rates otherwise follow the label rate: a "
        "per-frame method can switch faster than one that labels 100 ms bins. One row per "
        "method. Left: share of each 30 s window per cluster, and transitions per minute "
        "(black line). Middle: probability of the next cluster given the current one, "
        "self-transitions left out. Right: bout lengths on a log axis; the dashed line is the "
        "shortest bout the grid allows.</p>" + _img(plot_dynamics(seqs, T, fps, ds), dpi=70)))
    if kp is not None:
        emb = pose_embedding(kp)
        out.append(_details(
            "Cluster maps on shared axes",
            "<p>Every point is one frame, placed by a 2D PCA of posture (pairwise joint "
            f"distances, joint heights; {pose_embedding.explained:.0%} of the variance). The "
            "points are identical in every panel; only the coloring changes, by each method's "
            "labels. The figure answers one question: do a method's clusters differ in "
            "posture along these two axes? It does not rank methods: clusters that overlap "
            "here can differ in dynamics, and B-SOiD has an advantage because its features "
            "include the same joint distances. The 12 largest clusters are colored, the rest "
            "are grey.</p>" + _img(plot_method_maps(emb, seqs, ds), dpi=60)))
        own = {p.stem.removesuffix("_embedding"): np.load(p)
               for p in sorted((batch_dir / "arrays" / ds).glob("*_embedding.npy"))}
        own = {k: v[:, :2] for k, v in own.items() if k in seqs and v.ndim == 2}
        if own:
            out.append(_details(
                "Cluster maps on each method's own embedding",
                "<p>Each method on the 2D embedding it produced itself. Axes differ between "
                "panels. Clean islands here are separation by construction (k-means and B-SOiD "
                "cluster on this very embedding), not evidence of quality. keypoint-MoSeq "
                "stores no embedding.</p>" + _img(plot_method_maps(own, seqs, ds), dpi=60)))
    sm = _subtle_map(batch_dir, ds)
    if sm:
        tag, m = sm
        note = f"Run: {tag}."
        if tag == "extra":
            reps = sorted((batch_dir / "arrays" / ds / "repeats").glob("subtle_seed*.npy"))
            aris = [label_agreement({"map": m["labels"], "rep": np.load(p)})["ari"][0, 1]
                    for p in reps]
            note = ("Run: one extra run, not one of the runs in the tables (SUBTLE has no seed). "
                    "ARI of its labels to those runs: " + ", ".join(f"{a:.2f}" for a in aris) + ".")
        fig = plot_subtle_cluster_map(m["embedding"], m["subclusters"], m["superclusters"], ds)
        out.append(_details("SUBTLE's own map", "<p>SUBTLE's UMAP embedding, colored by "
                            f"subcluster and by supercluster. {html.escape(note)}</p>" + _img(fig)))
    out.append(_details("Cluster correspondence between methods", _matching(slice_matches(seqs))))
    return "".join(out)
