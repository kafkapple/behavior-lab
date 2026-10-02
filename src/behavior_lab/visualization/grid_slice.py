"""Per-slice blocks of the grid report: agreement, cluster sizes, shared maps, matching."""
from __future__ import annotations

import html
from itertools import combinations
from pathlib import Path

import numpy as np

from ._grid_common import (
    PAIR_HEADER,
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
from .keypoint_schema import plot_keypoint_schema

MATCH_ROWS, MATCH_P = 12, 0.05
SHORT_FRAMES = 2000  # below this, matching and bout statistics are reported as low confidence


def schema_block(name: str, kp: np.ndarray, node_names: list[str] | None) -> tuple[str, list]:
    """Figure plus the table cells ``[keypoints, dims, bones, names]`` of one keypoint layout."""
    skel, note = _skeleton(name, node_names, kp.shape[1])
    fig = plot_keypoint_schema(kp, skel.joint_names, skel.edges, name)
    return _img(fig), [kp.shape[1], kp.shape[2], f"{len(skel.edges)} ({note})",
                       ", ".join(f"{i} {n}" for i, n in enumerate(skel.joint_names))]


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
    if kp is not None:
        emb = pose_embedding(kp)
        out.append(_details(
            "Cluster maps on shared axes",
            f"<p>Every panel uses the same axes: a 2D PCA of posture and speed "
            f"({pose_embedding.explained:.0%} of the variance), which no method clusters on. "
            "A method whose clusters separate here differs in posture or speed; clusters that "
            "overlap here can still differ in dynamics.</p>"
            + _img(plot_method_maps(emb, seqs, ds))))
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
