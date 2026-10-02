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
from .agreement import label_agreement, match_clusters, plot_label_agreement, stretch_labels
from .cluster_map import (
    plot_cluster_sizes,
    plot_method_maps,
    plot_subtle_cluster_map,
    pose_embedding,
)
from .correspondence import plot_meta_graph, plot_recording_shares, recording_purity
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


def slice_matches(seqs: dict[str, np.ndarray],
                  lengths: list[int] | None = None) -> dict[tuple[str, str], dict]:
    return {(a, b): match_clusters(seqs[a], seqs[b], lengths=lengths)
            for a, b in combinations(seqs, 2)}


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


# Columns where one row of a slice can lead: label and which end leads. No direction is defined
# for the cluster count. A lead is a descriptive mark (largest or smallest value among the
# unflagged rows of that slice, ties excluded), not a quality ranking: there is no ground truth.
LEAD_METRICS = {"repeat_ari_mean": ("highest repeat ARI", max),
                "median_bout_sec": ("longest median bout", max),
                "noise_frac": ("lowest noise", min)}


def slice_leads(ok: list[dict], flags: dict) -> dict[tuple[str, str], str]:
    """``(slice, metric) -> method`` holding the unique extreme among unflagged rows."""
    out = {}
    for ds in dict.fromkeys(r["dataset"] for r in ok):
        for metric, (_, pick) in LEAD_METRICS.items():
            cand = [r for r in ok if r["dataset"] == ds and r.get(metric) is not None
                    and not flags[(ds, r["method"])]]
            if len(cand) < 2:
                continue
            best = pick(r[metric] for r in cand)
            holders = [r["method"] for r in cand if r[metric] == best]
            if len(holders) == 1:
                out[(ds, metric)] = holders[0]
    return out


def trend_summary(batch_dir: Path, ok: list[dict], leads: dict, slices: list[dict]) -> list[str]:
    """Bullets for the top of the page (HTML): the overall pattern across all cells."""
    per, aris = settings_summary(batch_dir)
    methods = sorted(per)
    n_slices = len({r["dataset"] for r in ok})

    def med(metric):
        vals = {m: [r[metric] for r in ok if r["method"] == m and r.get(metric) is not None]
                for m in methods}
        order = sorted((m for m in methods if vals[m]), key=lambda m: -np.median(vals[m]))
        return ", ".join(f"{html.escape(m)} {np.median(vals[m]):.2f}" for m in order)

    def lead_counts(metric):
        counts = {m: sum(v == m for (_, k), v in leads.items() if k == metric) for m in methods}
        return ", ".join(f"{html.escape(m)} {c}" for m, c in
                         sorted(counts.items(), key=lambda kv: -kv[1]) if c)

    out = [f"<b>Agreement between methods</b>: ARI median {np.median(aris):.2f}, range "
           f"{min(aris):.2f} to {max(aris):.2f} over {len(aris)} method pairs on the same frames "
           "(keypoint-MoSeq left out).",
           "<b>Repeat ARI</b> (same input, other seed), median over slices: "
           f"{med('repeat_ari_mean')}."
           f" Slices led (of {n_slices}): {lead_counts('repeat_ari_mean') or 'none'}.",
           f"<b>Median bout</b> in seconds, median over slices: {med('median_bout_sec')}. "
           f"Slices led: {lead_counts('median_bout_sec') or 'none'}.",
           "<b>Clusters</b>: " + ", ".join(
               f"{html.escape(m)} {per[m]['clusters'][0]}"
               + (f" to {per[m]['clusters'][1]}" if per[m]['clusters'][0] != per[m]['clusters'][1]
                  else "") for m in methods) + "."]
    pooled = [s for s in slices if s["notes"].get("lengths")]
    nmi = [recording_purity(stretch_labels(v, sum(s["notes"]["lengths"])), s["notes"]["lengths"])[0]
           for s in pooled for k, v in _labels(batch_dir, s["name"]).items()]
    if nmi:
        out.append(f"<b>Pooled fits</b> ({len(pooled)} slices): NMI between cluster and recording "
                   f"{min(nmi):.2f} to {max(nmi):.2f} (0 = clusters shared by the animals, "
                   "1 = clusters are the animals).")
    return out


def settings_summary(batch_dir: Path) -> tuple[dict[str, dict], list[float]]:
    """Per method: cluster count range, repeat ARI range, largest noise share over all slices;
    and the between-method ARI of every method pair on every slice (keypoint-MoSeq left out:
    its state bound is not steered by the target cluster count)."""
    import json

    rows = [r for r in json.loads((batch_dir / "batch_results.json").read_text())
            if r["status"] == "ok"]
    slices = json.loads((batch_dir / "dataset_slices.json").read_text())
    per: dict[str, dict] = {}
    for m in sorted({r["method"] for r in rows}):
        xs = [r for r in rows if r["method"] == m]
        rep = [r["repeat_ari_mean"] for r in xs if r.get("repeat_ari_mean") is not None]
        per[m] = {"clusters": (min(r["n_clusters"] for r in xs), max(r["n_clusters"] for r in xs)),
                  "repeat": (min(rep), max(rep)) if rep else None,
                  "noise": max((r.get("noise_frac") or 0) for r in xs)}
    aris: list[float] = []
    for s in slices:
        seqs = {k: v for k, v in _labels(batch_dir, s["name"]).items() if k != "keypoint_moseq"}
        if len(seqs) > 1:
            a = label_agreement(seqs, n_shifts=1, lengths=s["notes"].get("lengths"))["ari"]
            aris += [float(a[i, j]) for i in range(len(a)) for j in range(i + 1, len(a))]
    return per, aris


def baseline_block(batch_dir: Path, baseline_dir: Path, baseline_href: str | None) -> str:
    """Compact comparison of this batch with the batch run under each method's own settings."""
    here, ari_here = settings_summary(batch_dir)
    base, ari_base = settings_summary(baseline_dir)

    def rng(v, fmt="{:g}"):
        return "" if v is None else (fmt.format(v[0]) if v[0] == v[1]
                                     else f"{fmt.format(v[0])} to {fmt.format(v[1])}")

    rows = [[m, rng(base[m]["clusters"]) if m in base else "", rng(here[m]["clusters"]),
             rng(base[m]["repeat"], "{:.2f}") if m in base else "",
             rng(here[m]["repeat"], "{:.2f}"),
             f"{base[m]['noise']:.0%}" if m in base else "", f"{here[m]['noise']:.0%}"]
            for m in here]
    link = (f' The full page under own settings: <a href="{html.escape(baseline_href)}">'
            f"{html.escape(baseline_dir.name)}</a>." if baseline_href else "")
    return ("<h3>Own settings versus matched cluster count</h3><p>The same slices under each "
            "method's own settings and under the common target count. Between-method "
            f"ARI over all slices has median {np.median(ari_base):.2f} (max {max(ari_base):.2f}, "
            f"{len(ari_base)} pairs) under each method's own settings and median "
            f"{np.median(ari_here):.2f} (max {max(ari_here):.2f}, {len(ari_here)} pairs) here. "
            "Columns: own settings, then this page. keypoint-MoSeq is left out of the ARI "
            f"figures because its state bound is not steered.{link}</p>"
            + _table(["method", "clusters, own", "clusters, here", "repeat ARI, own",
                      "repeat ARI, here", "max noise, own", "max noise, here"], rows))


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


def slice_blocks(batch_dir: Path, ds: str, kp: np.ndarray | None, fps: float, *,
                 emb: np.ndarray | None = None, explained: float | None = None,
                 lengths: list[int] | None = None, recordings: list[str] | None = None) -> str:
    seqs = _labels(batch_dir, ds)
    if len(seqs) < 2:
        return "<p>Fewer than two methods finished on this slice.</p>"
    T = max(len(v) for v in seqs.values())
    out = []
    if T < SHORT_FRAMES:
        out.append(f"<p>{T} frames: too short for stable clusters; cluster sizes and matches "
                   "below are low confidence.</p>")
    if lengths:
        full = {k: stretch_labels(v, T) for k, v in seqs.items()}
        out.append(_details(
            "Cluster share per recording",
            "<p>One fit across all recordings, so a cluster id means the same in every "
            "recording. Bars: share of time per cluster in each recording. NMI(cluster, "
            "recording) near 0 means the clusters are shared by the animals; near 1 means "
            "the clusters mostly tell the animals apart (position, body size or session), "
            "not behavior. With one animal per group, a difference between bars cannot be "
            "attributed to the group.</p>"
            + _img(plot_recording_shares(full, lengths, recordings or [], ds)), open_=True))
    fig, agr = plot_label_agreement(seqs, ds, fps=fps, lengths=lengths)
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
        if emb is None:
            emb, explained = pose_embedding(kp), pose_embedding.explained
        out.append(_details(
            "Cluster maps on shared axes",
            "<p>Every point is one frame, placed by a 2D PCA of posture (pairwise joint "
            f"distances, joint heights; {explained:.0%} of the variance), fitted once on all "
            "recordings of this dataset family: the axes are the same linear combination of "
            "posture features on every slice, and PCA has no random component. The "
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
    matches = slice_matches(seqs, lengths)
    out.append(_details(
        "Cluster correspondence between methods",
        "<p>One column per method, one circle per cluster (1% of frames or more; size = share "
        "of frames, number = cluster id). A line joins two clusters that are matched one to "
        "one and clear the shifted null (p &lt; 0.05); its width is the Jaccard overlap. "
        "Clusters joined by lines form a group and share a color: read them as “the same "
        "segment of behavior under different methods”. White circles have no partner. "
        "Vertical position has no meaning. Construction after the cluster-ensemble "
        "meta-graph of Strehl &amp; Ghosh (2002).</p>" + _img(plot_meta_graph(matches, ds))
        + _matching(matches)))
    return "".join(out)
