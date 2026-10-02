"""Bottom section of the grid report: method profiles and how to read each block.

No ground truth exists for these recordings, so no block can name a correct method. The
profile table puts the criteria that the literature uses side by side per method; the
criteria table says which block shows each criterion, how to read it and where the reading
rule comes from. References are listed once at the end (``REFERENCES``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._grid_common import _labels, _stem, _table
from .agreement import label_agreement, stretch_labels
from .correspondence import recording_purity


def method_profiles(batch_dir: Path, ok: list[dict], flags: dict,
                    slices: list[dict]) -> tuple[list[str], list[list[object]]]:
    """``(header, rows)``: one row per method, each column a criterion (medians over slices)."""
    methods = sorted({r["method"] for r in ok})
    stem = {_stem(m): m for m in methods}
    to_others: dict[str, dict[str, list[float]]] = {m: {"ari": [], "ami": []} for m in methods}
    nmi: dict[str, list[float]] = {m: [] for m in methods}
    for s in slices:
        seqs = _labels(batch_dir, s["name"])
        lengths = s["notes"].get("lengths")
        if len(seqs) > 1:
            a = label_agreement(seqs, n_shifts=1, lengths=lengths)
            for i, n in enumerate(a["names"]):
                others = [j for j in range(len(a["names"])) if j != i]
                to_others[stem[n]]["ari"].append(float(np.mean(a["ari"][i, others])))
                to_others[stem[n]]["ami"].append(float(np.mean(a["ami"][i, others])))
        if lengths:
            for n, v in seqs.items():
                nmi[stem[n]].append(recording_purity(stretch_labels(v, sum(lengths)), lengths)[0])

    def med(vals: list[float]) -> object:
        return float(np.median(vals)) if vals else ""

    rows = []
    for m in methods:
        xs = [r for r in ok if r["method"] == m]
        rep = [r["repeat_ari_mean"] for r in xs if r.get("repeat_ari_mean") is not None]
        k = sorted({r["n_clusters"] for r in xs})
        bout = med([r["median_bout_sec"] for r in xs if r.get("median_bout_sec") is not None])
        in_range = isinstance(bout, float) and SYLLABLE_RANGE_S[0] <= bout <= SYLLABLE_RANGE_S[1]
        rows.append([m, f"{k[0]}" if len(k) == 1 else f"{k[0]} to {k[-1]}",
                     med(rep), med(to_others[m]["ari"]), med(to_others[m]["ami"]),
                     bout, "yes" if in_range else "no",
                     med(nmi[m]), f"{max((r.get('noise_frac') or 0) for r in xs):.0%}",
                     f"{sum(bool(flags[(r['dataset'], m)]) for r in xs)} of {len(xs)}"])
    header = ["method", "clusters", "repeat ARI", "ARI to other methods", "AMI to other methods",
              "median bout (s)",
              f"bout within {SYLLABLE_RANGE_S[0]:g} to {SYLLABLE_RANGE_S[1]:g} s [5]",
              "NMI with recording (pooled)", "max noise", "flagged slices"]
    return header, rows


def reading_section(batch_dir: Path, ok: list[dict], flags: dict, slices: list[dict]) -> str:
    header, rows = method_profiles(batch_dir, ok, flags, slices)
    crit = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in CRITERIA)
    refs = "".join(f"<li>{r}</li>" for r in REFERENCES)
    return ("<h2>Reading guide</h2><p>No block names a correct method: there are no behavior "
            "labels for these recordings. Each method has a profile over several criteria; "
            "which profile fits depends on the question asked of the data.</p>"
            "<h3>Method profiles</h3><p>Medians over all slices of this page. Columns follow "
            "the criteria of the next table; none is an accuracy.</p>" + _table(header, rows)
            + "<h3>Criteria and sources</h3><p>For each criterion: the block that shows it, "
            "how to read it, and the source of the reading rule.</p>"
            "<table><tr><th>criterion</th><th>block on this page</th><th>how to read it</th>"
            f"<th>source</th></tr>{crit}</table>"
            f"<h3>References</h3><ol>{refs}</ol>")


# Typical duration of mouse behavioral syllables in depth MoSeq (Wiltschko et al., 2015 [5]):
# "blocks typically lasting 200-900 ms". A reference band, not a pass mark for keypoint data.
SYLLABLE_RANGE_S = (0.2, 0.9)

# (criterion, block on the page, how to read it, source). Sources were checked against the
# original text on 2026-10-02 unless marked; numbers in brackets refer to REFERENCES.
CRITERIA: list[tuple[str, str, str, str]] = [
    ("Repeat stability", "Result grid: repeat ARI",
     "High = the same input gives the same labels again. Needed, but not enough: a trivial "
     "segmentation is also repeatable, and stability-based selection is not guaranteed to "
     "pick a meaningful clustering.", "[2]"),
    ("Agreement between methods", "Label sequences and agreement: ARI, AMI, shifted null",
     "ARI near 0 = chance level. With unbalanced or small clusters read AMI rather than ARI. "
     "Agreement says the methods find the same segmentation, not that it is correct.",
     "[3]; ARI from [4] (text not checked)"),
    ("Cluster correspondence", "Cluster correspondence between methods: Jaccard, graph",
     "Jaccard of 0.5 or less = no stable counterpart, 0.6 to 0.75 = a pattern with doubtful "
     "membership, 0.75 or more = stable. The rule was stated for one cluster under "
     "resampling; here it only orients the reading of pairs between methods. The graph "
     "follows the cluster-ensemble meta-graph.", "[1] (via the fpc documentation); [9]"),
    ("Temporal scale", "Dynamics: bout lengths, transitions per minute",
     "Mouse syllables in depth MoSeq last about 350 ms on average, typically 200 to 900 ms. "
     "Keypoint methods without a dynamics model gave median states of 33 to 100 ms in a "
     "published comparison, against about 400 ms for keypoint-MoSeq. B-SOiD cannot go below "
     "its 100 ms bin.", "[5]; [6]; [7]"),
    ("Sharing across animals", "Cluster share per recording (pooled slices)",
     "NMI between cluster and recording near 0 = the clusters occur in every animal. "
     "Fitting one model on pooled animals is the practice of behavior-map studies. A method "
     "whose input encodes position can share clusters without sharing behavior.",
     "[10]; own measure"),
    ("2D maps", "Cluster maps on shared axes and on own embeddings",
     "A two-dimensional embedding distorts distances; treat it as a diagram. Separation on "
     "a method's own embedding is by construction.", "[8]"),
    ("Agreement with human labels", "not available on this page",
     "Published comparisons score unsupervised labels against supervised labels with "
     "normalized mutual information. These recordings have no behavior labels, so no block "
     "measures accuracy.", "[6]"),
]
REFERENCES: list[str] = [
    "Hennig C (2007). Cluster-wise assessment of cluster stability. Computational Statistics "
    "&amp; Data Analysis 52:258–271. Thresholds as given in the author's fpc::clusterboot "
    "documentation; the paper itself was not read.",
    "von Luxburg U (2010). Clustering stability: an overview. Foundations and Trends in "
    "Machine Learning. arXiv:1007.1075.",
    "Romano S, Vinh NX, Bailey J, Verspoor K (2016). Adjusting for chance clustering "
    "comparison measures. Journal of Machine Learning Research 17(134).",
    "Hubert L, Arabie P (1985). Comparing partitions. Journal of Classification 2:193–218. "
    "Text not checked (paywalled).",
    "Wiltschko AB et al. (2015). Mapping sub-second structure in mouse behavior. Neuron.",
    "Weinreb C et al. (2024). Keypoint-MoSeq: parsing behavior by linking point tracking to "
    "pose dynamics. Nature Methods.",
    "Hsu AI, Yttri EA (2021). B-SOiD, an open-source unsupervised algorithm for "
    "identification and fast prediction of behaviors. Nature Communications 12:5188.",
    "Chari T, Pachter L (2023). The specious art of single-cell genomics. PLoS Computational "
    "Biology 19(8):e1011288.",
    "Strehl A, Ghosh J (2002). Cluster ensembles – a knowledge reuse framework for combining "
    "multiple partitions. Journal of Machine Learning Research 3:583–617.",
    "Berman GJ, Choi DM, Bialek W, Shaevitz JW (2014). Mapping the stereotyped behaviour of "
    "freely moving fruit flies. Journal of the Royal Society Interface.",
]

__all__ = ["CRITERIA", "REFERENCES", "method_profiles", "reading_section"]
