"""Source HTML for one dataset x method grid (``outputs/behavior_analysis_workbench/<out>/``).

Plain content only (h1, header-meta, tldr, sections with embedded figures); the vault's
``build_page.py`` supplies the theme and navigation. Rebuild from the batch folder; do not
hand-edit the output. Usage:
``python -m behavior_lab.visualization.grid_report <batch_dir> [gallery_href]`` writes
``batch_report.html`` and ``batch_gallery.html`` (per-cluster GIFs) into the batch folder.

Colors come from the page theme's CSS variables (``--c1..--c5`` per method, ``--a1`` for the
highlight, ``--warn`` for flags); this module defines no palette of its own.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from .agreement import label_agreement, plot_label_agreement, stretch_labels
from .cluster_map import plot_subtle_cluster_map

# Slices that are the same frames under different pose post-processing: labels are comparable
# frame by frame across the slices of one group.
SAME_FRAMES_PREFIXES = ("avatar_",)
COLS = ["dataset", "method", "n_frames", "n_clusters", "median_bout_sec", "noise_frac",
        "n_repeats", "repeat_ari_mean", "repeat_ari_min", "elapsed_sec"]
# Flags (own loose rule, no literature threshold): a row that trips one is described, not ranked.
MIN_CLUSTERS, MAX_NOISE = 3, 0.3
GALLERY_CLUSTERS, GALLERY_MAX_SEC, GALLERY_PAD_SEC = 6, 2.0, 0.5
GALLERY_GIF_FPS = 8  # frames are subsampled to about this rate: 3 slices come to about 9 MB
SKELETONS = {"subtle_": "subtle_mouse", "shank3ko_": "shank3ko"}


def _flags(r: dict, slice_sec: float) -> list[str]:
    """Reasons why a cell's segmentation is degenerate; empty list = eligible for the highlight."""
    out = []
    if (r.get("n_clusters") or 0) < MIN_CLUSTERS:
        out.append(f"fewer than {MIN_CLUSTERS} clusters")
    step = slice_sec / r["n_frames"]  # seconds per label
    if r.get("median_bout_sec") is not None and r["median_bout_sec"] <= step * 1.01:
        out.append("median bout = 1 label step")
    if (r.get("noise_frac") or 0) > MAX_NOISE:
        out.append(f"noise > {MAX_NOISE:.0%}")
    return out


def _method_color(methods: list[str]) -> dict[str, str]:
    return {m: f"var(--c{1 + i % 5})" for i, m in enumerate(methods)}


def _fmt(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.2f}" if isinstance(v, float) else html.escape(str(v))


def _table(header: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _img(fig) -> str:
    import matplotlib.pyplot as plt

    from .html_report import fig_to_base64

    uri = fig_to_base64(fig, dpi=90)  # already a full data URI
    assert uri.startswith("data:image/png;base64,")
    tag = f'<img src="{uri}" alt="">'
    plt.close(fig)
    return tag


def _stem(method: str) -> str:  # same rule as the batch script's label file names
    return method.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _labels(batch_dir: Path, dataset: str) -> dict[str, np.ndarray]:
    """Label files of cells that have a finished row (a running cell leaves a file but no row)."""
    rows = json.loads((batch_dir / "batch_results.json").read_text())
    done = {_stem(r["method"]) for r in rows if r["dataset"] == dataset and r["status"] == "ok"}
    files = sorted((batch_dir / "arrays" / dataset).glob("*_labels.npy"))
    return {p.stem.removesuffix("_labels"): np.load(p) for p in files
            if p.stem.removesuffix("_labels") in done}


def _pairs(agr: dict) -> list[list[object]]:
    n = agr["names"]
    return [[n[i], n[j], float(agr["ari"][i, j]), float(agr["null_ari"][i, j]),
             float(agr["ami"][i, j]), float(agr["homogeneity"][i, j]),
             float(agr["homogeneity"][j, i])]
            for i in range(len(n)) for j in range(i + 1, len(n))]


PAIR_HEADER = ["A", "B", "ARI", "ARI, shifted null", "AMI", "B inside A", "A inside B"]


def _subtle_map(batch_dir: Path, dataset: str):
    """``(tag, arrays)`` of one SUBTLE run with matching embedding and cluster levels, or None.

    Prefers the first seed of the table's runs; falls back to the extra map-only run."""
    files = sorted((batch_dir / "arrays" / dataset).glob("subtle_map_seed*.npz")) \
        or sorted((batch_dir / "arrays" / dataset).glob("subtle_map_extra.npz"))
    if not files:
        return None
    return files[0].stem.removeprefix("subtle_map_"), dict(np.load(files[0]))


def render_grid_report(batch_dir: str | Path, out_html: str | Path | None = None, *,
                       gallery_href: str = "batch_gallery.html") -> Path:
    import matplotlib

    matplotlib.use("Agg")

    batch_dir = Path(batch_dir)
    rows = json.loads((batch_dir / "batch_results.json").read_text())
    slices = json.loads((batch_dir / "dataset_slices.json").read_text())
    datasets = [s["name"] for s in slices if any(r["dataset"] == s["name"] for r in rows)]
    fps = {s["name"]: s["fps"] for s in slices}
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    methods = sorted({r["method"] for r in rows})

    parts: list[str] = []
    title = f"Behavior discovery grid: {batch_dir.name}"
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f'<div class="header-meta">{date.today().isoformat()} · {len(datasets)} dataset '
                 f"slices × {len(methods)} methods · source: <code>{html.escape(str(batch_dir))}"
                 "</code> · rebuild with <code>python -m behavior_lab.visualization.grid_report"
                 "</code></div>")

    rep = [r for r in ok if r.get("repeat_ari_mean") is not None]
    by_method = {m: [r["repeat_ari_mean"] for r in rep if r["method"] == m] for m in methods}
    stab = "; ".join(f"{m} {min(v):.2f} to {max(v):.2f}" for m, v in by_method.items() if v)
    tldr = [f"{len(ok)} of {len(rows)} cells ran; {len(failed)} failed or are not applicable "
            "(listed under Failures).",
            "Unsupervised labels have no ground truth here: every number describes agreement or "
            "stability, not accuracy."]
    if stab:
        tldr.insert(1, f"Repeat-run ARI (same input, different seed), range over slices: {stab}.")
    parts.append('<div class="tldr"><ul>' + "".join(f"<li>{html.escape(x)}</li>" for x in tldr)
                 + "</ul></div>")

    parts.append("<h2>Run setup</h2><p>Each slice is one recording or one pose variant; missing "
                 "keypoints are interpolated over time before any method runs.</p>")
    parts.append(_table(["slice", "frames, keypoints, dims", "fps", "missing fraction",
                         "longest gap (frames)", "units", "source"],
                        [[s["name"], "×".join(map(str, s["shape"])), s["fps"],
                          s["notes"].get("nan_frac"), s["notes"].get("max_gap_frames"),
                          s["notes"].get("units", "not stated in file"),
                          Path(str(s["notes"].get("source", ""))).name]
                         for s in slices if s["name"] in datasets]))

    color = _method_color(methods)
    slice_sec = {s["name"]: s["shape"][0] / s["fps"] for s in slices}
    flags = {(r["dataset"], r["method"]): _flags(r, slice_sec[r["dataset"]]) for r in ok}
    top = {}  # per slice: the unflagged method with the highest repeat ARI
    for ds in datasets:
        cand = [r for r in ok if r["dataset"] == ds and r.get("repeat_ari_mean") is not None
                and not flags[(ds, r["method"])]]
        if cand:
            top[ds] = max(cand, key=lambda r: r["repeat_ari_mean"])["method"]
    hi = "background:color-mix(in srgb, var(--a1) 28%, transparent);font-weight:600"

    parts.append("<h2>Result grid</h2><p>No column measures accuracy: there is no ground truth. "
                 "The highlight marks stability only.</p><ul>"
                 "<li><b>Colored bar</b>: one fixed color per method, the same on every page."
                 "</li><li><span style=\"" + hi + "\">Green</span>: the most repeatable method "
                 "on that slice, i.e. the highest repeat ARI (pairwise ARI between 3 runs that "
                 "differ only in seed) among unflagged rows. It is not the best method: a coarse "
                 "or trivial segmentation is easy to repeat.</li>"
                 "<li><span style=\"color:var(--warn)\">Flag</span>: degenerate segmentation "
                 f"(fewer than {MIN_CLUSTERS} clusters, median bout of one label step, or noise "
                 f"above {MAX_NOISE:.0%}); flagged rows are not ranked.</li>"
                 "<li>SUBTLE has no seed, so its repeats include all of its run-to-run "
                 "variation; the seeded methods only vary their initialization.</li></ul>")

    parts.append("<h3>Repeat ARI by slice and method</h3><p>Cell = repeat ARI (k = clusters). "
                 "Shade follows the value.</p>")
    cell = {(r["dataset"], r["method"]): r for r in ok}
    head = "".join(f'<th style="border-bottom:3px solid {color[m]}">{html.escape(m)}</th>'
                   for m in methods)
    body = []
    for ds in datasets:
        tds = []
        for m in methods:
            r = cell.get((ds, m))
            if r is None or r.get("repeat_ari_mean") is None:
                tds.append("<td></td>")
                continue
            v = r["repeat_ari_mean"]
            style = f"background:color-mix(in srgb, var(--a1) {max(0, v) * 45:.0f}%, transparent)"
            if flags[(ds, m)]:
                style = "color:var(--warn)"
            mark = " · most repeatable" if top.get(ds) == m else ""
            mark += " · flag" if flags[(ds, m)] else ""
            bold = ";font-weight:600" if top.get(ds) == m else ""
            tds.append(f'<td style="{style}{bold}">{v:.2f} (k={r["n_clusters"]}){mark}</td>')
        body.append(f"<tr><td>{html.escape(ds)}</td>{''.join(tds)}</tr>")
    parts.append(f"<table><tr><th>slice</th>{head}</tr>{''.join(body)}</table>")

    parts.append("<h3>All rows</h3><p>One row per slice and method, with the flag reason.</p>")
    body = []
    for r in sorted(ok, key=lambda r: (r["dataset"], r["method"])):
        key = (r["dataset"], r["method"])
        tds = []
        for c in COLS:
            style = ""
            if c == "method":
                style = f"border-left:5px solid {color[r['method']]}"
            elif c == "repeat_ari_mean" and top.get(r["dataset"]) == r["method"]:
                style = hi
            tds.append(f'<td style="{style}">{_fmt(r.get(c))}</td>')
        reason = "; ".join(flags[key])
        tds.append(f'<td style="color:var(--warn)">{html.escape(reason)}</td>')
        body.append(f"<tr>{''.join(tds)}</tr>")
    head = "".join(f"<th>{html.escape(c)}</th>" for c in COLS + ["flag"])
    parts.append(f"<table><tr>{head}</tr>{''.join(body)}</table>")

    parts.append("<h2>Method agreement</h2><p>Per slice: each method's label sequence, then ARI "
                 "and AMI between methods on the same frames. The shifted null is the ARI after "
                 "sliding one sequence in time. “B inside A” near 1 with a low reverse value means "
                 "B is a finer split of A rather than a different segmentation.</p>")
    for ds in datasets:
        seqs = _labels(batch_dir, ds)
        if len(seqs) < 2:
            continue
        fig, agr = plot_label_agreement(seqs, ds, fps=fps.get(ds))
        parts.append(f"<h3>{html.escape(ds)}</h3>{_img(fig)}{_table(PAIR_HEADER, _pairs(agr))}")

    for prefix in SAME_FRAMES_PREFIXES:
        group = [d for d in datasets if d.startswith(prefix)]
        if len(group) < 2:
            continue
        parts.append(f"<h2>Slice agreement: {html.escape(prefix)}*</h2><p>The slices are the "
                     "same frames under different pose post-processing, so one method's labels "
                     "can be compared across them. Read each table against that method's repeat "
                     "ARI in the result grid.</p>")
        for m in sorted({k for d in group for k in _labels(batch_dir, d)}):
            seqs = {d: _labels(batch_dir, d)[m] for d in group if m in _labels(batch_dir, d)}
            if len(seqs) > 1 and len({len(v) for v in seqs.values()}) == 1:
                parts.append(f"<h3>{html.escape(m)}</h3>"
                             + _table(PAIR_HEADER, _pairs(label_agreement(seqs))))

    maps = {ds: _subtle_map(batch_dir, ds) for ds in datasets}
    if any(maps.values()):
        parts.append("<h2>SUBTLE cluster map</h2><p>The 2D UMAP embedding of one SUBTLE run, "
                     "colored by subcluster (left) and by supercluster (right), with cluster "
                     "ids at the centroids. Arrows are transitions with probability of at least "
                     "0.15, width by probability. Positions are UMAP coordinates: arrow length "
                     "and centroid distance carry no meaning, and a centroid can fall outside a "
                     "non-convex cluster.</p>")
        for ds in datasets:
            if not maps[ds]:
                continue
            tag, m = maps[ds]
            fig = plot_subtle_cluster_map(m["embedding"], m["subclusters"], m["superclusters"], ds)
            note = f"Run: {tag}."
            if tag == "extra":
                reps = sorted((batch_dir / "arrays" / ds / "repeats").glob("subtle_seed*.npy"))
                aris = [label_agreement({"map": m["labels"], "rep": np.load(p)})["ari"][0, 1]
                        for p in reps]
                note = ("Run: one extra run, not one of the runs in the tables (SUBTLE has no "
                        "seed). ARI of its labels to those runs: "
                        + ", ".join(f"{a:.2f}" for a in aris) + ".")
            parts.append(f"<h3>{html.escape(ds)}</h3><p>{html.escape(note)}</p>{_img(fig)}")

    parts.append("<h2>Cluster gallery</h2><p>Skeleton animations of each method's clusters on "
                 f'representative slices are on a separate page: <a href="{html.escape(gallery_href)}">'
                 "cluster gallery</a>.</p>")

    if failed:
        parts.append("<h2>Failures</h2><p>Cells that did not produce labels, kept so that "
                     "nothing is silently missing.</p>")
        parts.append(_table(["dataset", "method", "error"],
                            [[r["dataset"], r["method"], (r.get("error") or "")[-300:]]
                             for r in failed]))

    parts.append("<h2>Limits</h2><ul>"
                 "<li>ARI and AMI between methods with very different cluster counts mostly "
                 "reflect granularity; use the two “inside” columns for that.</li>"
                 "<li>keypoint-MoSeq kappa is not tuned to a target syllable duration; compare "
                 "its median bout with the other rows before reading its agreement.</li>"
                 "<li>SUBTLE has no seed: its repeats differ by design.</li>"
                 "<li>Slices from different recordings share no frames, so no ARI is computed "
                 "between recordings.</li></ul>")

    out = Path(out_html) if out_html else batch_dir / "batch_report.html"
    out.write_text("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>"
                   f"{html.escape(title)}</title></head>\n<body>\n" + "\n".join(parts)
                   + "\n</body></html>\n", encoding="utf-8")
    return out


def _skeleton(name: str, node_names: list[str] | None, n_joints: int):
    from ..core.skeleton import SkeletonDefinition, get_skeleton

    for prefix, key in SKELETONS.items():
        if name.startswith(prefix):
            return get_skeleton(key), "bones from the skeleton registry"
    names = node_names or [f"kp{i}" for i in range(n_joints)]
    return (SkeletonDefinition(name=name, num_joints=n_joints, joint_names=list(names),
                               joint_parents=[-1] * n_joints, edges=[]),
            "points only: this layout has no bone list in behavior-lab")


def _bout_clip(labels: np.ndarray, cid: int, fps: float) -> tuple[int, int, int]:
    """Frame range of the median-length bout of ``cid`` plus padding: ``(start, end, bout_len)``.

    One real bout, never several stitched together; the median avoids showing the longest
    (often a resting or tracking-loss tail)."""
    edges = np.flatnonzero(np.diff(np.r_[0, (labels == cid).astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    order = np.argsort(ends - starts)
    k = order[len(order) // 2]
    pad = int(GALLERY_PAD_SEC * fps)
    length = min(int(ends[k] - starts[k]), int(GALLERY_MAX_SEC * fps))
    return (max(0, int(starts[k]) - pad), min(len(labels), int(starts[k]) + length + pad),
            int(ends[k] - starts[k]))


def render_gallery(batch_dir: str | Path, out_html: str | Path | None = None) -> Path:
    """Per-method, per-cluster skeleton GIFs on the representative slices (those with a map)."""
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .html_report import image_to_base64
    from .skeleton import animate_skeleton

    batch_dir = Path(batch_dir)
    rows = json.loads((batch_dir / "batch_results.json").read_text())
    slices = {s["name"]: s for s in json.loads((batch_dir / "dataset_slices.json").read_text())}
    methods = sorted({r["method"] for r in rows})
    color = {_stem(m): c for m, c in _method_color(methods).items()}
    chosen = [d for d in slices if (batch_dir / "arrays" / d / "keypoints.npy").exists()
              and _subtle_map(batch_dir, d)]

    title = f"Cluster gallery: {batch_dir.name}"
    parts = [f"<h1>{html.escape(title)}</h1>",
             f'<div class="header-meta">{date.today().isoformat()} · slices: '
             f'{html.escape(", ".join(chosen))} · source: <code>{html.escape(str(batch_dir))}'
             "</code></div>",
             '<div class="tldr"><ul><li>Each animation is one real bout of that cluster: the '
             f"bout of median length, at most {GALLERY_MAX_SEC:g} s, with {GALLERY_PAD_SEC:g} s of "
             f"context before and after, played at about {GALLERY_GIF_FPS} fps. Nothing is "
             "stitched.</li>"
             f"<li>Per method the {GALLERY_CLUSTERS} clusters with the most frames are shown; "
             "the heading says how many clusters exist and what share of frames is covered.</li>"
             "<li>Cluster ids are per method: cluster 3 of one method is unrelated to cluster 3 "
             "of another. The colored bar is the method color used in the result grid.</li>"
             "</ul></div>"]
    with tempfile.TemporaryDirectory() as tmp:
        for ds in chosen:
            kp = np.load(batch_dir / "arrays" / ds / "keypoints.npy")
            fps = slices[ds]["fps"]
            skel, skel_note = _skeleton(ds, slices[ds]["notes"].get("node_names"), kp.shape[1])
            parts.append(f"<h2>{html.escape(ds)}</h2><p>{kp.shape[0]} frames at {fps:g} fps; "
                         f"{html.escape(skel_note)}.</p>")
            for m, seq in _labels(batch_dir, ds).items():
                lab = stretch_labels(seq, len(kp))
                ids, counts = np.unique(lab[lab >= 0], return_counts=True)
                keep = ids[np.argsort(-counts)][:GALLERY_CLUSTERS]
                share = counts[np.argsort(-counts)][:GALLERY_CLUSTERS].sum() / len(lab)
                cards = []
                for cid in keep:
                    s, e, bout = _bout_clip(lab, int(cid), fps)
                    gif = Path(tmp) / f"{ds}_{m}_{cid}.gif"
                    hop = max(1, round(fps / GALLERY_GIF_FPS))
                    animate_skeleton(kp[s:e:hop], skeleton=skel, fps=fps / hop, figsize=(3, 3),
                                     title=f"cluster {cid}", save_path=str(gif))
                    plt.close("all")
                    occ = (lab == cid).mean()
                    cards.append(
                        '<figure style="display:inline-block;margin:4px;text-align:center">'
                        f'<img loading="lazy" src="{image_to_base64(gif)}" style="max-width:190px">'
                        f"<figcaption>cluster {cid} · {occ:.0%} of frames<br>bout {bout / fps:.1f} s"
                        f" · frames {s}–{e}</figcaption></figure>")
                parts.append(f'<h3 style="border-left:6px solid {color[m]};padding-left:8px">'
                             f"{html.escape(m)}</h3><p>{len(keep)} of {len(ids)} clusters, "
                             f"{share:.0%} of frames.</p><div>{''.join(cards)}</div>")
    out = Path(out_html) if out_html else batch_dir / "batch_gallery.html"
    out.write_text("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>"
                   f"{html.escape(title)}</title></head>\n<body>\n" + "\n".join(parts)
                   + "\n</body></html>\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    href = sys.argv[2] if len(sys.argv) > 2 else "batch_gallery.html"
    print(render_grid_report(sys.argv[1], gallery_href=href))
    print(render_gallery(sys.argv[1]))
