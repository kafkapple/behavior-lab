"""Source HTML for one dataset x method grid (``outputs/behavior_analysis_workbench/<out>/``).

Plain content only (h1, header-meta, tldr, sections with embedded figures); the vault's
``build_page.py`` supplies the theme and navigation. Rebuild from the batch folder; do not
hand-edit the output. Usage:
``python -m behavior_lab.visualization.grid_report <batch_dir> [gallery_href] [player_href]``
writes ``batch_report.html``, ``batch_gallery.html`` (per-cluster GIFs) and
``batch_player.html`` (synchronized playback) into the batch folder.

Page order: keypoint layouts, run setup, the result grid over all cells, then one section per
dataset family with one subsection per slice (blocks in ``grid_slice.py``).

Colors come from the page theme's CSS variables (``--c1..--c5`` per method, ``--a1`` for the
highlight, ``--warn`` for flags); this module defines no palette of its own.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import date
from pathlib import Path

from ._grid_common import (
    DATASET_INFO,
    FAMILIES,
    FAMILY_VARS,
    MAX_NOISE,
    METHOD_HEAD,
    METHOD_INFO,
    MIN_CLUSTERS,
    PAIR_HEADER,
    ROW_COLS,
    ROW_HEAD,
    SAME_FRAMES_PREFIXES,
    SORT_JS,
    RawHtml,
    _family,
    _flags,
    _fmt,
    _img,
    _keypoints,
    _labels,
    _method_color,
    _pairs,
    _table,
    _write,
    family_embeddings,
)
from .agreement import label_agreement
from .grid_slice import plot_grid_overview, schema_block, slice_blocks


def render_grid_report(batch_dir: str | Path, out_html: str | Path | None = None, *,
                       gallery_href: str = "batch_gallery.html",
                       player_href: str = "batch_player.html") -> Path:
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

    families = list(dict.fromkeys(_family(d) for d in datasets))
    by_name = {s["name"]: s for s in slices}
    settings_file = batch_dir / "method_settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    parts.append("<h2>Data and methods</h2><p>What was recorded and what each method does "
                 "with it. Every cell is one unsupervised fit: no pretrained model, no labels."
                 "</p><h3>Datasets</h3>")
    parts.append(_table(["family", "slices", "what it is"],
                        [[FAMILIES.get(f, f), ", ".join(d for d in datasets if _family(d) == f),
                          RawHtml(DATASET_INFO.get(f, ""))] for f in families]))
    parts.append("<h3>Methods and settings</h3><p>Slices named <code>*_pooled</code> are one "
                 "fit across all recordings of the family (recordings stay separate sequences); "
                 "all other slices are fitted alone.</p>")
    parts.append(_table(METHOD_HEAD, [
        [m, *METHOD_INFO.get(m, ("", "", "", "", "")),
         "; ".join(f"{k} = {v}" for k, v in settings.get(m, {}).items())] for m in methods]))
    emb, explained = family_embeddings(batch_dir, [by_name[d] for d in datasets])
    parts.append("<h2>Keypoint layouts</h2><p>One layout per dataset family. Every method takes "
                 "any layout as a (frames, keypoints, dims) array; only the drawing needs bones "
                 "and only keypoint-MoSeq needs to know the nose and the tail base, which it "
                 "finds by joint name. Clusters are never compared across layouts.</p>")
    schema_rows = []
    figs = []
    for fam in families:
        first = next(d for d in datasets if _family(d) == fam)
        kp = _keypoints(batch_dir, first)
        if kp is None:
            continue
        img, row = schema_block(first, kp, by_name[first]["notes"].get("node_names"))
        schema_rows.append([FAMILIES.get(fam, fam), sum(_family(d) == fam for d in datasets),
                            by_name[first]["fps"], *row])
        figs.append(f"<h3>{html.escape(FAMILIES.get(fam, fam))}</h3><p>Slice shown: "
                    f"{html.escape(first)}. Joint colors are the ones in the table above "
                    "and in the playback page. Which coordinate points up is not stated in "
                    f"the files.</p>{img}")
    parts.append(_table(["family", "slices", "fps", "keypoints", "dims", "bones", "joints"],
                        schema_rows) + "".join(figs))

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

    parts.append("<h3>All rows</h3><p>One row per slice and method; click a column header to "
                 "sort. Columns run from the descriptive results (clusters, median bout, repeat "
                 "ARI, noise, flag) to the run details. The slice cell is tinted by dataset "
                 "family, the method cell carries the method color, and the repeat ARI cell is "
                 "shaded by its value. No cell is marked “best”: no column measures quality, "
                 "and more clusters or longer bouts are not better.</p>")
    fam_var = {f: FAMILY_VARS[i % len(FAMILY_VARS)] for i, f in enumerate(families)}
    body = []
    for r in sorted(ok, key=lambda r: (datasets.index(r["dataset"]), r["method"])):
        key = (r["dataset"], r["method"])
        tds = []
        for c in ROW_COLS:
            style = ""
            if c == "dataset":
                style = (f"background:color-mix(in srgb, var({fam_var[_family(r['dataset'])]}) "
                         "22%, transparent)")
            elif c == "method":
                style = f"border-left:5px solid {color[r['method']]}"
            elif c == "repeat_ari_mean" and r.get(c) is not None:
                style = (f"background:color-mix(in srgb, var(--a1) {max(0, r[c]) * 45:.0f}%, "
                         "transparent)" + (";font-weight:600" if top.get(r["dataset"])
                                           == r["method"] else ""))
            if c == "flag":
                tds.append('<td style="color:var(--warn)">'
                           f'{html.escape("; ".join(flags[key]))}</td>')
            else:
                tds.append(f'<td style="{style}">{_fmt(r.get(c))}</td>')
        body.append(f"<tr>{''.join(tds)}</tr>")
    head = "".join(f"<th>{html.escape(ROW_HEAD.get(c, c))}</th>" for c in ROW_COLS)
    parts.append(f'<table class="sortable"><thead><tr>{head}</tr></thead><tbody>'
                 f"{''.join(body)}</tbody></table>")

    parts.append("<h3>Overview charts</h3><p>The same rows as charts: one dot per slice, colored "
                 "by dataset family, grouped by method. Hollow dots are flagged rows.</p>"
                 + _img(plot_grid_overview(ok, flags, families)))

    for fam in families:
        group = [d for d in datasets if _family(d) == fam]
        parts.append(f"<h2>{html.escape(FAMILIES.get(fam, fam))}</h2><p>One subsection per "
                     "slice; each block opens on click. ARI and AMI compare methods on the same "
                     "frames; the shifted null is the ARI after sliding one sequence in time; "
                     "“B inside A” near 1 with a low reverse value means B is a finer split of "
                     "A.</p>")
        for ds in group:
            notes = by_name[ds]["notes"]
            parts.append(f"<h3>{html.escape(ds)}</h3>" + slice_blocks(
                batch_dir, ds, _keypoints(batch_dir, ds), fps[ds], emb=emb.get(ds),
                explained=explained.get(fam), lengths=notes.get("lengths"),
                recordings=notes.get("recordings")))
        if fam + "_" in SAME_FRAMES_PREFIXES and len(group) > 1:
            body = ["<p>The slices are the same frames under different pose post-processing, "
                    "so one method's labels can be compared across them. Read each table "
                    "against that method's repeat ARI in the result grid.</p>"]
            for m in sorted({k for d in group for k in _labels(batch_dir, d)}):
                seqs = {d: _labels(batch_dir, d)[m] for d in group if m in _labels(batch_dir, d)}
                if len(seqs) > 1 and len({len(v) for v in seqs.values()}) == 1:
                    body.append(f"<p><b>{html.escape(m)}</b></p>"
                                + _table(PAIR_HEADER, _pairs(label_agreement(seqs))))
            parts.append(f"<h3>{html.escape(fam)}: pose variants compared</h3>" + "".join(body))

    parts.append("<h2>Animations</h2><ul>"
                 f'<li><a href="{html.escape(gallery_href)}">Cluster gallery</a>: skeleton '
                 "animations of each method's clusters and of the best matched cluster pairs.</li>"
                 f'<li><a href="{html.escape(player_href)}">Playback</a>: the whole recording '
                 "with the skeleton, the position on the cluster map and every method's label "
                 "sequence in sync.</li></ul>")

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
                 "between recordings, and clusters are not matched across recordings.</li>"
                 "<li>Cluster matching is by shared frames, so it needs the same frames: it "
                 "cannot say that two recordings contain the same behavior.</li>"
                 "<li>B-SOiD labels are 10 Hz bins repeated to frames; its bins start at frame 0 "
                 "and drop a tail shorter than two bins.</li></ul>")

    parts.append(SORT_JS)
    return _write(Path(out_html) if out_html else batch_dir / "batch_report.html", title, parts)


if __name__ == "__main__":
    from .grid_gallery import render_gallery, render_player

    args = sys.argv[2:] + ["batch_gallery.html", "batch_player.html"][len(sys.argv) - 2:]
    print(render_grid_report(sys.argv[1], gallery_href=args[0], player_href=args[1]))
    print(render_gallery(sys.argv[1]))
    print(render_player(sys.argv[1]))
