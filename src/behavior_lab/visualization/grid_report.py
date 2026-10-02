"""Source HTML for one dataset x method grid (``outputs/behavior_analysis_workbench/<out>/``).

Plain content only (h1, header-meta, tldr, sections with embedded figures); the vault's
``build_page.py`` supplies the theme and navigation. Rebuild from the batch folder; do not
hand-edit the output. Usage:
``python -m behavior_lab.visualization.grid_report <batch_dir> [gallery_href] [player_href]
[figs_href] [baseline_dir] [baseline_href]``; ``baseline_dir`` is a batch run under each
method's own settings, summarized in one table on this page; figures are embedded as previews and
link to full-resolution PNGs written to
``<batch_dir>/figs_report/`` (copy that folder next to the published page as ``figs_href``);
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
    set_figs,
)
from .agreement import label_agreement
from .grid_reading import reading_section
from .grid_slice import (
    LEAD_METRICS,
    baseline_block,
    plot_grid_overview,
    schema_block,
    slice_blocks,
    slice_leads,
    trend_summary,
)


def render_grid_report(batch_dir: str | Path, out_html: str | Path | None = None, *,
                       gallery_href: str = "batch_gallery.html",
                       player_href: str = "batch_player.html",
                       figs_href: str = "figs_report",
                       baseline_dir: str | Path | None = None,
                       baseline_href: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")

    batch_dir = Path(batch_dir)
    set_figs(batch_dir / "figs_report", figs_href)  # full-resolution copies, opened on click
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

    tldr_at = len(parts)  # filled after the rows are analysed (leads need the flags)
    parts.append("")

    families = list(dict.fromkeys(_family(d) for d in datasets))
    by_name = {s["name"]: s for s in slices}
    settings_file = batch_dir / "method_settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    parts.append("<h2>Data and methods</h2><p>What was recorded and what each method does "
                 "with it. Every cell is one unsupervised fit: no pretrained model, no labels."
                 "</p><h3>Datasets</h3>")
    def spec(f: str) -> list[object]:
        rec = [by_name[d] for d in datasets if _family(d) == f
               and not by_name[d]["notes"].get("lengths")]  # pooled slices repeat the same frames
        frames = sorted({s["shape"][0] for s in rec})
        fps_ = sorted({s["fps"] for s in rec})
        minutes = sorted({round(s["shape"][0] / s["fps"] / 60, 1) for s in rec})

        def span(v):
            return f"{v[0]:g}" if len(v) == 1 else f"{v[0]:g} to {v[-1]:g}"

        return [len(rec), f'{rec[0]["shape"][1]} × {rec[0]["shape"][2]}D', span(fps_),
                span(frames), span(minutes)]

    parts.append(_table(["family", "recordings", "keypoints", "fps", "frames per recording",
                         "minutes per recording", "slices", "what it is"],
                        [[FAMILIES.get(f, f), *spec(f),
                          ", ".join(d for d in datasets if _family(d) == f),
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

    lead = slice_leads(ok, flags)
    lead_n = {(r["dataset"], r["method"]): sum(lead.get((r["dataset"], k)) == r["method"]
                                               for k in LEAD_METRICS) for r in ok}
    parts[tldr_at] = ('<div class="tldr"><ul>'
                      f"<li>{len(ok)} of {len(rows)} cells ran; {len(failed)} failed or are not "
                      "applicable (listed under Failures).</li>"
                      + "".join(f"<li>{x}</li>"
                                for x in trend_summary(batch_dir, ok, lead,
                                                       [by_name[d] for d in datasets]))
                      + "<li>Unsupervised labels have no ground truth here: every number "
                      "describes agreement or stability, not accuracy.</li></ul></div>")
    parts.append("<h3>All rows</h3><p>One row per slice and method; click a column header to "
                 "sort. Columns run from the descriptive results (clusters, median bout, repeat "
                 "ARI, noise, flag) to the run details. The slice cell is tinted by dataset "
                 "family and the method cell carries the method color.</p><ul>"
                 "<li><b>▲ and bold</b>: the leading value of that slice in the column "
                 "(highest repeat ARI, longest median bout, lowest noise) among unflagged rows; "
                 "ties are not marked. The method cell of a leading row is bold and shows how "
                 "many columns it leads.</li>"
                 "<li>A lead is a description, not a ranking of quality: there is no ground "
                 "truth, a trivial segmentation is easy to repeat, and a longer bout is not a "
                 "better bout. The cluster count has no leading direction.</li></ul>")
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
                if lead_n[key]:
                    style += ";font-weight:700"
            elif c == "repeat_ari_mean" and r.get(c) is not None:
                style = (f"background:color-mix(in srgb, var(--a1) {max(0, r[c]) * 45:.0f}%, "
                         "transparent)")
            leads_here = lead.get((r["dataset"], c)) == r["method"]
            if leads_here:
                style += (";font-weight:700;outline:2px solid var(--a1);outline-offset:-2px")
            text = _fmt(r.get(c))
            if c == "method" and lead_n[key]:
                text += f' <span style="color:var(--a1)">▲×{lead_n[key]}</span>'
            elif leads_here:
                text = "▲ " + text
            if c == "flag":
                tds.append('<td style="color:var(--warn)">'
                           f'{html.escape("; ".join(flags[key]))}</td>')
            else:
                v = r.get(c)
                dv = f' data-v="{v}"' if isinstance(v, (int, float)) else ""
                tds.append(f'<td style="{style}"{dv}>{text}</td>')
        body.append(f"<tr>{''.join(tds)}</tr>")
    head = "".join(f"<th>{html.escape(ROW_HEAD.get(c, c))}</th>" for c in ROW_COLS)
    parts.append(f'<table class="sortable"><thead><tr>{head}</tr></thead><tbody>'
                 f"{''.join(body)}</tbody></table>")

    if baseline_dir:
        parts.append(baseline_block(batch_dir, Path(baseline_dir), baseline_href))

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

    parts.append(reading_section(batch_dir, ok, flags, [by_name[d] for d in datasets]))

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
    set_figs(None, None)
    return _write(Path(out_html) if out_html else batch_dir / "batch_report.html", title, parts)


if __name__ == "__main__":
    from .grid_gallery import render_gallery, render_player

    defaults = ["batch_gallery.html", "batch_player.html", "figs_report", "", ""]
    args = sys.argv[2:] + defaults[len(sys.argv) - 2:]
    print(render_grid_report(sys.argv[1], gallery_href=args[0], player_href=args[1],
                             figs_href=args[2], baseline_dir=args[3] or None,
                             baseline_href=args[4] or None))
    print(render_gallery(sys.argv[1]))
    print(render_player(sys.argv[1]))
