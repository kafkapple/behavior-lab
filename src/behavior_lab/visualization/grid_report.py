"""Source HTML for one dataset x method grid (``outputs/behavior_analysis_workbench/<out>/``).

Plain content only (h1, header-meta, tldr, sections with embedded figures); the vault's
``build_page.py`` supplies the theme and navigation. Rebuild from the batch folder; do not
hand-edit the output. Usage: ``python -m behavior_lab.visualization.grid_report <batch_dir>``.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from .agreement import label_agreement, plot_label_agreement

# Slices that are the same frames under different pose post-processing: labels are comparable
# frame by frame across the slices of one group.
SAME_FRAMES_PREFIXES = ("avatar_",)
COLS = ["dataset", "method", "n_frames", "n_clusters", "median_bout_sec", "noise_frac",
        "n_repeats", "repeat_ari_mean", "repeat_ari_min", "elapsed_sec"]


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


def render_grid_report(batch_dir: str | Path, out_html: str | Path | None = None) -> Path:
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

    parts.append("<h2>Result grid</h2><p>One row per slice and method. Repeat ARI is the pairwise "
                 "ARI between runs that differ only in seed; a between-method or between-slice "
                 "ARI below it is the only kind worth describing.</p>")
    parts.append(_table(COLS, [[r.get(c) for c in COLS]
                               for r in sorted(ok, key=lambda r: (r["dataset"], r["method"]))]))

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


if __name__ == "__main__":
    print(render_grid_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
