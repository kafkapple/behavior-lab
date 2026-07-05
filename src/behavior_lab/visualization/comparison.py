"""Single common entry point for COMPARABLE multi-method visualization.

Consolidates the scattered per-script report/plot code onto the shared
``visualization`` primitives (``analysis`` plots + ``html_report`` helpers), so
every method — light (kmeans/B-SOiD/pca-hmm/SUBTLE) or heavy/isolated
(keypoint-MoSeq/VAME) — is rendered the same way in ONE self-contained HTML.

Input is intentionally loose: a list of ``DiscoveryRun`` OR a plain
``{method: {"labels": ndarray, "features"?: ndarray, "metrics"?: dict}}`` dict,
so results collected across conda envs (as ``labels.npy``) drop straight in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation import compute_behavior_metrics, compute_cluster_metrics
from .analysis import plot_bout_duration, plot_multiscale_ethogram
from .html_report import _escape, _render_table, fig_to_base64


def _normalize(runs: Any) -> dict[str, dict]:
    """Accept DiscoveryRun list or {method: {...}} dict -> {method: {labels, features, metrics}}."""
    out: dict[str, dict] = {}
    if isinstance(runs, dict):
        for name, d in runs.items():
            out[name] = {
                "labels": np.asarray(d["labels"]),
                "features": None if d.get("features") is None else np.asarray(d["features"]),
                "metrics": dict(d.get("metrics") or {}),
            }
        return out
    for r in runs:  # DiscoveryRun
        res = r.result
        out[r.name] = {
            "labels": np.asarray(res.labels),
            "features": None if res.features is None else np.asarray(res.features),
            "metrics": dict(r.cluster_metrics or {}),
        }
    return out


def render_comparison_report(runs: Any, out_html: str | Path, *, fps: float = 30.0,
                             title: str = "Behavior Method Comparison") -> Path:
    """Render ONE self-contained HTML comparing all methods.

    Sections: (1) metrics table, (2) multi-method ethogram (aligned rasters),
    (3) bout-duration panels. All figures embedded as base64 (no external files).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _normalize(runs)

    # (1) metrics table (compute bouts uniformly; silhouette from features or provided)
    headers = ["Method", "Clusters", "Silhouette", "Bouts", "Mean bout (s)"]
    rows: list[list[Any]] = []
    labels_dict: dict[str, np.ndarray] = {}
    for name, d in data.items():
        labels = d["labels"]
        labels_dict[name] = labels
        n_clusters = int(len({int(x) for x in labels if x >= 0}))
        sil = d["metrics"].get("silhouette")
        if sil is None and d["features"] is not None and len(d["features"]) == len(labels):
            try:
                sil = compute_cluster_metrics(d["features"], labels).__dict__.get("silhouette")
            except Exception:
                sil = None
        bm = compute_behavior_metrics(labels, fps=fps)
        durations = getattr(bm, "bout_durations", {}) or {}
        mean_bout = round(float(np.mean(list(durations.values()))), 3) if durations else "—"
        rows.append([name, n_clusters, round(sil, 3) if isinstance(sil, (int, float)) else "—",
                     int(getattr(bm, "num_bouts", 0)), mean_bout])
    metrics_html = _render_table(headers, rows)

    def _fig(ret):
        """analysis.* plots return (fig, ax); unwrap and skip empties."""
        fig = ret[0] if isinstance(ret, tuple) else ret
        return fig

    # (2) multi-method ethogram (the canonical "all methods comparable" figure)
    figures: list[str] = []
    try:
        fig = _fig(plot_multiscale_ethogram(labels_dict, fps=fps, title="Ethogram — all methods aligned"))
        if fig is not None:
            figures.append(fig_to_base64(fig)); plt.close(fig)
    except Exception as exc:  # never let one plot sink the report
        figures.append(f"<p>ethogram failed: {_escape(str(exc))}</p>")

    # (3) bout-duration panels per method
    for name, d in data.items():
        try:
            bm = compute_behavior_metrics(d["labels"], fps=fps)
            fig = _fig(plot_bout_duration(getattr(bm, "bout_durations", {}) or {}, title=f"Bout duration — {name}"))
            if fig is not None:
                figures.append(fig_to_base64(fig)); plt.close(fig)
        except Exception:
            pass

    imgs = "\n".join(
        f'<img src="{f}" style="max-width:900px;width:100%;margin:10px 0"/>' if f.startswith("data:") else f
        for f in figures
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{_escape(title)}</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;line-height:1.5;color:#1e293b}}
table{{border-collapse:collapse;font-size:14px;margin:12px 0}}td,th{{border:1px solid #ddd;padding:6px 10px}}
th{{background:#f4f4f4}}h1{{font-size:1.3rem}}h2{{font-size:1rem;margin-top:22px}}</style></head><body>
<h1>{_escape(title)}</h1>
<p>{len(data)} methods · fps={fps} · self-contained (base64 figures).</p>
<h2>Metrics</h2>{metrics_html}
<h2>Comparable visualization</h2>{imgs}
</body></html>"""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


__all__ = ["render_comparison_report"]
