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


def _resample(lab: np.ndarray, T: int) -> np.ndarray:
    lab = np.asarray(lab)
    return lab if len(lab) == T else lab[np.linspace(0, len(lab) - 1, T).astype(int)]


def render_comparison_report(runs: Any, out_html: str | Path, *, fps: float = 30.0,
                             title: str = "Behavior Method Comparison",
                             ground_truth: Any = None) -> Path:
    """Render ONE self-contained HTML comparing all methods.

    Sections: (1) metrics table (+ ARI/NMI vs ground_truth when given), (2)
    multi-method ethogram, (2b) pairwise method agreement (ARI), (3) bout panels.
    All figures embedded as base64. ``ground_truth`` = optional per-frame label
    array for gold-standard ARI/NMI evaluation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _normalize(runs)

    def _r(x, n=3):
        return round(float(x), n) if isinstance(x, (int, float)) and not isinstance(x, bool) else "—"

    gt = None if ground_truth is None else np.asarray(ground_truth)

    # (1) metrics table — quantitative per-method summary (+ vs ground truth)
    truth_cols = ["ARI (GT)", "NMI (GT)"] if gt is not None else []
    headers = ["Method", "Clusters", "Silhouette", "Bouts", "Mean bout (s)",
               "Temporal consist.", "Entropy rate"] + truth_cols
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
        mean_bout = _r(np.mean(list(durations.values()))) if durations else "—"
        row = [name, n_clusters, _r(sil), int(getattr(bm, "num_bouts", 0)), mean_bout,
               _r(getattr(bm, "temporal_consistency", None)), _r(getattr(bm, "entropy_rate", None))]
        if gt is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            lab_a = _resample(labels, len(gt))
            row += [_r(adjusted_rand_score(gt, lab_a)), _r(normalized_mutual_info_score(gt, lab_a))]
        rows.append(row)
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

    # (2b) pairwise method-agreement (ARI) — quantitative, label-free.
    # Methods run at different cadences, so labels are time-aligned (resampled to
    # the longest) before comparison; high ARI = methods discover similar structure.
    if len(labels_dict) >= 2:
        try:
            from sklearn.metrics import adjusted_rand_score
            names = list(labels_dict)
            T = max(len(v) for v in labels_dict.values())
            aligned = {}
            for k, lab in labels_dict.items():
                lab = np.asarray(lab)
                aligned[k] = lab if len(lab) == T else lab[np.linspace(0, len(lab) - 1, T).astype(int)]
            n = len(names)
            mat = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    mat[i, j] = mat[j, i] = adjusted_rand_score(aligned[names[i]], aligned[names[j]])
            fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.0 * n + 1.5))
            im = ax.imshow(mat, cmap="viridis", vmin=-0.1, vmax=1.0)
            ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks(range(n)); ax.set_yticklabels(names)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
            ax.set_title("Pairwise method agreement (ARI, time-aligned)")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            figures.append(fig_to_base64(fig)); plt.close(fig)
        except Exception as exc:
            figures.append(f"<p>agreement matrix failed: {_escape(str(exc))}</p>")

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


def render_cluster_gallery(keypoints: np.ndarray, runs: Any, skeleton_name: str,
                           out_html: str | Path, *, fps: float = 30.0, n_frames: int = 150,
                           max_clusters: int = 8, title: str = "Per-cluster behavior gallery") -> Path:
    """Per-method x per-cluster representative skeleton GIF gallery (qualitative).

    For each discovered cluster, animates the longest contiguous bout of that
    cluster as a keypoint-skeleton GIF (reusing ``generate_cluster_animations``).
    Lets you *see* what each syllable/cluster looks like, per method.

    Args:
        keypoints: full ``(T, K, D)`` array the labels were computed on.
        runs: DiscoveryRun list or ``{method: {"labels": ndarray}}``.
        skeleton_name: skeleton for edge rendering (e.g. "calms21").
    """
    from ..core.skeleton import get_skeleton
    from .html_report import _escape, generate_cluster_animations

    skel = get_skeleton(skeleton_name)
    data = _normalize(runs)
    out = Path(out_html)
    gif_dir = out.parent / "_cluster_gifs"

    sections = []
    for name, d in data.items():
        labels = np.asarray(d["labels"])
        # generate_cluster_animations maps labels[i] -> keypoints[i]; if the method
        # ran on a shorter slice, align by resampled indices.
        sample_idx = None
        if len(labels) != len(keypoints):
            sample_idx = np.linspace(0, len(keypoints) - 1, len(labels)).astype(int)
        try:
            anims = generate_cluster_animations(
                keypoints, labels, skel, gif_dir / name,
                sample_indices=sample_idx, n_frames=n_frames, fps=fps, max_clusters=max_clusters)
        except Exception as exc:
            sections.append(f"<h2>{_escape(name)}</h2><p>gallery failed: {_escape(str(exc))}</p>")
            continue
        cards = "".join(
            f'<figure style="margin:6px;display:inline-block;text-align:center">'
            f'<img src="{a["src"]}" style="max-width:220px;border:1px solid #ccc;border-radius:6px"/>'
            f'<figcaption style="font-size:12px;color:#555">cluster {_escape(str(a["label"]))}</figcaption></figure>'
            for a in anims)
        sections.append(f'<h2>{_escape(name)} ({len(anims)} clusters)</h2><div>{cards or "(no clusters)"}</div>')

    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{_escape(title)}</title>"
            "<style>body{font-family:-apple-system,sans-serif;margin:20px;color:#1e293b}"
            "h1{font-size:1.3rem}h2{font-size:1rem;margin-top:20px}</style></head><body>"
            f"<h1>{_escape(title)}</h1><p>대표 bout의 keypoint-skeleton 애니메이션 (방법별 클러스터). fps={fps}.</p>"
            + "".join(sections) + "</body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


__all__ = ["render_comparison_report", "render_cluster_gallery"]
