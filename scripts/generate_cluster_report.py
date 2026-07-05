#!/usr/bin/env python3
"""Generate an interactive HTML report for clustering comparison results.

Combines comparison PNG figures with per-cluster skeleton GIF animations
into a self-contained HTML report with tab navigation.

Requires: Pre-computed results from compare_clustering.py (cached .npz files
and comparison PNGs). If cache is missing, re-runs models.

Usage:
    python scripts/generate_cluster_report.py
    python scripts/generate_cluster_report.py --models clustering,cebra
    python scripts/generate_cluster_report.py --max-clusters 6
    python scripts/generate_cluster_report.py --rerun
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from behavior_lab.core.skeleton import get_skeleton
from behavior_lab.visualization.html_report import (
    generate_cluster_animations,
    generate_pipeline_report,
    image_to_base64,
)

# Reuse data loaders and model runners from compare_clustering
from compare_clustering import (
    OUTPUT_DIR,
    ModelResult,
    compute_metrics,
    load_calms21,
    load_mabe22,
    load_results_cache,
    load_shank3ko,
    load_subtle,
    run_bsoid,
    run_cebra,
    run_clustering,
    run_moseq_fallback,
    run_subtle,
)

# Dataset name → skeleton registry key
SKELETON_MAP = {
    "CalMS21": "calms21",
    "SUBTLE": "subtle",
    "Shank3KO": "shank3ko",
    "MABe22": "mabe22",
}

# All available model runners (order matters for display)
MODEL_RUNNERS = {
    "Clustering": run_clustering,
    "B-SOiD": run_bsoid,
    "MoSeq (HMM)": run_moseq_fallback,
    "SUBTLE": run_subtle,
    "CEBRA": run_cebra,
}

GIF_DIR = OUTPUT_DIR / "gifs"


def load_or_run_models(
    dataset: dict,
    model_filter: list[str] | None = None,
    force_rerun: bool = False,
) -> list[ModelResult]:
    """Load cached results or re-run models for a dataset.

    Args:
        dataset: Dataset dict from compare_clustering loaders.
        model_filter: If set, only include these model names (case-insensitive).
        force_rerun: If True, skip cache and re-run all models.

    Returns:
        List of ModelResult objects.
    """
    ds_name = dataset["name"]
    results = []

    # Try loading from cache
    if not force_rerun:
        cached = load_results_cache(ds_name)
        if cached is not None:
            model_names = cached.get("__model_names", np.array([]))
            elapsed_arr = cached.get("__elapsed", np.array([]))
            errors_arr = cached.get("__errors", np.array([]))
            for i, mname in enumerate(model_names):
                mname = str(mname)

                # Filter models
                if model_filter:
                    if not any(mname.lower().startswith(f.lower()) for f in model_filter):
                        continue

                labels_key = f"{mname}__labels"
                emb_key = f"{mname}__embedding_2d"
                feat_key = f"{mname}__features"

                labels = cached.get(labels_key)
                if labels is None or len(labels) == 0:
                    error = str(errors_arr[i]) if i < len(errors_arr) else "No labels"
                    results.append(ModelResult(
                        model_name=mname, dataset_name=ds_name,
                        labels=np.array([]), error=error if error else "No labels",
                    ))
                    continue

                embedding = cached.get(emb_key)
                features = cached.get(feat_key)
                elapsed = float(elapsed_arr[i]) if i < len(elapsed_arr) else 0.0

                results.append(ModelResult(
                    model_name=mname,
                    dataset_name=ds_name,
                    labels=labels,
                    embedding_2d=embedding,
                    n_clusters=len(set(labels) - {-1}),
                    features=features,
                    elapsed_sec=elapsed,
                ))

            if results:
                print(f"  Loaded {len(results)} cached results for {ds_name}")
                return results
            print(f"  Cache found but no matching models for {ds_name}, re-running...")

    # Re-run models
    print(f"  Running models for {ds_name}...")
    for mname, runner in MODEL_RUNNERS.items():
        if model_filter and not any(mname.lower().startswith(f.lower()) for f in model_filter):
            continue
        try:
            print(f"    {mname}...")
            r = runner(dataset)
            results.append(r)
            if r.error:
                print(f"      SKIP: {r.error}")
            else:
                print(f"      {r.n_clusters} clusters, {r.elapsed_sec:.1f}s")
        except Exception as e:
            print(f"      FAILED: {e}")
            results.append(ModelResult(mname, ds_name, np.array([]), error=str(e)))

    return results


def select_top_models(
    results: list[ModelResult],
    dataset: dict,
    top_n: int = 2,
) -> list[ModelResult]:
    """Select top N models by silhouette score for GIF generation.

    Always includes the top-ranked models. Errors and no-label results
    are excluded from ranking but may still appear in the report table.
    """
    scored = []
    gt_labels = dataset.get("labels")
    for r in results:
        if r.error or r.labels is None or len(r.labels) == 0:
            continue
        metrics = compute_metrics(r.labels, r.features, gt_labels)
        sil = metrics.get("silhouette", -1.0)
        scored.append((sil, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_n]]


def build_report_data(
    datasets: list[dict],
    all_results: dict[str, list[ModelResult]],
    top_n_gifs: int = 2,
    max_clusters: int = 8,
    n_frames: int = 480,
    fps: float = 15.0,
) -> dict:
    """Build the report_data dict consumed by generate_pipeline_report."""
    report_datasets = {}

    for dataset in datasets:
        ds_name = dataset["name"]
        ds_key = ds_name.lower()
        results = all_results.get(ds_name, [])

        ds_data = {}

        # Data summary
        kp = dataset["keypoints"]
        ds_data["data"] = {
            "shape": str(kp.shape),
            "total_frames": kp.shape[0],
            "n_joints": kp.shape[1] if kp.ndim >= 2 else "?",
            "ndim": dataset.get("ndim", kp.shape[-1] if kp.ndim >= 2 else "?"),
            "fps": dataset.get("fps", "?"),
            "gt_labels": "Yes" if dataset.get("labels") is not None else "No",
        }

        # Comparison PNG (from compare_clustering.py)
        comparison_png = OUTPUT_DIR / f"comparison_{ds_key}.png"
        if comparison_png.exists():
            ds_data["comparison_image"] = image_to_base64(comparison_png)

        # Metrics table
        gt_labels = dataset.get("labels")
        table_headers = ["Model", "Clusters", "Silhouette", "ARI", "NMI", "Time (s)"]
        table_rows = []
        for r in results:
            if r.error:
                table_rows.append([r.model_name, "ERR", "-", "-", "-", f"({r.error[:50]})"])
                continue
            m = compute_metrics(r.labels, r.features, gt_labels)
            sil = f"{m['silhouette']:.3f}" if "silhouette" in m else "-"
            ari = f"{m['ARI']:.3f}" if "ARI" in m else "-"
            nmi = f"{m['NMI']:.3f}" if "NMI" in m else "-"
            table_rows.append([
                r.model_name, str(m["n_clusters"]),
                sil, ari, nmi, f"{r.elapsed_sec:.1f}",
            ])
        ds_data["metrics_table"] = {"headers": table_headers, "rows": table_rows}

        # Per-model cluster GIFs (top N by silhouette)
        skeleton_key = SKELETON_MAP.get(ds_name)
        if skeleton_key:
            try:
                skeleton = get_skeleton(skeleton_key)
            except ValueError:
                skeleton = None
        else:
            skeleton = None

        model_cluster_gifs = {}
        if skeleton is not None:
            top_models = select_top_models(results, dataset, top_n=top_n_gifs)
            for r in top_models:
                gif_out = GIF_DIR / ds_key / r.model_name.lower().replace(" ", "_")
                print(f"  Generating GIFs: {ds_name} / {r.model_name}")
                gifs = generate_cluster_animations(
                    keypoints=kp,
                    labels=r.labels,
                    skeleton=skeleton,
                    out_dir=gif_out,
                    n_frames=n_frames,
                    fps=fps,
                    max_clusters=max_clusters,
                )
                if gifs:
                    model_cluster_gifs[r.model_name] = gifs

        ds_data["model_cluster_gifs"] = model_cluster_gifs

        # GT per-class GIFs (if ground truth labels exist)
        if dataset.get("labels") is not None and dataset.get("class_names"):
            ds_data["data"]["classes"] = ", ".join(dataset["class_names"])

        report_datasets[ds_key] = ds_data

    return {
        "title": "Clustering Comparison Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": report_datasets,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate clustering comparison HTML report")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to include (default: all)")
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run models instead of using cache")
    parser.add_argument("--max-clusters", type=int, default=8,
                        help="Max clusters per model for GIF generation")
    parser.add_argument("--top-n", type=int, default=2,
                        help="Number of top models (by silhouette) to generate GIFs for")
    parser.add_argument("--n-frames", type=int, default=480,
                        help="Frames per GIF animation")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="GIF playback fps")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: outputs/clustering_comparison/report.html)")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "report.html"

    print("=" * 60)
    print("Clustering Comparison HTML Report Generator")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    datasets = []

    ds = load_calms21(n_sequences=2000)
    datasets.append(ds)
    print(f"  CalMS21: {ds['keypoints'].shape}")

    ds = load_subtle()
    if ds:
        datasets.append(ds)
        print(f"  SUBTLE: {ds['keypoints'].shape}")

    ds = load_shank3ko()
    if ds:
        datasets.append(ds)
        print(f"  Shank3KO: {ds['keypoints'].shape}")

    ds = load_mabe22()
    if ds:
        datasets.append(ds)
        print(f"  MABe22: {ds['keypoints'].shape}")

    # Load/run models for each dataset
    print("\nLoading model results...")
    all_results: dict[str, list[ModelResult]] = {}
    for dataset in datasets:
        results = load_or_run_models(dataset, model_filter, args.rerun)
        all_results[dataset["name"]] = results

    # Build report data with GIF generation
    print("\nBuilding report...")
    report_data = build_report_data(
        datasets=datasets,
        all_results=all_results,
        top_n_gifs=args.top_n,
        max_clusters=args.max_clusters,
        n_frames=args.n_frames,
        fps=args.fps,
    )

    # Generate HTML
    print(f"\nGenerating HTML report: {output_path}")
    generate_pipeline_report(report_data, output_path)

    size_kb = output_path.stat().st_size / 1024
    print(f"\nDone! Report: {output_path} ({size_kb:.0f} KB)")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
