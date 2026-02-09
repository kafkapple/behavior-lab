#!/usr/bin/env python3
"""Visualize Feature Backend pipeline on real CalMS21 data.

Generates a multi-panel figure comparing:
1. Raw kinematic features (4D) distribution
2. SkeletonBackend → PCA → UMAP 2D embedding (colored by behavior)
3. Temporal aggregation comparison (frame vs segment level)
4. Clustering result: ethogram + transition matrix + dendrogram
5. Feature correlation heatmap across backends

Uses existing viz patterns: dpi=150, tab20, alpha=0.5, figsize conventions.

Usage:
    LOKY_MAX_CPU_COUNT=1 python scripts/viz_feature_backends.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "outputs" / "feature_backend_viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CalMS21 behavior classes
CLASS_NAMES = ["attack", "investigation", "mount", "other"]
CMAP = "tab20"
DPI = 150


def load_calms21():
    """Load CalMS21 data → (N_seq, T, K*2, 2) for mouse 0, labels."""
    data = np.load(ROOT / "data" / "calms21" / "calms21_aligned.npz")
    x = data["x_train"]  # (19144, 2, 64, 7, 2)
    y = data["y_train"]  # (19144, 4) one-hot

    # Use mouse 0, flatten to (N_seq, T=64, K=7, D=2)
    keypoints = x[:, 0]  # (19144, 64, 7, 2)
    labels = np.argmax(y, axis=1)  # (19144,)

    return keypoints, labels


def fig1_feature_distributions(features_per_class, labels):
    """Panel 1: Per-feature violin/box plots colored by behavior class."""
    import matplotlib.pyplot as plt

    feat_names = ["Speed", "Acceleration", "Body Spread", "Spatial Variance"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    unique_labels = sorted(set(labels))
    cmap_obj = plt.cm.get_cmap(CMAP, len(unique_labels))
    colors = [cmap_obj(i) for i in range(len(unique_labels))]

    for col, (ax, fname) in enumerate(zip(axes, feat_names)):
        bp_data = []
        bp_labels = []
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            vals = features_per_class[mask, col]
            # Subsample for speed
            if len(vals) > 2000:
                vals = np.random.default_rng(42).choice(vals, 2000, replace=False)
            bp_data.append(vals)
            bp_labels.append(CLASS_NAMES[lbl])

        bplot = ax.boxplot(bp_data, patch_artist=True, widths=0.6,
                           medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(bp_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(fname, fontsize=10)
        ax.set_ylabel("Value" if col == 0 else "", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Kinematic Feature Distributions by Behavior (CalMS21)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_feature_distributions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [1/5] Feature distributions saved")


def fig2_embedding_comparison(features_frame, labels_frame, features_seg, labels_seg):
    """Panel 2: Frame-level vs segment-level UMAP embedding."""
    import matplotlib.pyplot as plt
    from behavior_lab.visualization import plot_embedding

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Frame-level
    plot_embedding(
        features_frame[:, :2], labels=labels_frame,
        title="Frame-Level Features (PCA 2D)", alpha=0.3, s=0.5,
        cmap=CMAP, class_names=CLASS_NAMES, ax=axes[0],
    )

    # Segment-level
    plot_embedding(
        features_seg[:, :2], labels=labels_seg,
        title="Segment-Level Features (Temporal Agg, PCA 2D)", alpha=0.3, s=1.0,
        cmap=CMAP, class_names=CLASS_NAMES, ax=axes[1],
    )

    fig.suptitle("Frame vs Segment Embedding (CalMS21, SkeletonBackend)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_embedding_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [2/5] Embedding comparison saved")


def fig3_umap_clusters(embedding_2d, cluster_labels, gt_labels):
    """Panel 3: UMAP colored by (a) cluster assignment (b) ground truth."""
    import matplotlib.pyplot as plt
    from behavior_lab.visualization import plot_embedding

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cluster_names = [f"C{i}" for i in range(len(set(cluster_labels)))]
    plot_embedding(
        embedding_2d, labels=cluster_labels,
        title="Unsupervised Clusters (KMeans)", alpha=0.3, s=1.0,
        cmap=CMAP, class_names=cluster_names, ax=axes[0],
    )

    plot_embedding(
        embedding_2d, labels=gt_labels,
        title="Ground Truth Labels", alpha=0.3, s=1.0,
        cmap=CMAP, class_names=CLASS_NAMES, ax=axes[1],
    )

    fig.suptitle("UMAP Embedding: Clusters vs Ground Truth (CalMS21)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_umap_clusters.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [3/5] UMAP cluster comparison saved")


def fig4_behavior_analysis(cluster_labels, gt_labels, cluster_centers):
    """Panel 4: Ethogram + Transition Matrix + Dendrogram."""
    import matplotlib.pyplot as plt
    from behavior_lab.visualization import (
        plot_temporal_raster, plot_transition_matrix,
        plot_behavior_dendrogram,
    )

    fig = plt.figure(figsize=(16, 10))

    # Layout: top row = ethograms, bottom-left = transition, bottom-right = dendrogram
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 3], hspace=0.4, wspace=0.3)

    # Ethogram: first 3000 frames of cluster labels
    ax_eth_cluster = fig.add_subplot(gs[0, :])
    n_show = min(3000, len(cluster_labels))
    cluster_names = [f"C{i}" for i in range(len(set(cluster_labels[:n_show])))]
    plot_temporal_raster(
        cluster_labels[:n_show], fps=30.0,
        class_names=cluster_names,
        title="Cluster Ethogram (first 100s)",
        cmap=CMAP, ax=ax_eth_cluster,
    )

    # Ethogram: ground truth
    ax_eth_gt = fig.add_subplot(gs[1, :])
    plot_temporal_raster(
        gt_labels[:n_show], fps=30.0,
        class_names=CLASS_NAMES,
        title="Ground Truth Ethogram (first 100s)",
        cmap=CMAP, ax=ax_eth_gt,
    )

    # Transition matrix
    ax_trans = fig.add_subplot(gs[2, 0])
    n_clusters = len(set(cluster_labels))
    trans = np.zeros((n_clusters, n_clusters))
    for i in range(len(cluster_labels) - 1):
        trans[cluster_labels[i], cluster_labels[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = trans / row_sums
    plot_transition_matrix(
        trans, class_names=cluster_names,
        title="Cluster Transition Matrix",
        ax=ax_trans,
    )

    # Dendrogram
    ax_dendro = fig.add_subplot(gs[2, 1])
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(cluster_centers, method="ward")
    dendrogram(Z, labels=cluster_names, ax=ax_dendro,
               leaf_rotation=45, leaf_font_size=8)
    ax_dendro.set_title("Behavior Hierarchy (Ward)")
    ax_dendro.set_ylabel("Distance")

    fig.suptitle("Behavior Analysis: Clusters (CalMS21, SkeletonBackend)", fontsize=12, y=1.01)
    fig.savefig(OUTPUT_DIR / "fig4_behavior_analysis.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [4/5] Behavior analysis saved")


def fig5_temporal_agg_comparison(features_raw, labels_raw):
    """Panel 5: Compare temporal aggregation methods (mean/max/concat_stats)."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from behavior_lab.data.features.temporal import aggregate_temporal
    from behavior_lab.visualization import plot_embedding

    methods = ["mean", "max", "concat_stats"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, method in zip(axes, methods):
        agg = aggregate_temporal(features_raw, window_size=64, stride=32, method=method)
        # Assign labels by majority vote per segment
        n_seg = agg.shape[0]
        seg_labels = np.zeros(n_seg, dtype=int)
        for i in range(n_seg):
            start = i * 32
            end = min(start + 64, len(labels_raw))
            if start < len(labels_raw):
                seg_labels[i] = np.bincount(labels_raw[start:end]).argmax()

        # PCA → 2D
        n_comp = min(agg.shape[0], agg.shape[1], 50)
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(agg)

        dim_label = f"{agg.shape[1]}D"
        plot_embedding(
            X_pca[:, :2], labels=seg_labels,
            title=f"{method} ({dim_label})", alpha=0.4, s=2.0,
            cmap=CMAP, class_names=CLASS_NAMES, ax=ax,
        )

    fig.suptitle("Temporal Aggregation Methods Comparison (window=64, stride=32)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_temporal_methods.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [5/5] Temporal aggregation comparison saved")


def main():
    print("Loading CalMS21 data...")
    keypoints, labels = load_calms21()
    print(f"  Loaded: {keypoints.shape}, labels: {labels.shape}")

    # Subsample for manageable viz (4000 sequences per class, or all if fewer)
    rng = np.random.default_rng(42)
    n_per_class = 1000
    indices = []
    for lbl in range(4):
        cls_idx = np.where(labels == lbl)[0]
        if len(cls_idx) > n_per_class:
            cls_idx = rng.choice(cls_idx, n_per_class, replace=False)
        indices.append(cls_idx)
    indices = np.concatenate(indices)
    rng.shuffle(indices)

    kp_sub = keypoints[indices]  # (N_sub, 64, 7, 2)
    labels_sub = labels[indices]  # (N_sub,)
    print(f"  Subsampled: {kp_sub.shape}")

    # === Extract features ===
    print("\nExtracting SkeletonBackend features...")
    from behavior_lab.data.features import SkeletonBackend, FeaturePipeline
    from behavior_lab.data.features.temporal import aggregate_temporal
    from behavior_lab.models.discovery.clustering import cluster_features

    backend = SkeletonBackend(fps=30.0)

    # Per-sequence extraction → (N_sub, 64, 4) → flatten to (N_sub*64, 4)
    all_features = []
    for i in range(len(kp_sub)):
        feat = backend.extract(kp_sub[i])  # (64, 4)
        all_features.append(feat)
    all_features = np.stack(all_features)  # (N_sub, 64, 4)

    # Frame-level: flatten
    N_sub, T, D = all_features.shape
    features_flat = all_features.reshape(-1, D)  # (N_sub*64, 4)
    labels_flat = np.repeat(labels_sub, T)  # (N_sub*64,)
    print(f"  Frame-level features: {features_flat.shape}")

    # Sequence-level: mean per sequence
    features_seq = all_features.mean(axis=1)  # (N_sub, 4)
    print(f"  Sequence-level features: {features_seq.shape}")

    # === Figure 1: Feature distributions ===
    print("\nGenerating figures...")
    fig1_feature_distributions(features_seq, labels_sub)

    # === PCA for embeddings ===
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_seq = scaler.fit_transform(features_seq)

    # Frame-level PCA (subsample to 10K for speed)
    n_frame_sub = 10000
    frame_idx = rng.choice(len(features_flat), n_frame_sub, replace=False)
    X_frame = scaler.fit_transform(features_flat[frame_idx])

    # === Figure 2: Frame vs Segment embedding ===
    fig2_embedding_comparison(X_frame, labels_flat[frame_idx], X_seq, labels_sub)

    # === Clustering ===
    print("  Running clustering pipeline...")
    result = cluster_features(features_seq, n_clusters=6, use_umap=True)
    cluster_labels = result["labels"]
    embedding_2d = result["embedding_2d"]

    # === Figure 3: UMAP clusters vs GT ===
    fig3_umap_clusters(embedding_2d, cluster_labels, labels_sub)

    # === Cluster centers for dendrogram ===
    n_clusters = result["n_clusters"]
    centers = np.array([
        features_seq[cluster_labels == i].mean(axis=0)
        for i in range(n_clusters)
    ])

    # === Figure 4: Ethogram + Transition + Dendrogram ===
    fig4_behavior_analysis(cluster_labels, labels_sub, centers)

    # === Figure 5: Temporal aggregation comparison ===
    fig5_temporal_agg_comparison(features_flat[:20000], labels_flat[:20000])

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
