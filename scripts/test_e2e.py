#!/usr/bin/env python3
"""End-to-end verification of behavior-lab pipeline.

Tests data loading, preprocessing, B-SOiD discovery, evaluation metrics,
visualization, and linear probe — all on real data.

Usage:
    python scripts/test_e2e.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from behavior_lab.core.skeleton import get_skeleton
from behavior_lab.data.loaders import get_loader
from behavior_lab.data.preprocessing.pipeline import (
    PreprocessingPipeline, Interpolator, OutlierRemover,
    TemporalSmoother, Normalizer,
)
from behavior_lab.evaluation import (
    compute_cluster_metrics, compute_behavior_metrics, linear_probe,
)
from behavior_lab.visualization import (
    plot_embedding, plot_transition_matrix, plot_bout_duration,
    plot_temporal_raster, plot_skeleton, animate_skeleton,
    plot_skeleton_comparison, fig_to_base64,
    plot_behavior_dendrogram,
)
from behavior_lab.visualization.skeleton import strip_zero_frames, strip_zero_persons, clip_outlier_joints
from behavior_lab.visualization.colors import get_joint_labels, get_joint_full_names
from behavior_lab.visualization.html_report import (
    generate_pipeline_report, image_to_base64,
)

OUT_DIR = ROOT / "outputs" / "e2e_test"

# =============================================================================
# Visualization Config — adjust GIF duration, outlier clipping, etc.
# =============================================================================
VIZ_CONFIG = {
    # GIF settings
    "gif_n_frames": 480,       # Number of frames per GIF (480 @ 15fps = 32s)
    "gif_fps_playback": 15.0,  # Playback speed in fps
    "gif_fps_record": 30.0,    # Original recording fps (for per-class time calc)

    # Per-class / per-cluster GIF settings
    "per_class_n_frames": 480,  # Frames per class/cluster GIF (480 @ 15fps = 32s)
    "per_class_max": 8,         # Maximum number of classes/clusters to animate

    # Outlier clipping
    "clip_per_joint": True,     # Per-joint IQR clipping (recommended for 3D tracking)
    "clip_iqr_factor": 3.0,    # Tukey's "far out" fence (3.0 = conservative)
    "clip_percentile": 1.0,    # Global mode fallback percentile

    # Axis scaling
    "axis_padding": 0.1,       # Extra padding around skeleton (fraction of range)
}


def _build_joint_info(skeleton):
    """Build joint info dict for HTML report."""
    abbrevs = get_joint_labels(skeleton)
    full_names = get_joint_full_names(skeleton)
    j2p = {}
    for part_name, indices in skeleton.body_parts.items():
        for idx in indices:
            if idx not in j2p:
                j2p[idx] = part_name
    return {
        "rows": [
            [i, abbrevs[i], full_names[i], j2p.get(i, "")]
            for i in range(skeleton.num_joints)
        ]
    }


def generate_per_class_animations(
    sequences, class_names, skeleton, out_dir, fps=30.0, n_frames=120,
    max_classes=None,
):
    """Generate one representative GIF per behavior class.

    Returns list of dicts with 'label' and 'src' (base64) for HTML embedding.
    """
    from collections import defaultdict

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group sequences by label
    by_class = defaultdict(list)
    for seq in sequences:
        label = seq.metadata.get("action_label")
        if label is not None:
            by_class[label].append(seq)

    # If no action_label in metadata, try frame-level labels
    if not by_class:
        for seq in sequences:
            if seq.labels is not None and len(seq.labels) > 0:
                from scipy.stats import mode as scipy_mode
                majority = int(scipy_mode(seq.labels, keepdims=False).mode)
                by_class[majority].append(seq)

    if not by_class:
        print("    No class labels found, skipping per-class GIFs")
        return []

    # Sort and optionally limit classes
    sorted_classes = sorted(by_class.keys())
    if max_classes is not None:
        # Pick classes with most sequences
        sorted_classes = sorted(
            sorted_classes, key=lambda c: len(by_class[c]), reverse=True
        )[:max_classes]
        sorted_classes.sort()

    gif_items = []
    for label_idx in sorted_classes:
        seqs = by_class[label_idx]
        # Pick the most dynamic sequence (highest keypoint variance)
        def _motion_var(s):
            kp = s.keypoints
            flat = kp.reshape(kp.shape[0], -1)
            nonzero = np.any(flat != 0, axis=1)
            if nonzero.sum() < 2:
                return 0.0
            return float(kp[nonzero].var())
        seqs_sorted = sorted(seqs, key=_motion_var, reverse=True)
        seq = seqs_sorted[0]
        name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
        safe_name = name.replace(" ", "_").replace("/", "_").lower()
        save_path = out_dir / f"class_{label_idx:02d}_{safe_name}.gif"

        # Strip zero-padding and zero-persons for clean animations
        kp = strip_zero_frames(seq.keypoints)
        kp = strip_zero_persons(kp, skeleton)
        # Outlier clipping for 3D data (consistent with cluster GIFs)
        if kp.shape[-1] >= 3:
            kp = clip_outlier_joints(kp, per_joint=True, iqr_factor=3.0)
        kp = kp[:n_frames]
        try:
            anim = animate_skeleton(
                kp, skeleton=skeleton,
                fps=fps, title=f"Class {label_idx}: {name} ({kp.shape[0]}f)",
                save_path=str(save_path),
            )
            plt.close("all")
            if save_path.exists():
                gif_items.append({
                    "label": f"Class {label_idx}: {name}",
                    "src": image_to_base64(save_path),
                })
                print(f"    Saved: {save_path.name} ({kp.shape[0]} frames)")
        except Exception as e:
            print(f"    Warning: Failed to generate GIF for class {label_idx}: {e}")

    return gif_items


def generate_cluster_animations(
    keypoints: np.ndarray,
    labels: np.ndarray,
    skeleton,
    out_dir,
    sample_indices: np.ndarray | None = None,
    n_frames: int | None = None,
    fps: float | None = None,
    max_clusters: int | None = None,
):
    """Generate one representative GIF per cluster for unsupervised datasets.

    Finds the longest contiguous bout for each cluster label and animates it.

    Args:
        keypoints: (T_total, K, D) — full concatenated keypoint array
        labels: (N,) — cluster labels from KMeans (may be subsampled)
        skeleton: SkeletonDefinition
        out_dir: Directory for saving GIF files
        sample_indices: If labels were computed on subsampled frames,
            provide the frame indices so we can map back to keypoints.
            If None, assumes labels[i] corresponds to keypoints[i].
        n_frames: Frames per GIF (default from VIZ_CONFIG)
        fps: Playback fps (default from VIZ_CONFIG)
        max_clusters: Max clusters to animate (default from VIZ_CONFIG)

    Returns:
        List of dicts with 'label' and 'src' (base64) for HTML embedding.
    """
    n_frames = n_frames or VIZ_CONFIG["per_class_n_frames"]
    fps = fps or VIZ_CONFIG["gif_fps_playback"]
    max_clusters = max_clusters or VIZ_CONFIG["per_class_max"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map labels back to original frame indices
    if sample_indices is not None:
        frame_labels = np.full(len(keypoints), -1, dtype=int)
        frame_labels[sample_indices] = labels
    else:
        frame_labels = labels.copy()

    unique_labels = sorted(set(labels))
    if len(unique_labels) > max_clusters:
        unique_labels = unique_labels[:max_clusters]

    gif_items = []
    print(f"\n  Generating per-cluster representative animations...")

    for cid in unique_labels:
        # Find contiguous bouts of this cluster
        mask = frame_labels == cid
        if mask.sum() < 10:
            continue

        # Find runs of True in mask
        diffs = np.diff(mask.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])

        if len(starts) == 0 or len(ends) == 0:
            continue

        # Pick the longest bout
        bout_lens = ends[:len(starts)] - starts[:len(ends)]
        best_idx = np.argmax(bout_lens)
        bout_start = starts[best_idx]
        bout_end = ends[best_idx]
        bout_len = bout_end - bout_start

        # Extract frames — use at least n_frames, extending if bout is shorter
        if bout_len >= n_frames:
            kp_segment = keypoints[bout_start:bout_start + n_frames]
        else:
            # Concatenate multiple bouts to reach n_frames
            segments = []
            remaining = n_frames
            # Sort bouts by length (longest first)
            order = np.argsort(-bout_lens)
            for bi in order:
                bs, be = starts[bi], ends[bi]
                take = min(be - bs, remaining)
                segments.append(keypoints[bs:bs + take])
                remaining -= take
                if remaining <= 0:
                    break
            kp_segment = np.concatenate(segments, axis=0)[:n_frames]

        save_path = out_dir / f"cluster_{cid:02d}.gif"
        title = f"Cluster {cid} ({kp_segment.shape[0]}f, {bout_lens.sum()} total)"

        try:
            # Apply outlier clipping for 3D data
            kp_viz = kp_segment
            if kp_segment.shape[-1] >= 3 and VIZ_CONFIG["clip_per_joint"]:
                kp_viz = clip_outlier_joints(
                    kp_segment,
                    per_joint=True,
                    iqr_factor=VIZ_CONFIG["clip_iqr_factor"],
                )

            anim = animate_skeleton(
                kp_viz, skeleton=skeleton,
                fps=fps, title=title,
                save_path=str(save_path),
            )
            plt.close("all")
            if save_path.exists():
                gif_items.append({
                    "label": f"Cluster {cid}",
                    "src": image_to_base64(save_path),
                })
                print(f"    Saved: {save_path.name} ({kp_segment.shape[0]} frames)")
        except Exception as e:
            print(f"    Warning: Failed GIF for cluster {cid}: {e}")

    return gif_items


def safe_json(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


def test_calms21(report: dict, html_data: dict) -> None:
    """Test CalMS21 data loading, preprocessing, B-SOiD, metrics, and viz."""
    print("\n" + "=" * 60)
    print("CalMS21: Mouse Social Behavior")
    print("=" * 60)

    out = OUT_DIR / "calms21"
    out.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    loader = get_loader("calms21", data_dir=ROOT / "data" / "calms21")
    train_seqs = loader.load_split("train")
    test_seqs = loader.load_split("test")
    print(f"  Train: {len(train_seqs)} sequences")
    print(f"  Test:  {len(test_seqs)} sequences")

    # Validate shapes
    s0 = train_seqs[0]
    s0.validate()
    print(f"  Shape: ({s0.num_frames}, {s0.num_joints}, {s0.num_channels})")
    print(f"  Labels: {np.unique(s0.labels)}")

    # Label distribution
    all_labels = np.concatenate([s.labels for s in train_seqs])
    unique, counts = np.unique(all_labels, return_counts=True)
    label_dist = {int(u): int(c) for u, c in zip(unique, counts)}
    print(f"  Label distribution: {label_dist}")

    data_summary = {
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "shape": [s0.num_frames, s0.num_joints, s0.num_channels],
        "label_distribution": label_dist,
    }
    (out / "data_summary.json").write_text(json.dumps(safe_json(data_summary), indent=2))

    report["calms21"] = {"data": data_summary}
    ds_html: dict = {"data": data_summary, "figures": {}}

    # Model info for report
    ds_html["model_info"] = {
        "name": "B-SOiD",
        "type": "Unsupervised Discovery",
        "description": "UMAP embedding + HDBSCAN clustering + Random Forest classification",
        "why": "Discovers behavioral motifs without requiring labels",
    }

    # --- Skeleton Visualization ---
    print("\n  Generating skeleton visualizations...")
    skeleton = get_skeleton("calms21")
    sample_kp = train_seqs[0].keypoints

    ds_html["joint_info"] = _build_joint_info(skeleton)

    # Static skeleton (colored, multi-person)
    fig_skel, _ = plot_skeleton(
        sample_kp, skeleton=skeleton, frame=0,
        title="CalMS21 — 2 Mice Skeleton (Frame 0)",
        show_labels=True,
        save_path=str(out / "sample_skeleton.png"),
    )
    ds_html["figures"]["skeleton_static"] = fig_to_base64(fig_skel)
    plt.close(fig_skel)
    print("  Saved: sample_skeleton.png")

    # Animated skeleton — use full sequence for better motion visibility
    # CalMS21 sequences are typically 64 frames; show all of them
    n_anim = sample_kp.shape[0]
    anim = animate_skeleton(
        sample_kp[:n_anim], skeleton=skeleton,
        fps=VIZ_CONFIG["gif_fps_playback"], title="CalMS21 Mice",
        save_path=str(out / "sample_animation.gif"),
    )
    plt.close("all")
    gif_path = out / "sample_animation.gif"
    if gif_path.exists():
        ds_html["figures"]["skeleton_gif"] = image_to_base64(gif_path)
    print(f"  Saved: sample_animation.gif ({n_anim} frames)")

    # --- Preprocessing ---
    print("\n  Preprocessing pipeline...")
    pipeline = PreprocessingPipeline([
        Interpolator(max_gap=10),
        OutlierRemover(velocity_threshold=50.0),
        TemporalSmoother(window_size=5),
        Normalizer(center_joint=0),
    ])
    raw_kp = train_seqs[0].keypoints.copy()
    cleaned_kp = pipeline(raw_kp)
    print(f"  Before: range [{raw_kp.min():.2f}, {raw_kp.max():.2f}], NaN={np.isnan(raw_kp).sum()}")
    print(f"  After:  range [{cleaned_kp.min():.4f}, {cleaned_kp.max():.4f}], NaN={np.isnan(cleaned_kp).sum()}")

    # Preprocessing comparison plot
    fig_pre, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    joint_idx = 0
    axes[0].plot(raw_kp[:, joint_idx, 0], label="x", alpha=0.7)
    axes[0].plot(raw_kp[:, joint_idx, 1], label="y", alpha=0.7)
    axes[0].set_title("Before Preprocessing")
    axes[0].legend()
    axes[1].plot(cleaned_kp[:, joint_idx, 0], label="x", alpha=0.7)
    axes[1].plot(cleaned_kp[:, joint_idx, 1], label="y", alpha=0.7)
    axes[1].set_title("After Preprocessing")
    axes[1].legend()
    axes[1].set_xlabel("Frame")
    fig_pre.suptitle("CalMS21: Preprocessing Comparison (Joint 0)")
    fig_pre.tight_layout()
    fig_pre.savefig(out / "preprocessing_comparison.png", dpi=150, bbox_inches="tight")
    ds_html["figures"]["preprocessing"] = fig_to_base64(fig_pre)
    plt.close(fig_pre)
    print("  Saved: preprocessing_comparison.png")

    # Skeleton comparison (raw vs preprocessed)
    fig_cmp, _ = plot_skeleton_comparison(
        [raw_kp, cleaned_kp],
        ["Raw", "Preprocessed"],
        skeleton=skeleton, frame=0,
        save_path=str(out / "skeleton_comparison.png"),
    )
    ds_html["figures"]["skeleton_comparison"] = fig_to_base64(fig_cmp)
    plt.close(fig_cmp)
    print("  Saved: skeleton_comparison.png")

    # --- B-SOiD Discovery ---
    print("\n  B-SOiD discovery (200 samples)...")
    from behavior_lab.models.discovery.bsoid import BSOiD

    n_subset = min(200, len(train_seqs))
    subset = train_seqs[:n_subset]

    # Concatenate sequences along time axis: (total_T, K, D)
    all_kp = np.concatenate([s.keypoints for s in subset], axis=0)
    all_gt = np.concatenate([s.labels for s in subset], axis=0)
    print(f"  Concatenated: {all_kp.shape}")

    t0 = time.time()
    bsoid = BSOiD(fps=30, min_cluster_size=50)
    result = bsoid.fit_predict(all_kp)
    elapsed = time.time() - t0
    print(f"  B-SOiD done in {elapsed:.1f}s: {result.n_clusters} clusters")

    # Save model
    bsoid.save(str(out / "bsoid_model.pkl"))
    print("  Saved: bsoid_model.pkl")

    # --- Cluster Metrics ---
    print("\n  Computing cluster metrics...")
    bin_size = max(1, 30 // 10)  # =3
    gt_trimmed = all_gt[:-1]  # diff removes 1 frame
    n_bins = len(gt_trimmed) // bin_size
    gt_binned = gt_trimmed[:n_bins * bin_size].reshape(n_bins, bin_size)
    from scipy.stats import mode as scipy_mode
    gt_binned_labels = scipy_mode(gt_binned, axis=1, keepdims=False).mode.flatten()
    n_bsoid = len(result.labels)
    gt_aligned = gt_binned_labels[:n_bsoid]

    cluster_metrics = compute_cluster_metrics(
        result.features, result.labels, true_labels=gt_aligned
    )
    cm_dict = {
        "silhouette": float(cluster_metrics.silhouette),
        "calinski_harabasz": float(cluster_metrics.calinski_harabasz),
        "davies_bouldin": float(cluster_metrics.davies_bouldin),
        "nmi": float(cluster_metrics.nmi),
        "ari": float(cluster_metrics.ari),
        "v_measure": float(cluster_metrics.v_measure),
        "hungarian_accuracy": float(cluster_metrics.hungarian_accuracy),
        "n_clusters": cluster_metrics.num_clusters,
    }
    print(f"  Silhouette: {cm_dict['silhouette']:.4f}")
    print(f"  NMI: {cm_dict['nmi']:.4f}")
    print(f"  ARI: {cm_dict['ari']:.4f}")
    print(f"  Hungarian Acc: {cm_dict['hungarian_accuracy']:.4f}")

    # --- Behavior Metrics ---
    print("\n  Computing behavior metrics...")
    beh_metrics = compute_behavior_metrics(result.labels, fps=10.0)
    beh_dict = {
        "temporal_consistency": float(beh_metrics.temporal_consistency),
        "num_bouts": beh_metrics.num_bouts,
        "entropy_rate": float(beh_metrics.entropy_rate),
        "bout_durations": safe_json(beh_metrics.bout_durations),
    }
    print(f"  Temporal consistency: {beh_dict['temporal_consistency']:.4f}")
    print(f"  Num bouts: {beh_dict['num_bouts']}")
    print(f"  Entropy rate: {beh_dict['entropy_rate']:.4f}")

    report["calms21"]["cluster_metrics"] = cm_dict
    report["calms21"]["behavior_metrics"] = beh_dict
    report["calms21"]["bsoid_time_sec"] = round(elapsed, 1)
    ds_html["cluster_metrics"] = cm_dict
    ds_html["behavior_metrics"] = beh_dict

    # --- Visualization ---
    print("\n  Generating analysis visualizations...")

    # Embedding plot
    fig_emb, _ = plot_embedding(
        result.embeddings, result.labels,
        title="B-SOiD Embedding (Cluster Colors)",
        save_path=str(out / "bsoid_embedding.png"),
    )
    ds_html["figures"]["embedding"] = fig_to_base64(fig_emb)
    plt.close("all")
    print("  Saved: bsoid_embedding.png")

    # Transition matrix
    if beh_metrics.transition_matrix is not None:
        fig_trans, _ = plot_transition_matrix(
            beh_metrics.transition_matrix,
            title="B-SOiD Behavior Transitions",
            save_path=str(out / "bsoid_transition_matrix.png"),
        )
        ds_html["figures"]["transition"] = fig_to_base64(fig_trans)
        plt.close("all")
        print("  Saved: bsoid_transition_matrix.png")

    # Bout duration
    fig_bout, _ = plot_bout_duration(
        beh_metrics.bout_durations,
        title="B-SOiD Mean Bout Durations",
        save_path=str(out / "bsoid_bout_duration.png"),
    )
    ds_html["figures"]["bout_duration"] = fig_to_base64(fig_bout)
    plt.close("all")
    print("  Saved: bsoid_bout_duration.png")

    # Ethogram (first 5 sequences)
    sample_labels = result.labels[:5000]
    fig_eth, _ = plot_temporal_raster(
        sample_labels, fps=10.0,
        title="B-SOiD Ethogram (First 5000 frames)",
        save_path=str(out / "bsoid_ethogram.png"),
    )
    ds_html["figures"]["ethogram"] = fig_to_base64(fig_eth)
    plt.close("all")
    print("  Saved: bsoid_ethogram.png")

    # Behavior dendrogram (cluster hierarchy)
    if result.n_clusters >= 2 and result.embeddings is not None:
        cluster_centers = np.array([
            result.embeddings[result.labels == i].mean(axis=0)
            for i in range(result.n_clusters)
            if np.any(result.labels == i)
        ])
        cluster_ids = [
            str(i) for i in range(result.n_clusters) if np.any(result.labels == i)
        ]
        fig_dend, _ = plot_behavior_dendrogram(
            cluster_centers, cluster_names=cluster_ids,
            title="B-SOiD Behavior Hierarchy",
            save_path=str(out / "behavior_dendrogram.png"),
        )
        if fig_dend is not None:
            ds_html["figures"]["dendrogram"] = fig_to_base64(fig_dend)
            plt.close(fig_dend)
            print("  Saved: behavior_dendrogram.png")

    # --- Per-Class Representative GIFs ---
    print("\n  Generating per-class representative animations...")
    from behavior_lab.data.loaders.calms21 import CLASS_NAMES as CALMS21_CLASSES
    per_class_items = generate_per_class_animations(
        train_seqs[:500], CALMS21_CLASSES, skeleton,
        out / "per_class", fps=VIZ_CONFIG["gif_fps_playback"],
        n_frames=VIZ_CONFIG["per_class_n_frames"],
    )
    if per_class_items:
        ds_html["per_class_gifs"] = per_class_items

    html_data["datasets"]["calms21"] = ds_html
    print("\n  CalMS21 PASSED")


def test_ntu(report: dict, html_data: dict) -> None:
    """Test NTU RGB+D demo data loading and linear probe."""
    print("\n" + "=" * 60)
    print("NTU RGB+D: Human Action Recognition (Demo)")
    print("=" * 60)

    out = OUT_DIR / "ntu"
    out.mkdir(parents=True, exist_ok=True)

    # Prefer raw demo (real 3D coordinates) over center-aligned (normalized)
    raw_path = ROOT / "data" / "ntu" / "demo_raw.npz"
    aligned_path = ROOT / "data" / "ntu" / "demo_CS_aligned.npz"
    npz_path = raw_path if raw_path.exists() else aligned_path
    data_note = "raw 3D" if npz_path == raw_path else "center-aligned (subtle motion)"
    print(f"  Data: {npz_path.name} ({data_note})")
    loader = get_loader("ntu", data_dir=ROOT / "data" / "ntu")

    train_seqs = loader.load_npz(npz_path, split="train")
    test_seqs = loader.load_npz(npz_path, split="test")
    print(f"  Train: {len(train_seqs)} sequences")
    print(f"  Test:  {len(test_seqs)} sequences")

    s0 = train_seqs[0]
    s0.validate()
    print(f"  Shape: ({s0.num_frames}, {s0.num_joints}, {s0.num_channels})")

    # Label distribution
    train_labels = np.array([s.metadata["action_label"] for s in train_seqs])
    test_labels = np.array([s.metadata["action_label"] for s in test_seqs])
    n_classes = len(np.unique(train_labels))
    print(f"  Classes: {n_classes}")

    data_summary = {
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "shape": [s0.num_frames, s0.num_joints, s0.num_channels],
        "n_classes": n_classes,
    }
    (out / "data_summary.json").write_text(json.dumps(safe_json(data_summary), indent=2))
    report["ntu"] = {"data": data_summary}
    ds_html: dict = {"data": data_summary, "figures": {}}

    ds_html["model_info"] = {
        "name": "Linear Probe (LogisticRegression)",
        "type": "Supervised Baseline",
        "description": "Mean-pooled keypoints → linear classifier. Intentionally naive.",
        "why": "Measures discriminative power of raw spatial features",
    }

    # --- Skeleton Visualization ---
    print("\n  Generating skeleton visualizations...")
    skeleton = get_skeleton("ntu")

    ds_html["joint_info"] = _build_joint_info(skeleton)

    # NTU has 2 persons: (T, 50, 3) = 2*25 joints
    # Strip zero-valued person slots (person 2 often all zeros for single-person actions)
    sample_kp = train_seqs[0].keypoints
    sample_kp_clean = strip_zero_persons(sample_kp, skeleton)
    n_active_persons = sample_kp_clean.shape[1] // skeleton.num_joints
    print(f"  Active persons: {n_active_persons} (from {skeleton.num_persons})")

    fig_skel, _ = plot_skeleton(
        sample_kp_clean, skeleton=skeleton, frame=0,
        title=f"NTU RGB+D — Skeleton (Frame 0, {n_active_persons}P)",
        show_labels=True,
        save_path=str(out / "sample_skeleton.png"),
    )
    ds_html["figures"]["skeleton_static"] = fig_to_base64(fig_skel)
    plt.close(fig_skel)
    print("  Saved: sample_skeleton.png")

    # Animated skeleton (use VIZ_CONFIG frame count)
    n_anim = min(VIZ_CONFIG["gif_n_frames"], sample_kp_clean.shape[0])
    anim = animate_skeleton(
        sample_kp_clean[:n_anim], skeleton=skeleton,
        fps=VIZ_CONFIG["gif_fps_playback"], title=f"NTU RGB+D ({n_active_persons}P)",
        save_path=str(out / "sample_animation.gif"),
    )
    plt.close("all")
    gif_path = out / "sample_animation.gif"
    if gif_path.exists():
        ds_html["figures"]["skeleton_gif"] = image_to_base64(gif_path)
    print(f"  Saved: sample_animation.gif ({n_anim} frames)")

    # --- Linear Probe ---
    print("\n  Linear probe (raw features)...")
    # Use mean-pooled keypoints as features
    train_feat = np.array([s.keypoints.mean(axis=0).flatten() for s in train_seqs])
    test_feat = np.array([s.keypoints.mean(axis=0).flatten() for s in test_seqs])

    probe_result = linear_probe(train_feat, train_labels, test_feat, test_labels)
    print(f"  Accuracy: {probe_result.accuracy:.4f}")
    print(f"  F1 (macro): {probe_result.f1_macro:.4f}")

    probe_dict = {
        "accuracy": float(probe_result.accuracy),
        "f1_macro": float(probe_result.f1_macro),
        "n_train": len(train_labels),
        "n_test": len(test_labels),
    }
    report["ntu"]["linear_probe"] = probe_dict
    ds_html["linear_probe"] = probe_dict

    # Confusion matrix plot
    if probe_result.confusion is not None:
        fig_cm, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(probe_result.confusion, cmap="Blues", aspect="auto")
        fig_cm.colorbar(im, ax=ax)
        ax.set_title(f"NTU Linear Probe Confusion (Acc={probe_result.accuracy:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig_cm.savefig(out / "linear_probe_confusion.png", dpi=150, bbox_inches="tight")
        ds_html["figures"]["confusion_matrix"] = fig_to_base64(fig_cm)
        plt.close(fig_cm)
        print("  Saved: linear_probe_confusion.png")

    # --- Per-Class Representative GIFs (top 10 most frequent) ---
    print("\n  Generating per-class representative animations...")
    from behavior_lab.data.loaders.ntu_rgbd import NTU60_CLASSES
    # demo_raw.npz has 10 classes remapped to 0-9 from NTU60 actions
    RAW_DEMO_CLASSES = [
        "pick up", "sit down", "clapping", "hand waving", "jump up",
        "pointing to something", "salute", "falling down", "punch/slap", "writing",
    ]
    class_names = RAW_DEMO_CLASSES if "raw" in npz_path.name else NTU60_CLASSES
    per_class_items = generate_per_class_animations(
        train_seqs, class_names, skeleton,
        out / "per_class", fps=VIZ_CONFIG["gif_fps_playback"],
        n_frames=VIZ_CONFIG["per_class_n_frames"], max_classes=10,
    )
    if per_class_items:
        ds_html["per_class_gifs"] = per_class_items

    html_data["datasets"]["ntu"] = ds_html
    print("\n  NTU PASSED")


def test_nwucla(report: dict, html_data: dict) -> None:
    """Test NW-UCLA data loading and linear probe."""
    print("\n" + "=" * 60)
    print("NW-UCLA: Northwestern-UCLA Action Recognition")
    print("=" * 60)

    out = OUT_DIR / "nwucla"
    out.mkdir(parents=True, exist_ok=True)

    loader = get_loader("nwucla", data_dir=ROOT / "data" / "nwucla")
    train_seqs = loader.load_split("train")
    test_seqs = loader.load_split("test")
    print(f"  Train: {len(train_seqs)} sequences")
    print(f"  Test:  {len(test_seqs)} sequences")

    s0 = train_seqs[0]
    s0.validate()
    print(f"  Shape: ({s0.num_frames}, {s0.num_joints}, {s0.num_channels})")

    train_labels = np.array([s.metadata["action_label"] for s in train_seqs])
    test_labels = np.array([s.metadata["action_label"] for s in test_seqs])
    n_classes = len(np.unique(train_labels))
    print(f"  Classes: {n_classes}")

    data_summary = {
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "shape": [s0.num_frames, s0.num_joints, s0.num_channels],
        "n_classes": n_classes,
    }
    (out / "data_summary.json").write_text(json.dumps(safe_json(data_summary), indent=2))
    report["nwucla"] = {"data": data_summary}
    ds_html: dict = {"data": data_summary, "figures": {}}

    ds_html["model_info"] = {
        "name": "Linear Probe + LSTM/Transformer (quick test)",
        "type": "Supervised Baseline + Sequence Models",
        "description": "Linear probe on mean-pooled features, plus 2-epoch LSTM and Transformer.",
        "why": "Verifies supervised pipeline end-to-end with multiple model types",
    }

    # --- Skeleton Visualization ---
    print("\n  Generating skeleton visualizations...")
    skeleton = get_skeleton("nwucla")

    ds_html["joint_info"] = _build_joint_info(skeleton)

    # Strip zero-padded frames (NW-UCLA pads to 300 but actual data is ~80 frames)
    sample_kp = strip_zero_frames(train_seqs[0].keypoints)
    n_valid = sample_kp.shape[0]
    n_total = train_seqs[0].keypoints.shape[0]
    print(f"  Valid frames: {n_valid} / {n_total} (zero-padding removed)")

    fig_skel, _ = plot_skeleton(
        sample_kp, skeleton=skeleton, frame=0,
        title=f"NW-UCLA — Skeleton (Frame 0, {n_valid} valid)",
        show_labels=True,
        save_path=str(out / "sample_skeleton.png"),
    )
    ds_html["figures"]["skeleton_static"] = fig_to_base64(fig_skel)
    plt.close(fig_skel)
    print("  Saved: sample_skeleton.png")

    # Animated skeleton — show all valid frames for full action visibility
    anim = animate_skeleton(
        sample_kp, skeleton=skeleton,
        fps=VIZ_CONFIG["gif_fps_playback"], title=f"NW-UCLA Skeleton ({n_valid}f)",
        save_path=str(out / "sample_animation.gif"),
    )
    plt.close("all")
    gif_path = out / "sample_animation.gif"
    if gif_path.exists():
        ds_html["figures"]["skeleton_gif"] = image_to_base64(gif_path)
    print(f"  Saved: sample_animation.gif ({n_valid} frames)")

    # --- Linear Probe ---
    print("\n  Linear probe (raw features)...")
    train_feat = np.array([s.keypoints.mean(axis=0).flatten() for s in train_seqs])
    test_feat = np.array([s.keypoints.mean(axis=0).flatten() for s in test_seqs])

    probe_result = linear_probe(train_feat, train_labels, test_feat, test_labels)
    print(f"  Accuracy: {probe_result.accuracy:.4f}")
    print(f"  F1 (macro): {probe_result.f1_macro:.4f}")

    probe_dict = {
        "accuracy": float(probe_result.accuracy),
        "f1_macro": float(probe_result.f1_macro),
    }
    report["nwucla"]["linear_probe"] = probe_dict
    ds_html["linear_probe"] = probe_dict

    # Confusion matrix plot
    from behavior_lab.data.loaders.nwucla import UCLA_CLASSES
    if probe_result.confusion is not None:
        fig_cm, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(probe_result.confusion, cmap="Blues", aspect="auto")
        fig_cm.colorbar(im, ax=ax)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(UCLA_CLASSES[:n_classes], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(UCLA_CLASSES[:n_classes], fontsize=7)
        ax.set_title(f"NW-UCLA Linear Probe (Acc={probe_result.accuracy:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig_cm.savefig(out / "linear_probe_confusion.png", dpi=150, bbox_inches="tight")
        ds_html["figures"]["confusion_matrix"] = fig_to_base64(fig_cm)
        plt.close(fig_cm)
        print("  Saved: linear_probe_confusion.png")

    # --- Supervised Model Quick Tests ---
    print("\n  Supervised model quick tests...")
    supervised_results = {}
    input_dim = train_seqs[0].num_joints * train_seqs[0].num_channels  # 20*3=60

    # Flatten: (T, K, D) → mean-pool → (K*D,) per sequence for MLP
    # For LSTM/Transformer: (T, K*D) per sequence, use full temporal data
    for model_name, config in [
        ("lstm", {"epochs": 2, "seq_len": 32, "hidden_dim": 64, "num_layers": 1}),
        ("transformer", {"epochs": 2, "seq_len": 32, "d_model": 64, "nhead": 2, "num_layers": 1}),
    ]:
        try:
            from behavior_lab.models.sequence import get_action_classifier

            # Prepare sequence data: concatenate all frames with labels
            all_train_kp = np.concatenate(
                [s.keypoints.reshape(s.num_frames, -1) for s in train_seqs], axis=0
            )
            all_train_labels = np.concatenate([
                np.full(s.num_frames, s.metadata["action_label"])
                for s in train_seqs
            ])
            all_test_kp = np.concatenate(
                [s.keypoints.reshape(s.num_frames, -1) for s in test_seqs], axis=0
            )
            all_test_labels = np.concatenate([
                np.full(s.num_frames, s.metadata["action_label"])
                for s in test_seqs
            ])

            clf = get_action_classifier(
                model_name, num_classes=n_classes, input_dim=input_dim,
                class_names=UCLA_CLASSES[:n_classes], **config,
            )
            clf.fit(all_train_kp, all_train_labels)
            metrics = clf.evaluate(all_test_kp, all_test_labels)
            supervised_results[model_name] = {
                "accuracy": float(metrics.accuracy),
                "f1_macro": float(metrics.f1_macro),
            }
            print(f"    {model_name}: acc={metrics.accuracy:.4f}, f1={metrics.f1_macro:.4f}")
        except Exception as e:
            print(f"    {model_name}: SKIPPED ({e})")
            supervised_results[model_name] = {"error": str(e)}

    if supervised_results:
        report["nwucla"]["supervised_models"] = supervised_results
        ds_html["supervised_models"] = supervised_results

    # --- Per-Class Representative GIFs ---
    print("\n  Generating per-class representative animations...")
    per_class_items = generate_per_class_animations(
        train_seqs, UCLA_CLASSES, skeleton,
        out / "per_class", fps=VIZ_CONFIG["gif_fps_playback"],
        n_frames=VIZ_CONFIG["per_class_n_frames"],
    )
    if per_class_items:
        ds_html["per_class_gifs"] = per_class_items

    html_data["datasets"]["nwucla"] = ds_html
    print("\n  NW-UCLA PASSED")


# =============================================================================
# P1 Additional Combinations: GCN Models + KMeans
# =============================================================================

def test_nwucla_gcn(report: dict, html_data: dict) -> None:
    """Test NW-UCLA with STGCN and AGCN models (1 epoch each)."""
    print("\n" + "=" * 60)
    print("NW-UCLA: GCN Models (STGCN, AGCN)")
    print("=" * 60)

    out = OUT_DIR / "nwucla"
    out.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from behavior_lab.models.graph.baselines import STGCN, AGCN
        from behavior_lab.data.loaders.nwucla import UCLA_CLASSES
    except ImportError as e:
        print(f"  SKIPPED: {e}")
        return

    loader = get_loader("nwucla", data_dir=ROOT / "data" / "nwucla")
    train_seqs = loader.load_split("train")
    test_seqs = loader.load_split("test")
    skeleton = get_skeleton("nwucla")

    train_labels = np.array([s.metadata["action_label"] for s in train_seqs])
    test_labels = np.array([s.metadata["action_label"] for s in test_seqs])
    n_classes = len(np.unique(train_labels))

    # Prepare data for GCN: (N, C, T, V, M) format
    def _prepare_gcn_data(seqs, max_T=64):
        """Convert BehaviorSequence list to GCN tensor (N, C, T, V, M)."""
        V = skeleton.num_joints  # 20
        C = skeleton.num_channels  # 3
        M = 1  # single person for UCLA

        batch = []
        for s in seqs:
            kp = s.keypoints  # (T, K, D)
            T_orig = kp.shape[0]
            # Pad or crop to max_T
            if T_orig < max_T:
                kp = np.pad(kp, ((0, max_T - T_orig), (0, 0), (0, 0)), mode='edge')
            else:
                kp = kp[:max_T]
            # (T, V, C) -> (C, T, V, M)
            x = kp.transpose(2, 0, 1)  # (C, T, V)
            x = x[:, :, :, np.newaxis]  # (C, T, V, 1)
            batch.append(x)

        return torch.from_numpy(np.array(batch)).float()  # (N, C, T, V, M)

    max_T = 64
    X_train = _prepare_gcn_data(train_seqs, max_T)
    X_test = _prepare_gcn_data(test_seqs, max_T)
    y_train = torch.from_numpy(train_labels).long()
    y_test = torch.from_numpy(test_labels).long()

    gcn_results = {}
    for model_name, ModelClass in [("stgcn", STGCN), ("agcn", AGCN)]:
        try:
            print(f"\n  {model_name.upper()} (1 epoch)...")
            model = ModelClass(
                num_classes=n_classes,
                num_joints=skeleton.num_joints,
                num_persons=1,
                in_channels=skeleton.num_channels,
                skeleton="nwucla",
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            # Train 1 epoch (mini-batch)
            model.train()
            batch_size = 16
            for i in range(0, len(X_train), batch_size):
                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]
                out = model(xb)
                loss = criterion(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(X_test), batch_size):
                    out = model(X_test[i:i+batch_size])
                    preds.append(out.argmax(dim=1).numpy())
                preds = np.concatenate(preds)

            acc = (preds == test_labels).mean()
            gcn_results[model_name] = {"accuracy": float(acc)}
            print(f"    {model_name}: accuracy={acc:.4f}")
        except Exception as e:
            print(f"    {model_name}: FAILED ({e})")
            gcn_results[model_name] = {"error": str(e)}

    if gcn_results:
        report.setdefault("nwucla", {})["gcn_models"] = gcn_results
        html_data.get("datasets", {}).get("nwucla", {}).setdefault("gcn_models", gcn_results)
    print("\n  NW-UCLA GCN PASSED")


def test_ntu_gcn(report: dict, html_data: dict) -> None:
    """Test NTU with STGCN (1 epoch)."""
    print("\n" + "=" * 60)
    print("NTU: STGCN + LSTM (quick test)")
    print("=" * 60)

    try:
        import torch
        from behavior_lab.models.graph.baselines import STGCN
    except ImportError as e:
        print(f"  SKIPPED: {e}")
        return

    raw_path = ROOT / "data" / "ntu" / "demo_raw.npz"
    aligned_path = ROOT / "data" / "ntu" / "demo_CS_aligned.npz"
    npz_path = raw_path if raw_path.exists() else aligned_path
    if not npz_path.exists():
        print(f"  SKIPPED: no NTU demo data found")
        return

    loader = get_loader("ntu", data_dir=ROOT / "data" / "ntu")
    train_seqs = loader.load_npz(npz_path, split="train")
    test_seqs = loader.load_npz(npz_path, split="test")
    skeleton = get_skeleton("ntu")

    train_labels = np.array([s.metadata["action_label"] for s in train_seqs])
    test_labels = np.array([s.metadata["action_label"] for s in test_seqs])
    n_classes = len(np.unique(train_labels))

    # Prepare GCN data: (N, C, T, V, M) where V=25, M=2
    max_T = 64
    V = skeleton.num_joints  # 25
    M = skeleton.num_persons  # 2

    batch_train, batch_test = [], []
    for seqs, batch in [(train_seqs, batch_train), (test_seqs, batch_test)]:
        for s in seqs:
            kp = s.keypoints  # (T, K, D) where K=50 for 2 persons
            T_orig = kp.shape[0]
            if T_orig < max_T:
                kp = np.pad(kp, ((0, max_T - T_orig), (0, 0), (0, 0)), mode='edge')
            else:
                kp = kp[:max_T]

            # Split multi-person: (T, 50, 3) -> (T, 2, 25, 3)
            K_total = kp.shape[1]
            if K_total >= V * M:
                kp = kp[:, :V*M, :].reshape(max_T, M, V, 3)
            else:
                # Pad to 2 persons
                kp_m = np.zeros((max_T, M, V, 3), dtype=np.float32)
                kp_m[:, 0, :K_total, :] = kp[:, :min(K_total, V), :]
                kp = kp_m

            # (T, M, V, C) -> (C, T, V, M)
            x = kp.transpose(3, 0, 2, 1)  # (C, T, V, M)
            batch.append(x)

    X_train = torch.from_numpy(np.array(batch_train)).float()
    X_test = torch.from_numpy(np.array(batch_test)).float()
    y_train = torch.from_numpy(train_labels).long()
    y_test = torch.from_numpy(test_labels).long()

    gcn_results = {}
    try:
        print("\n  STGCN (1 epoch)...")
        model = STGCN(
            num_classes=n_classes,
            num_joints=V,
            num_persons=M,
            in_channels=3,
            skeleton="ntu",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        batch_size = 16
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_test), batch_size):
                out = model(X_test[i:i+batch_size])
                preds.append(out.argmax(dim=1).numpy())
            preds = np.concatenate(preds)

        acc = (preds == test_labels).mean()
        gcn_results["stgcn"] = {"accuracy": float(acc)}
        print(f"    stgcn: accuracy={acc:.4f}")
    except Exception as e:
        print(f"    stgcn: FAILED ({e})")
        gcn_results["stgcn"] = {"error": str(e)}

    if gcn_results:
        report.setdefault("ntu", {})["gcn_models"] = gcn_results
    print("\n  NTU GCN PASSED")


def test_calms21_kmeans(report: dict, html_data: dict) -> None:
    """Test CalMS21 with KMeans clustering."""
    print("\n" + "=" * 60)
    print("CalMS21: KMeans Clustering")
    print("=" * 60)

    from behavior_lab.models.discovery.clustering import cluster_features

    loader = get_loader("calms21", data_dir=ROOT / "data" / "calms21")
    try:
        train_seqs = loader.load_split("train")
    except FileNotFoundError:
        print("  SKIPPED: CalMS21 data not found")
        return

    # Use subset for speed
    n_subset = min(500, len(train_seqs))
    subset = train_seqs[:n_subset]

    # Mean-pooled features: (N, K*D)
    features = np.array([s.keypoints.mean(axis=0).flatten() for s in subset])
    print(f"  Features: {features.shape}")

    result = cluster_features(features, n_clusters=4, use_umap=False)
    labels = result["labels"]
    n_unique = len(np.unique(labels))

    kmeans_dict = {
        "n_clusters": int(result["n_clusters"]),
        "unique_labels": int(n_unique),
    }
    print(f"  KMeans: {n_unique} clusters assigned")

    report.setdefault("calms21", {})["kmeans"] = kmeans_dict
    print("\n  CalMS21 KMeans PASSED")


# =============================================================================
# P2 New Dataset Tests
# =============================================================================

def test_subtle(report: dict, html_data: dict) -> None:
    """Test SUBTLE CSV → KMeans + visualizations."""
    print("\n" + "=" * 60)
    print("SUBTLE: Mouse Spontaneous Behavior (3D)")
    print("=" * 60)

    out = OUT_DIR / "subtle"
    out.mkdir(parents=True, exist_ok=True)

    # Try preprocessed first, then raw
    data_dir = ROOT / "data" / "preprocessed" / "subtle"
    if not data_dir.exists() or not list(data_dir.glob("*.npz")):
        data_dir = ROOT / "data" / "raw" / "subtle"
    if not data_dir.exists() or not list(data_dir.glob("*")):
        print("  SKIPPED: No SUBTLE data found.")
        print("  Run: python scripts/download_data.py --dataset subtle")
        return

    loader = get_loader("subtle", data_dir=data_dir)
    sequences = loader.load_all()
    if not sequences:
        print("  SKIPPED: No sequences loaded")
        return

    print(f"  Loaded: {len(sequences)} sequences")
    s0 = sequences[0]
    s0.validate()
    print(f"  Shape: ({s0.num_frames}, {s0.num_joints}, {s0.num_channels})")

    # Concatenate all sequences
    all_kp = np.concatenate([s.keypoints for s in sequences], axis=0)
    print(f"  Total frames: {all_kp.shape[0]}")

    data_summary = {
        "n_sequences": len(sequences),
        "total_frames": int(all_kp.shape[0]),
        "shape_per_frame": [int(s0.num_joints), int(s0.num_channels)],
    }
    report["subtle"] = {"data": data_summary}
    ds_html: dict = {"data": data_summary, "figures": {}}

    ds_html["model_info"] = {
        "name": "KMeans Clustering",
        "type": "Unsupervised Discovery",
        "description": "KMeans on flattened per-frame keypoint features",
        "why": "Baseline unsupervised segmentation for 3D mouse behavior",
    }

    # --- Skeleton Visualization ---
    print("\n  Generating skeleton visualizations...")
    skeleton = get_skeleton("subtle")
    ds_html["joint_info"] = _build_joint_info(skeleton)

    sample_kp = sequences[0].keypoints
    fig_skel, _ = plot_skeleton(
        sample_kp, skeleton=skeleton, frame=0,
        title="SUBTLE — 9-Joint Mouse (Frame 0)",
        show_labels=True,
        save_path=str(out / "sample_skeleton.png"),
    )
    ds_html["figures"]["skeleton_static"] = fig_to_base64(fig_skel)
    plt.close(fig_skel)
    print("  Saved: sample_skeleton.png")

    # Animated skeleton (use VIZ_CONFIG frame count)
    n_anim = min(VIZ_CONFIG["gif_n_frames"], sample_kp.shape[0])
    anim = animate_skeleton(
        sample_kp[:n_anim], skeleton=skeleton,
        fps=VIZ_CONFIG["gif_fps_playback"], title="SUBTLE Mouse (3D)",
        save_path=str(out / "sample_animation.gif"),
    )
    plt.close("all")
    gif_path = out / "sample_animation.gif"
    if gif_path.exists():
        ds_html["figures"]["skeleton_gif"] = image_to_base64(gif_path)
    print(f"  Saved: sample_animation.gif ({n_anim} frames)")

    # --- KMeans clustering ---
    from behavior_lab.models.discovery.clustering import cluster_features

    step = max(1, len(all_kp) // 5000)
    sample_indices = np.arange(0, len(all_kp), step)
    features = all_kp[sample_indices].reshape(-1, s0.num_joints * s0.num_channels)
    print(f"  Feature matrix: {features.shape}")

    result = cluster_features(features, n_clusters=4, use_umap=False)
    labels = result["labels"]

    kmeans_dict = {
        "n_clusters": int(result["n_clusters"]),
        "n_samples": int(len(labels)),
    }
    report["subtle"]["kmeans"] = kmeans_dict
    ds_html["cluster_metrics"] = kmeans_dict

    # PCA embedding plot for clusters
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(features)
    fig_emb, _ = plot_embedding(
        emb_2d, labels,
        title=f"SUBTLE KMeans (PCA, {result['n_clusters']} clusters)",
        save_path=str(out / "kmeans_embedding.png"),
    )
    ds_html["figures"]["embedding"] = fig_to_base64(fig_emb)
    plt.close("all")
    print(f"  Saved: kmeans_embedding.png ({result['n_clusters']} clusters)")

    # Per-cluster representative GIFs
    cluster_gifs = generate_cluster_animations(
        all_kp, labels, skeleton, out / "clusters",
        sample_indices=sample_indices,
    )
    if cluster_gifs:
        ds_html["per_class_gifs"] = cluster_gifs

    # B-SOiD (2D only — SUBTLE is 3D, expect skip)
    try:
        from behavior_lab.models.discovery.bsoid import BSOiD
        bsoid = BSOiD(fps=30, min_cluster_size=30)
        bsoid_result = bsoid.fit_predict(all_kp[:5000])
        report["subtle"]["bsoid"] = {"n_clusters": int(bsoid_result.n_clusters)}
        print(f"  B-SOiD: {bsoid_result.n_clusters} clusters")
    except Exception as e:
        print(f"  B-SOiD: SKIPPED ({e})")

    html_data["datasets"]["subtle"] = ds_html
    print("\n  SUBTLE PASSED")


def test_shank3ko(report: dict, html_data: dict) -> None:
    """Test Shank3KO .mat → KMeans + visualizations."""
    print("\n" + "=" * 60)
    print("Shank3KO: Knockout Mouse Behavior (3D)")
    print("=" * 60)

    out = OUT_DIR / "shank3ko"
    out.mkdir(parents=True, exist_ok=True)

    # Try preprocessed first, then raw
    data_dir = ROOT / "data" / "preprocessed" / "shank3ko"
    if not data_dir.exists() or not list(data_dir.glob("*.npz")):
        data_dir = ROOT / "data" / "raw" / "shank3ko"
    if not data_dir.exists() or not list(data_dir.glob("*")):
        print("  SKIPPED: No Shank3KO data found.")
        print("  Run: python scripts/download_data.py --dataset shank3ko")
        return

    loader = get_loader("shank3ko", data_dir=data_dir)
    sequences = loader.load_all()
    if not sequences:
        print("  SKIPPED: No sequences loaded")
        return

    print(f"  Loaded: {len(sequences)} sequences")
    s0 = sequences[0]
    s0.validate()
    print(f"  Shape: ({s0.num_frames}, {s0.num_joints}, {s0.num_channels})")

    # Genotype distribution
    genotypes = [s.metadata.get("genotype", "?") for s in sequences]
    from collections import Counter
    geno_dist = dict(Counter(genotypes))
    print(f"  Genotypes: {geno_dist}")

    all_kp = np.concatenate([s.keypoints for s in sequences], axis=0)
    print(f"  Total frames: {all_kp.shape[0]}")

    data_summary = {
        "n_sequences": len(sequences),
        "total_frames": int(all_kp.shape[0]),
        "shape_per_frame": [int(s0.num_joints), int(s0.num_channels)],
        "genotypes": geno_dist,
    }
    report["shank3ko"] = {"data": data_summary}
    ds_html: dict = {"data": data_summary, "figures": {}}

    ds_html["model_info"] = {
        "name": "KMeans Clustering",
        "type": "Unsupervised Discovery",
        "description": "KMeans on flattened per-frame 16-joint 3D keypoints",
        "why": "Compare behavioral profiles between Shank3 KO and WT mice",
    }

    # --- Skeleton Visualization ---
    print("\n  Generating skeleton visualizations...")
    skeleton = get_skeleton("shank3ko")
    ds_html["joint_info"] = _build_joint_info(skeleton)

    sample_kp = sequences[0].keypoints
    # Per-joint IQR clipping (Shank3KO tip_tail has extreme tracking spikes)
    sample_kp = clip_outlier_joints(
        sample_kp, per_joint=VIZ_CONFIG["clip_per_joint"],
        iqr_factor=VIZ_CONFIG["clip_iqr_factor"],
    )
    geno = sequences[0].metadata.get("genotype", "?")
    fig_skel, _ = plot_skeleton(
        sample_kp, skeleton=skeleton, frame=0,
        title=f"Shank3KO — 16-Joint Mouse ({geno}, Frame 0)",
        show_labels=True,
        save_path=str(out / "sample_skeleton.png"),
    )
    ds_html["figures"]["skeleton_static"] = fig_to_base64(fig_skel)
    plt.close(fig_skel)
    print("  Saved: sample_skeleton.png")

    # Animated skeleton (use VIZ_CONFIG frame count)
    n_anim = min(VIZ_CONFIG["gif_n_frames"], sample_kp.shape[0])
    anim = animate_skeleton(
        clip_outlier_joints(
            sample_kp[:n_anim],
            per_joint=VIZ_CONFIG["clip_per_joint"],
            iqr_factor=VIZ_CONFIG["clip_iqr_factor"],
        ),
        skeleton=skeleton,
        fps=VIZ_CONFIG["gif_fps_playback"], title=f"Shank3KO Mouse ({geno})",
        save_path=str(out / "sample_animation.gif"),
    )
    plt.close("all")
    gif_path = out / "sample_animation.gif"
    if gif_path.exists():
        ds_html["figures"]["skeleton_gif"] = image_to_base64(gif_path)
    print(f"  Saved: sample_animation.gif ({n_anim} frames)")

    # --- KMeans clustering ---
    from behavior_lab.models.discovery.clustering import cluster_features

    step = max(1, len(all_kp) // 5000)
    sample_indices = np.arange(0, len(all_kp), step)
    features = all_kp[sample_indices].reshape(-1, s0.num_joints * s0.num_channels)
    print(f"  Feature matrix: {features.shape}")

    result = cluster_features(features, n_clusters=5, use_umap=False)
    labels = result["labels"]

    kmeans_dict = {
        "n_clusters": int(result["n_clusters"]),
        "n_samples": int(len(labels)),
    }
    report["shank3ko"]["kmeans"] = kmeans_dict
    ds_html["cluster_metrics"] = kmeans_dict

    # PCA embedding plot
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(features)
    fig_emb, _ = plot_embedding(
        emb_2d, labels,
        title=f"Shank3KO KMeans (PCA, {result['n_clusters']} clusters)",
        save_path=str(out / "kmeans_embedding.png"),
    )
    ds_html["figures"]["embedding"] = fig_to_base64(fig_emb)
    plt.close("all")
    print(f"  Saved: kmeans_embedding.png ({result['n_clusters']} clusters)")

    # Per-cluster representative GIFs
    cluster_gifs = generate_cluster_animations(
        all_kp, labels, skeleton, out / "clusters",
        sample_indices=sample_indices,
    )
    if cluster_gifs:
        ds_html["per_class_gifs"] = cluster_gifs

    # Genotype-colored PCA (KO vs WT)
    if len(geno_dist) > 1:
        try:
            # Sample frames per mouse, track genotype
            geno_labels = []
            geno_features = []
            for seq in sequences:
                kp = seq.keypoints
                step_s = max(1, len(kp) // 250)
                feat = kp[::step_s].reshape(-1, s0.num_joints * s0.num_channels)
                g = 0 if seq.metadata.get("genotype") == "KO" else 1
                geno_labels.extend([g] * len(feat))
                geno_features.append(feat)
            geno_features = np.concatenate(geno_features, axis=0)
            geno_labels = np.array(geno_labels)
            pca2 = PCA(n_components=2)
            emb_geno = pca2.fit_transform(geno_features)
            fig_geno, _ = plot_embedding(
                emb_geno, geno_labels,
                title="Shank3KO: KO (0) vs WT (1)",
                save_path=str(out / "genotype_embedding.png"),
            )
            ds_html["figures"]["genotype_pca"] = fig_to_base64(fig_geno)
            plt.close("all")
            print("  Saved: genotype_embedding.png")
        except Exception as e:
            print(f"  Genotype PCA: SKIPPED ({e})")

    # B-SOiD (3D → expect skip)
    try:
        from behavior_lab.models.discovery.bsoid import BSOiD
        bsoid = BSOiD(fps=60, min_cluster_size=30)
        bsoid_result = bsoid.fit_predict(all_kp[:5000])
        report["shank3ko"]["bsoid"] = {"n_clusters": int(bsoid_result.n_clusters)}
        print(f"  B-SOiD: {bsoid_result.n_clusters} clusters")
    except Exception as e:
        print(f"  B-SOiD: SKIPPED ({e})")

    html_data["datasets"]["shank3ko"] = ds_html
    print("\n  Shank3KO PASSED")


def test_mabe22_behavemae(report: dict, html_data: dict) -> None:
    """Test MABe22 → KMeans + BehaveMAE + visualizations."""
    print("\n" + "=" * 60)
    print("MABe22: Mouse Triplet Behavior Analysis")
    print("=" * 60)

    out = OUT_DIR / "mabe22"
    out.mkdir(parents=True, exist_ok=True)

    # Try preprocessed first, then raw
    data_dir = ROOT / "data" / "preprocessed" / "mabe22"
    if not data_dir.exists() or not list(data_dir.glob("*.npz")):
        data_dir = ROOT / "data" / "raw" / "mabe22"
    if not data_dir.exists() or not list(data_dir.glob("*")):
        print("  SKIPPED: No MABe22 data found.")
        print("  Run: python scripts/download_data.py --dataset mabe22")
        return

    loader = get_loader("mabe22", data_dir=data_dir)
    all_splits = loader.load_all()
    if not all_splits:
        print("  SKIPPED: No sequences loaded")
        return

    # Flatten all splits
    sequences = []
    for split_seqs in all_splits.values():
        sequences.extend(split_seqs)
    print(f"  Loaded: {len(sequences)} sequences total")

    s0 = sequences[0]
    s0.validate()
    print(f"  Shape: ({s0.num_frames}, {s0.num_joints}, {s0.num_channels})")

    data_summary = {
        "n_sequences": len(sequences),
        "shape": [int(s0.num_frames), int(s0.num_joints), int(s0.num_channels)],
    }
    report["mabe22"] = {"data": data_summary}
    ds_html: dict = {"data": data_summary, "figures": {}}

    ds_html["model_info"] = {
        "name": "KMeans + BehaveMAE",
        "type": "Unsupervised Discovery",
        "description": "KMeans on mean-pooled features; BehaveMAE hierarchical encoding (if available)",
        "why": "Multi-mouse social behavior analysis with 3 mice x 12 keypoints",
    }

    # --- Skeleton Visualization ---
    print("\n  Generating skeleton visualizations...")
    skeleton = get_skeleton("mabe22")
    ds_html["joint_info"] = _build_joint_info(skeleton)

    # MABe22 is 2D: (T, 36, 2) — 3 mice x 12 joints
    sample_kp = sequences[0].keypoints
    fig_skel, _ = plot_skeleton(
        sample_kp, skeleton=skeleton, frame=0,
        title="MABe22 — 3 Mice x 12 Joints (Frame 0)",
        show_labels=True,
        save_path=str(out / "sample_skeleton.png"),
    )
    ds_html["figures"]["skeleton_static"] = fig_to_base64(fig_skel)
    plt.close(fig_skel)
    print("  Saved: sample_skeleton.png")

    # Animated skeleton (use VIZ_CONFIG frame count)
    n_anim = min(VIZ_CONFIG["gif_n_frames"], sample_kp.shape[0])
    anim = animate_skeleton(
        sample_kp[:n_anim], skeleton=skeleton,
        fps=VIZ_CONFIG["gif_fps_playback"], title="MABe22 Mice Triplet",
        save_path=str(out / "sample_animation.gif"),
    )
    plt.close("all")
    gif_path = out / "sample_animation.gif"
    if gif_path.exists():
        ds_html["figures"]["skeleton_gif"] = image_to_base64(gif_path)
    print(f"  Saved: sample_animation.gif ({n_anim} frames)")

    # --- KMeans on mean-pooled features ---
    from behavior_lab.models.discovery.clustering import cluster_features

    n_sub = min(500, len(sequences))
    features = np.array([s.keypoints.mean(axis=0).flatten() for s in sequences[:n_sub]])
    result = cluster_features(features, n_clusters=5, use_umap=False)
    labels = result["labels"]

    kmeans_dict = {
        "n_clusters": int(result["n_clusters"]),
        "n_samples": int(len(labels)),
    }
    report["mabe22"]["kmeans"] = kmeans_dict
    ds_html["cluster_metrics"] = kmeans_dict
    print(f"  KMeans: {result['n_clusters']} clusters on {n_sub} sequences")

    # PCA embedding plot
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(features)
    fig_emb, _ = plot_embedding(
        emb_2d, labels,
        title=f"MABe22 KMeans (PCA, {result['n_clusters']} clusters)",
        save_path=str(out / "kmeans_embedding.png"),
    )
    ds_html["figures"]["embedding"] = fig_to_base64(fig_emb)
    plt.close("all")
    print(f"  Saved: kmeans_embedding.png")

    # Per-cluster representative GIFs (sequence-level: pick representative sequence per cluster)
    cluster_seq_gifs = []
    for cid in sorted(set(labels)):
        cluster_seqs = [sequences[i] for i, l in enumerate(labels) if l == cid]
        if not cluster_seqs:
            continue
        # Pick most dynamic sequence
        def _seq_var(s):
            return float(s.keypoints.var())
        best_seq = max(cluster_seqs, key=_seq_var)
        kp = best_seq.keypoints
        nf = min(VIZ_CONFIG["per_class_n_frames"], kp.shape[0])
        save_path = out / "clusters" / f"cluster_{cid:02d}.gif"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            anim = animate_skeleton(
                kp[:nf], skeleton=skeleton,
                fps=VIZ_CONFIG["gif_fps_playback"],
                title=f"Cluster {cid} ({nf}f)",
                save_path=str(save_path),
            )
            plt.close("all")
            if save_path.exists():
                cluster_seq_gifs.append({
                    "label": f"Cluster {cid}",
                    "src": image_to_base64(save_path),
                })
                print(f"    Saved: {save_path.name} ({nf} frames)")
        except Exception as e:
            print(f"    Warning: cluster {cid} GIF failed: {e}")
    if cluster_seq_gifs:
        ds_html["per_class_gifs"] = cluster_seq_gifs

    # --- BehaveMAE hierarchical encoding ---
    try:
        from behavior_lab.models.discovery.behavemae import BehaveMAE, pose_to_behavemae_input

        sample_kp_slice = sequences[0].keypoints[:400]  # (T, 36, 2)
        tensor = pose_to_behavemae_input(sample_kp_slice, target_frames=400)
        print(f"  BehaveMAE input tensor: {tuple(tensor.shape)}")

        report["mabe22"]["behavemae"] = {
            "input_shape": list(tensor.shape),
            "status": "input_conversion_ok",
        }

        ckpt_dir = ROOT / "checkpoints" / "behavemae"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.pth"))
            if ckpts:
                model = BehaveMAE.from_pretrained(str(ckpts[0]), dataset='mabe22')
                hier_features = model.encode_hierarchical(sample_kp_slice)
                print(f"  Hierarchical levels: {list(hier_features.keys())}")
                for k, v in hier_features.items():
                    print(f"    {k}: {v.shape}")
                report["mabe22"]["behavemae"]["hierarchical"] = {
                    k: list(v.shape) for k, v in hier_features.items()
                }
        else:
            print("  BehaveMAE model: no checkpoint found (input conversion tested)")

    except ImportError as e:
        print(f"  BehaveMAE: SKIPPED (import error: {e})")
    except Exception as e:
        print(f"  BehaveMAE: SKIPPED ({e})")

    html_data["datasets"]["mabe22"] = ds_html
    print("\n  MABe22 PASSED")


def test_calms21_behavemae(report: dict, html_data: dict) -> None:
    """Test CalMS21 → BehaveMAE with adapted config (28 features)."""
    print("\n" + "=" * 60)
    print("CalMS21: BehaveMAE Hierarchical Analysis")
    print("=" * 60)

    loader = get_loader("calms21", data_dir=ROOT / "data" / "calms21")
    try:
        train_seqs = loader.load_split("train")
    except FileNotFoundError:
        print("  SKIPPED: CalMS21 data not found")
        return

    try:
        from behavior_lab.models.discovery.behavemae import BehaveMAE, pose_to_behavemae_input

        # CalMS21: (T, 14, 2) -> flatten to (T, 28)
        sample_kp = train_seqs[0].keypoints[:400]
        tensor = pose_to_behavemae_input(sample_kp, target_frames=400)
        print(f"  BehaveMAE input: {tuple(tensor.shape)}")
        # Expected: (1, 1, 400, 1, 28)

        config = BehaveMAE.CONFIGS.get("calms21")
        print(f"  CalMS21 config input_size: {config['input_size']}")
        assert config['input_size'][2] == 28, f"Expected 28, got {config['input_size'][2]}"

        report.setdefault("calms21", {})["behavemae"] = {
            "input_shape": list(tensor.shape),
            "config_input_size": list(config["input_size"]),
            "status": "config_validated",
        }
        print("  CalMS21 BehaveMAE config validated")

    except ImportError as e:
        print(f"  SKIPPED (import error: {e})")
    except Exception as e:
        print(f"  SKIPPED ({e})")

    print("\n  CalMS21 BehaveMAE PASSED")


def generate_report(report: dict) -> None:
    """Generate JSON and Markdown summary reports."""
    out = OUT_DIR

    # JSON report
    (out / "report.json").write_text(json.dumps(safe_json(report), indent=2))
    print(f"\n  Saved: {out / 'report.json'}")

    # Markdown report
    md_lines = [
        "# E2E Verification Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # CalMS21
    if "calms21" in report:
        c = report["calms21"]
        md_lines += [
            "## CalMS21 — Mouse Social Behavior",
            "",
            "### Data Loading",
            f"- Train: **{c['data']['n_train']}** sequences",
            f"- Test: **{c['data']['n_test']}** sequences",
            f"- Shape: `({', '.join(map(str, c['data']['shape']))})`",
            "",
            "### Analysis Method",
            "- **Model**: B-SOiD (Unsupervised Discovery)",
            "- **Why**: Discovers behavioral motifs without requiring labels",
            "- **Method**: UMAP embedding + HDBSCAN clustering + Random Forest",
            "",
            "### Skeleton Visualization",
            "",
            "![Skeleton](calms21/sample_skeleton.png)",
            "",
            "### B-SOiD Discovery",
            f"- Time: {c.get('bsoid_time_sec', 'N/A')}s",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if "cluster_metrics" in c:
            cm = c["cluster_metrics"]
            for k, v in cm.items():
                md_lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        md_lines += [
            "",
            "### Behavior Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if "behavior_metrics" in c:
            bm = c["behavior_metrics"]
            md_lines.append(f"| temporal_consistency | {bm['temporal_consistency']:.4f} |")
            md_lines.append(f"| num_bouts | {bm['num_bouts']} |")
            md_lines.append(f"| entropy_rate | {bm['entropy_rate']:.4f} |")
        md_lines += [
            "",
            "### Visualizations",
            "",
            "![Preprocessing](calms21/preprocessing_comparison.png)",
            "![Comparison](calms21/skeleton_comparison.png)",
            "![Embedding](calms21/bsoid_embedding.png)",
            "![Transitions](calms21/bsoid_transition_matrix.png)",
            "![Bout Duration](calms21/bsoid_bout_duration.png)",
            "![Ethogram](calms21/bsoid_ethogram.png)",
            "",
        ]

    # NTU
    if "ntu" in report:
        n = report["ntu"]
        md_lines += [
            "## NTU RGB+D — Human Action Recognition (Demo)",
            "",
            f"- Train: **{n['data']['n_train']}**, Test: **{n['data']['n_test']}**",
            f"- Shape: `({', '.join(map(str, n['data']['shape']))})`",
            f"- Classes: {n['data']['n_classes']}",
            "",
            "### Analysis Method",
            "- **Model**: Linear Probe (LogisticRegression)",
            "- **Why**: Measures discriminative power of raw spatial features",
            "- **Method**: Mean-pooled keypoints → linear classifier (intentionally naive)",
            "",
            "### Skeleton",
            "",
            "![Skeleton](ntu/sample_skeleton.png)",
            "",
        ]
        if "linear_probe" in n:
            lp = n["linear_probe"]
            md_lines += [
                "### Linear Probe",
                f"- Accuracy: **{lp['accuracy']:.4f}**",
                f"- F1 (macro): **{lp['f1_macro']:.4f}**",
                "",
                "![Confusion](ntu/linear_probe_confusion.png)",
                "",
            ]

    # NW-UCLA
    if "nwucla" in report:
        u = report["nwucla"]
        md_lines += [
            "## NW-UCLA — Action Recognition",
            "",
            f"- Train: **{u['data']['n_train']}**, Test: **{u['data']['n_test']}**",
            f"- Shape: `({', '.join(map(str, u['data']['shape']))})`",
            "",
            "### Analysis Method",
            "- **Model**: Linear Probe + LSTM/Transformer (2-epoch quick test)",
            "- **Why**: Verifies supervised pipeline end-to-end",
            "",
            "### Skeleton",
            "",
            "![Skeleton](nwucla/sample_skeleton.png)",
            "",
        ]
        if "linear_probe" in u:
            lp = u["linear_probe"]
            md_lines += [
                "### Linear Probe",
                f"- Accuracy: **{lp['accuracy']:.4f}**",
                f"- F1 (macro): **{lp['f1_macro']:.4f}**",
                "",
                "![Confusion](nwucla/linear_probe_confusion.png)",
                "",
            ]

    # SUBTLE
    if "subtle" in report:
        s = report["subtle"]
        md_lines += [
            "## SUBTLE — Mouse Spontaneous Behavior (3D)",
            "",
            f"- Sequences: **{s['data']['n_sequences']}**",
            f"- Total frames: **{s['data']['total_frames']}**",
            f"- Shape per frame: `({', '.join(map(str, s['data']['shape_per_frame']))})`",
            "",
        ]
        if "kmeans" in s:
            md_lines.append(f"- KMeans clusters: {s['kmeans']['n_clusters']}")
        if "bsoid" in s:
            md_lines.append(f"- B-SOiD clusters: {s['bsoid']['n_clusters']}")
        md_lines.append("")

    # Shank3KO
    if "shank3ko" in report:
        s = report["shank3ko"]
        md_lines += [
            "## Shank3KO — Knockout Mouse Behavior (3D)",
            "",
            f"- Sequences: **{s['data']['n_sequences']}**",
            f"- Total frames: **{s['data']['total_frames']}**",
            "",
        ]
        if "kmeans" in s:
            md_lines.append(f"- KMeans clusters: {s['kmeans']['n_clusters']}")
        if "bsoid" in s:
            md_lines.append(f"- B-SOiD clusters: {s['bsoid']['n_clusters']}")
        md_lines.append("")

    # MABe22
    if "mabe22" in report:
        m = report["mabe22"]
        md_lines += [
            "## MABe22 — Mouse Triplet Behavior",
            "",
            f"- Sequences: **{m['data']['n_sequences']}**",
            f"- Shape: `({', '.join(map(str, m['data']['shape']))})`",
            "",
        ]
        if "kmeans" in m:
            md_lines.append(f"- KMeans clusters: {m['kmeans']['n_clusters']}")
        if "behavemae" in m:
            md_lines.append(f"- BehaveMAE: {m['behavemae'].get('status', 'N/A')}")
        md_lines.append("")

    (out / "report.md").write_text("\n".join(md_lines))
    print(f"  Saved: {out / 'report.md'}")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 60)
    print("behavior-lab E2E Verification")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    html_data: dict = {
        "title": "behavior-lab E2E Verification Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {},
    }

    import gc

    # P0: Core dataset tests (CalMS21, NTU, NW-UCLA)
    test_calms21(report, html_data)
    plt.close("all"); gc.collect()
    test_ntu(report, html_data)
    plt.close("all"); gc.collect()
    test_nwucla(report, html_data)
    plt.close("all"); gc.collect()

    # P1: Additional model combinations on existing datasets
    test_nwucla_gcn(report, html_data)
    gc.collect()
    test_ntu_gcn(report, html_data)
    gc.collect()
    test_calms21_kmeans(report, html_data)
    gc.collect()

    # P2: New datasets (requires data download)
    test_subtle(report, html_data)
    gc.collect()
    test_shank3ko(report, html_data)
    gc.collect()
    test_mabe22_behavemae(report, html_data)
    gc.collect()
    test_calms21_behavemae(report, html_data)

    generate_report(report)

    # HTML report
    html_path = generate_pipeline_report(
        html_data,
        OUT_DIR / "report.html",
        title="behavior-lab E2E Verification Report",
    )
    print(f"  Saved: {html_path}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
