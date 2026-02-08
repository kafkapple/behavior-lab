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
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

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
    plot_temporal_raster,
)

OUT_DIR = ROOT / "outputs" / "e2e_test"


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


def test_calms21(report: dict) -> None:
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

    # --- Preprocessing ---
    print("\n  Preprocessing pipeline...")
    pipeline = PreprocessingPipeline([
        Interpolator(max_gap=10),
        OutlierRemover(velocity_threshold=50.0),
        TemporalSmoother(window_size=5),
        Normalizer(center_joint=0),
    ])
    sample_kp = train_seqs[0].keypoints.copy()
    cleaned_kp = pipeline(sample_kp)
    print(f"  Before: range [{sample_kp.min():.2f}, {sample_kp.max():.2f}], NaN={np.isnan(sample_kp).sum()}")
    print(f"  After:  range [{cleaned_kp.min():.4f}, {cleaned_kp.max():.4f}], NaN={np.isnan(cleaned_kp).sum()}")

    # Preprocessing comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    joint_idx = 0
    axes[0].plot(sample_kp[:, joint_idx, 0], label="x", alpha=0.7)
    axes[0].plot(sample_kp[:, joint_idx, 1], label="y", alpha=0.7)
    axes[0].set_title("Before Preprocessing")
    axes[0].legend()
    axes[1].plot(cleaned_kp[:, joint_idx, 0], label="x", alpha=0.7)
    axes[1].plot(cleaned_kp[:, joint_idx, 1], label="y", alpha=0.7)
    axes[1].set_title("After Preprocessing")
    axes[1].legend()
    axes[1].set_xlabel("Frame")
    fig.suptitle("CalMS21: Preprocessing Comparison (Joint 0)")
    fig.tight_layout()
    fig.savefig(out / "preprocessing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: preprocessing_comparison.png")

    # --- B-SOiD Discovery ---
    print("\n  B-SOiD discovery (1000 samples)...")
    from behavior_lab.models.discovery.bsoid import BSOiD

    n_subset = min(1000, len(train_seqs))
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
    # B-SOiD bins features to 10fps (bin_size = fps//10 = 3), then drops last partial bin.
    # Also _compute_bsoid_features uses diff (T-1). Align GT labels to match.
    bin_size = max(1, 30 // 10)  # =3
    gt_trimmed = all_gt[:-1]  # diff removes 1 frame
    n_bins = len(gt_trimmed) // bin_size
    gt_binned = gt_trimmed[:n_bins * bin_size].reshape(n_bins, bin_size)
    # Use majority vote per bin
    from scipy.stats import mode as scipy_mode
    gt_binned_labels = scipy_mode(gt_binned, axis=1, keepdims=False).mode.flatten()
    # Trim to match B-SOiD output length
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

    # --- Visualization ---
    print("\n  Generating visualizations...")

    # Embedding plot
    plot_embedding(
        result.embeddings, result.labels,
        title="B-SOiD Embedding (Cluster Colors)",
        save_path=str(out / "bsoid_embedding.png"),
    )
    plt.close("all")
    print("  Saved: bsoid_embedding.png")

    # Transition matrix
    if beh_metrics.transition_matrix is not None:
        plot_transition_matrix(
            beh_metrics.transition_matrix,
            title="B-SOiD Behavior Transitions",
            save_path=str(out / "bsoid_transition_matrix.png"),
        )
        plt.close("all")
        print("  Saved: bsoid_transition_matrix.png")

    # Bout duration
    plot_bout_duration(
        beh_metrics.bout_durations,
        title="B-SOiD Mean Bout Durations",
        save_path=str(out / "bsoid_bout_duration.png"),
    )
    plt.close("all")
    print("  Saved: bsoid_bout_duration.png")

    # Ethogram (first 5 sequences)
    sample_labels = result.labels[:5000]
    plot_temporal_raster(
        sample_labels, fps=10.0,
        title="B-SOiD Ethogram (First 5000 frames)",
        save_path=str(out / "bsoid_ethogram.png"),
    )
    plt.close("all")
    print("  Saved: bsoid_ethogram.png")

    print("\n  CalMS21 PASSED")


def test_ntu(report: dict) -> None:
    """Test NTU RGB+D demo data loading and linear probe."""
    print("\n" + "=" * 60)
    print("NTU RGB+D: Human Action Recognition (Demo)")
    print("=" * 60)

    out = OUT_DIR / "ntu"
    out.mkdir(parents=True, exist_ok=True)

    npz_path = ROOT / "data" / "ntu" / "demo_CS_aligned.npz"
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

    # Confusion matrix plot
    if probe_result.confusion is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(probe_result.confusion, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"NTU Linear Probe Confusion (Acc={probe_result.accuracy:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.savefig(out / "linear_probe_confusion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: linear_probe_confusion.png")

    print("\n  NTU PASSED")


def test_nwucla(report: dict) -> None:
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

    # Confusion matrix plot
    from behavior_lab.data.loaders.nwucla import UCLA_CLASSES
    if probe_result.confusion is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(probe_result.confusion, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(UCLA_CLASSES[:n_classes], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(UCLA_CLASSES[:n_classes], fontsize=7)
        ax.set_title(f"NW-UCLA Linear Probe (Acc={probe_result.accuracy:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.savefig(out / "linear_probe_confusion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: linear_probe_confusion.png")

    print("\n  NW-UCLA PASSED")


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

    test_calms21(report)
    test_ntu(report)
    test_nwucla(report)
    generate_report(report)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
