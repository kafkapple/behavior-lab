"""E2E integration test for feature backends and pipeline.

Tests:
1. SkeletonBackend: synthetic keypoints → (N, 4) → cluster_features
2. DINOv2Backend: synthetic frames → (N, 384) → cluster (requires torch)
3. FeaturePipeline: skeleton + dinov2 concat → cluster
4. Temporal aggregation: (T, D) → (N_seg, D)
5. Metrics: silhouette score, cluster count

Usage:
    python scripts/test_feature_backends.py           # skeleton only (no torch)
    python scripts/test_feature_backends.py --visual   # include DINOv2 tests
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _make_synthetic_keypoints(n_frames: int = 300, n_joints: int = 7,
                              n_dims: int = 2) -> np.ndarray:
    """Generate synthetic keypoints with 3 distinct movement patterns."""
    rng = np.random.default_rng(42)
    kp = np.zeros((n_frames, n_joints, n_dims))
    seg = n_frames // 3

    # Segment 1: stationary (small jitter)
    kp[:seg] = rng.normal(0, 0.1, (seg, n_joints, n_dims))

    # Segment 2: fast linear motion
    for t in range(seg, 2 * seg):
        kp[t] = kp[seg - 1] + (t - seg) * 0.5 + rng.normal(0, 0.05, (n_joints, n_dims))

    # Segment 3: circular motion
    for t in range(2 * seg, n_frames):
        angle = (t - 2 * seg) * 0.1
        center = np.array([np.cos(angle), np.sin(angle)]) * 3
        kp[t] = center + rng.normal(0, 0.2, (n_joints, n_dims))

    return kp


def test_skeleton_backend():
    """Test SkeletonBackend: keypoints → (T, 4) features."""
    from behavior_lab.data.features import SkeletonBackend

    print("=== Test 1: SkeletonBackend ===")
    backend = SkeletonBackend(fps=30.0)
    kp = _make_synthetic_keypoints()

    features = backend.extract(kp)
    assert features.shape == (300, 4), f"Expected (300, 4), got {features.shape}"
    assert not np.any(np.isnan(features)), "NaN in features"
    print(f"  Input:  {kp.shape}")
    print(f"  Output: {features.shape}")
    print(f"  Stats:  mean={features.mean(0)}, std={features.std(0)}")

    # Verify protocol compliance
    from behavior_lab.data.features import FeatureBackend
    assert isinstance(backend, FeatureBackend), "SkeletonBackend does not satisfy Protocol"
    print("  Protocol: OK")
    print("  PASSED\n")
    return features


def test_skeleton_clustering(features: np.ndarray):
    """Test cluster_features with skeleton features."""
    from behavior_lab.models.discovery.clustering import cluster_features

    print("=== Test 2: Skeleton → Clustering ===")
    result = cluster_features(features, n_clusters=3, use_umap=False)
    labels = result["labels"]
    n_clusters = result["n_clusters"]

    assert labels.shape == (300,), f"Expected (300,), got {labels.shape}"
    assert n_clusters == 3
    unique = np.unique(labels)
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {unique}")

    # Silhouette check
    try:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(features, labels)
        print(f"  Silhouette: {sil:.3f}")
        assert sil > -1.0, "Silhouette out of range"
    except ImportError:
        print("  Silhouette: skipped (sklearn not available)")

    print("  PASSED\n")


def test_temporal_aggregation():
    """Test temporal aggregation methods."""
    from behavior_lab.data.features.temporal import aggregate_temporal

    print("=== Test 3: Temporal Aggregation ===")
    rng = np.random.default_rng(42)
    features = rng.standard_normal((300, 8))  # (T=300, D=8)

    # mean
    seg_mean = aggregate_temporal(features, window_size=30, stride=15, method="mean")
    expected_n = (300 - 30) // 15 + 1  # 19
    assert seg_mean.shape == (expected_n, 8), f"mean: expected ({expected_n}, 8), got {seg_mean.shape}"
    print(f"  mean:         {features.shape} → {seg_mean.shape}")

    # max
    seg_max = aggregate_temporal(features, window_size=30, stride=15, method="max")
    assert seg_max.shape == (expected_n, 8)
    print(f"  max:          {features.shape} → {seg_max.shape}")

    # concat_stats (mean+std+min+max → 4x dim)
    seg_stats = aggregate_temporal(features, window_size=30, stride=15, method="concat_stats")
    assert seg_stats.shape == (expected_n, 32), f"concat_stats: expected ({expected_n}, 32), got {seg_stats.shape}"
    print(f"  concat_stats: {features.shape} → {seg_stats.shape}")

    # Edge case: T < window_size
    short = rng.standard_normal((10, 8))
    seg_short = aggregate_temporal(short, window_size=30, stride=15, method="mean")
    assert seg_short.shape == (1, 8), f"Short: expected (1, 8), got {seg_short.shape}"
    print(f"  short input:  {short.shape} → {seg_short.shape}")
    print("  PASSED\n")


def test_pipeline_skeleton_only():
    """Test FeaturePipeline with skeleton backend only."""
    from behavior_lab.data.features import FeaturePipeline, SkeletonBackend

    print("=== Test 4: FeaturePipeline (skeleton only) ===")
    pipe = FeaturePipeline(
        backends=[SkeletonBackend()],
        temporal_agg={"window_size": 30, "stride": 15, "method": "mean"},
    )

    kp = _make_synthetic_keypoints()
    features = pipe.extract(keypoints=kp)
    expected_n = (300 - 30) // 15 + 1
    assert features.shape == (expected_n, 4), f"Expected ({expected_n}, 4), got {features.shape}"
    print(f"  dim property: {pipe.dim}")
    print(f"  Output: {features.shape}")
    print("  PASSED\n")
    return features


def test_pipeline_clustering(features: np.ndarray):
    """Test cluster_features with pipeline output."""
    from behavior_lab.models.discovery.clustering import cluster_features

    print("=== Test 5: Pipeline → Clustering ===")
    result = cluster_features(features, n_clusters=3, use_umap=False)
    labels = result["labels"]
    assert labels.shape[0] == features.shape[0]
    print(f"  Input: {features.shape} → labels: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    print("  PASSED\n")


# ---- Visual tests (require torch) ----

def test_dinov2_backend():
    """Test DINOv2Backend with synthetic frames."""
    from behavior_lab.data.features.visual_backend import DINOv2Backend

    print("=== Test 6: DINOv2Backend ===")
    backend = DINOv2Backend(model_name="dinov2_vits14", device="cpu",
                            pool="cls", batch_size=16)

    rng = np.random.default_rng(42)
    frames = rng.integers(0, 255, (16, 224, 224, 3), dtype=np.uint8)

    features = backend.extract(frames)
    assert features.shape == (16, 384), f"Expected (16, 384), got {features.shape}"
    assert not np.any(np.isnan(features)), "NaN in DINOv2 features"
    print(f"  Input:  {frames.shape}")
    print(f"  Output: {features.shape}")
    print(f"  dim:    {backend.dim}")

    # Test mean pooling
    backend_mean = DINOv2Backend(model_name="dinov2_vits14", device="cpu",
                                 pool="mean", batch_size=16)
    features_mean = backend_mean.extract(frames)
    assert features_mean.shape == (16, 384)
    print(f"  mean pool: {features_mean.shape}")

    # Test concat pooling
    backend_concat = DINOv2Backend(model_name="dinov2_vits14", device="cpu",
                                   pool="concat", batch_size=16)
    features_concat = backend_concat.extract(frames)
    assert features_concat.shape == (16, 768), f"Expected (16, 768), got {features_concat.shape}"
    print(f"  concat pool: {features_concat.shape}")

    print("  PASSED\n")
    return features


def test_pipeline_multimodal():
    """Test FeaturePipeline with skeleton + dinov2."""
    from behavior_lab.data.features import FeaturePipeline, SkeletonBackend
    from behavior_lab.data.features.visual_backend import DINOv2Backend

    print("=== Test 7: Multi-modal Pipeline (skeleton + dinov2) ===")
    pipe = FeaturePipeline(
        backends=[SkeletonBackend(), DINOv2Backend(device="cpu", pool="cls")],
        temporal_agg={"window_size": 10, "stride": 5, "method": "mean"},
    )

    n_frames = 50
    rng = np.random.default_rng(42)
    kp = _make_synthetic_keypoints(n_frames=n_frames)
    frames = rng.integers(0, 255, (n_frames, 224, 224, 3), dtype=np.uint8)

    features = pipe.extract(keypoints=kp, frames=frames)
    expected_n = (n_frames - 10) // 5 + 1
    expected_d = 4 + 384  # skeleton + dinov2
    assert features.shape == (expected_n, expected_d), \
        f"Expected ({expected_n}, {expected_d}), got {features.shape}"
    print(f"  dim property: {pipe.dim}")
    print(f"  Output: {features.shape}")

    # Cluster
    from behavior_lab.models.discovery.clustering import cluster_features
    result = cluster_features(features, n_clusters=3, use_umap=False)
    print(f"  Clustering: {result['labels'].shape}, labels={np.unique(result['labels'])}")
    print("  PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="Test feature backends")
    parser.add_argument("--visual", action="store_true",
                        help="Include DINOv2/visual tests (requires torch)")
    args = parser.parse_args()

    passed = 0
    failed = 0

    tests_core = [
        ("SkeletonBackend", lambda: test_skeleton_backend()),
        ("Skeleton clustering", lambda: test_skeleton_clustering(
            test_skeleton_backend())),
        ("Temporal aggregation", lambda: test_temporal_aggregation()),
        ("Pipeline (skeleton)", lambda: test_pipeline_skeleton_only()),
        ("Pipeline clustering", lambda: test_pipeline_clustering(
            test_pipeline_skeleton_only())),
    ]

    tests_visual = [
        ("DINOv2Backend", lambda: test_dinov2_backend()),
        ("Multi-modal pipeline", lambda: test_pipeline_multimodal()),
    ]

    tests = tests_core + (tests_visual if args.visual else [])

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")

    if not args.visual:
        print("(Run with --visual to include DINOv2 tests)")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
