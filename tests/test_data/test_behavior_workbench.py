import numpy as np

from behavior_lab.data.features import list_discovery_methods, list_feature_blocks, list_pose_sources
from behavior_lab.experiments import compare_discovery_methods, extract_feature_matrix


def test_catalog_has_requested_modules():
    poses = {spec.name for spec in list_pose_sources()}
    features = {spec.name for spec in list_feature_blocks()}
    methods = {spec.name for spec in list_discovery_methods()}

    assert {"DeepLabCut", "SLEAP", "Anipose/Triangulation"} <= poses
    assert {"raw_keypoints", "skeleton_kinematic", "bsoid_spatiotemporal"} <= features
    assert {"B-SOiD", "keypoint-moseq", "SUBTLE", "hBehaveMAE"} <= methods


def test_lightweight_discovery_comparison_runs():
    rng = np.random.default_rng(0)
    keypoints = rng.normal(size=(60, 5, 2)).astype(np.float32)

    features = extract_feature_matrix(keypoints, feature="skeleton_kinematic", fps=30)
    runs = compare_discovery_methods(
        keypoints,
        methods=("kmeans_pca_umap",),
        feature="skeleton_kinematic",
        n_clusters=3,
        max_frames=60,
    )

    assert features.shape == (60, 4)
    assert len(runs) == 1
    assert runs[0].result.labels.shape == (60,)
    assert runs[0].result.n_clusters == 3
