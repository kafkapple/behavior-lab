"""Catalog of pose sources, feature blocks, and discovery methods.

This is the single index used by docs and notebooks to avoid drifting lists.
It separates:

1. where keypoints come from,
2. how keypoints become feature matrices,
3. how features or raw keypoints become unsupervised behavior labels.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    category: str
    input_format: str
    output_format: str
    module_path: str
    dependencies: tuple[str, ...]
    strengths: tuple[str, ...]
    caveats: tuple[str, ...]


POSE_SOURCES: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        name="DeepLabCut",
        category="pose_source",
        input_format="video or DLC h5/csv",
        output_format="(T,K,2/3) keypoints + likelihood",
        module_path="scripts/*dlc*.sh; outputs/kp_benchmark/*.npz",
        dependencies=("deeplabcut",),
        strengths=("strong supervised/SuperAnimal ecosystem", "good 2D animal keypoint tooling"),
        caveats=("project-specific bodypart order must be mapped to skeleton registry",),
    ),
    ModuleSpec(
        name="SLEAP",
        category="pose_source",
        input_format=".slp or SLEAP analysis .h5",
        output_format="BehaviorSequence (T,K,D), flattened or per-track",
        module_path="behavior_lab.pose.sleap; behavior_lab.data.loaders.sleap",
        dependencies=("sleap-io", "h5py"),
        strengths=("multi-animal tracking metadata", "confidence-aware NaN masking"),
        caveats=("3D requires external triangulation or calibrated multi-view export",),
    ),
    ModuleSpec(
        name="Anipose/Triangulation",
        category="pose_source",
        input_format="multi-view 2D keypoints + camera calibration",
        output_format="(T,K,3) triangulated sparse keypoints",
        module_path="scripts/anipose_triangulate.py",
        dependencies=("opencv-python",),
        strengths=("3D reconstruction from DLC/SLEAP 2D tracks", "RANSAC/linear comparison possible"),
        caveats=("accuracy dominated by calibration, sync, and 2D confidence filtering"),
    ),
    ModuleSpec(
        name="Canonical NPZ loaders",
        category="pose_source",
        input_format="CalMS21, MABe22, Rat7M, NTU, NW-UCLA, SUBTLE, Shank3KO npz/npy/mat/csv",
        output_format="BehaviorSequence list with (T,K,D)",
        module_path="behavior_lab.data.loaders",
        dependencies=("numpy", "scipy", "h5py"),
        strengths=("current local datasets use one common format", "works without pose-estimator installs"),
        caveats=("dataset coordinate frames and units differ; normalize before cross-dataset comparison"),
    ),
    ModuleSpec(
        name="Unified ingestion",
        category="pose_source",
        input_format="any keypoint file — DeepLabCut CSV/H5, SLEAP .slp/.h5, npz (auto-detect)",
        output_format="list[BehaviorSequence] + standard metadata['provenance'] (+ optional manifest.jsonl)",
        module_path="behavior_lab.data.ingestion.ingest",
        dependencies=("numpy", "pandas", "h5py"),
        strengths=("one consistent standard for arbitrary new files", "provenance + checksum + preprocessing recorded", "append-only manifest for management"),
        caveats=("video->keypoints stays external (run pose estimator first); units/coordinate_frame None unless supplied"),
    ),
)


FEATURE_BLOCKS: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        name="raw_keypoints",
        category="feature",
        input_format="(T,K,D)",
        output_format="(T,K*D)",
        module_path="numpy reshape",
        dependencies=("numpy",),
        strengths=("preserves pose geometry", "best baseline for PCA/UMAP/HMM"),
        caveats=("sensitive to camera frame, scale, and identity swaps"),
    ),
    ModuleSpec(
        name="skeleton_kinematic",
        category="feature",
        input_format="(T,K,D)",
        output_format="(T,4): speed, acceleration, spread, spatial variance",
        module_path="behavior_lab.data.features.SkeletonBackend",
        dependencies=("scipy",),
        strengths=("fast, interpretable summary", "works for 2D and 3D"),
        caveats=("coarse; misses joint-specific motifs"),
    ),
    ModuleSpec(
        name="dyadic_egocentric",
        category="feature",
        input_format="CalMS21 raw (T,2,2,7)",
        output_format="(T,24)",
        module_path="behavior_lab.data.features.dyadic.ego_centric_dyadic",
        dependencies=("numpy",),
        strengths=("explicit social geometry", "rotation/translation invariant"),
        caveats=("currently specialized to two-mouse MARS/CalMS21 layout"),
    ),
    ModuleSpec(
        name="bsoid_spatiotemporal",
        category="feature",
        input_format="(T,K,2/3)",
        output_format="10 fps displacement + pairwise distance + angular-change matrix",
        module_path="behavior_lab.models.discovery.bsoid._compute_bsoid_features",
        dependencies=("numpy",),
        strengths=("classic behavior segmentation feature set", "dimension-agnostic for 2D/3D"),
        caveats=("temporal binning changes label length; align before metrics"),
    ),
    ModuleSpec(
        name="morlet_cwt",
        category="feature",
        input_format="(T,K,D)",
        output_format="time-frequency spectral features",
        module_path="behavior_lab.data.features.morlet_backend.MorletCWTBackend",
        dependencies=("scipy",),
        strengths=("captures rhythmic motifs", "SUBTLE-style spectral representation"),
        caveats=("feature dimension grows quickly with joints/frequencies"),
    ),
    ModuleSpec(
        name="self_supervised_embedding",
        category="feature",
        input_format="windows of (T,K,D)",
        output_format="latent vectors",
        module_path="behavior_lab.models.ssl; behavior_lab.models.discovery.behavemae",
        dependencies=("torch",),
        strengths=("best when motifs are nonlinear or cross-species", "supports frozen representation comparison"),
        caveats=("requires checkpoint/training protocol; less interpretable than hand features"),
    ),
)


DISCOVERY_METHODS: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        name="kmeans_pca_umap",
        category="discovery",
        input_format="feature matrix (N,F)",
        output_format="labels + 2D embedding",
        module_path="behavior_lab.models.discovery.clustering.cluster_features",
        dependencies=("scikit-learn", "umap-learn"),
        strengths=("fast baseline", "fixed cluster count for controlled ablations"),
        caveats=("not temporal; cluster count is user-chosen"),
    ),
    ModuleSpec(
        name="B-SOiD",
        category="discovery",
        input_format="(T,K,2/3)",
        output_format="10 fps labels, embedding, RF classifier",
        module_path="behavior_lab.models.discovery.bsoid.BSOiD",
        dependencies=("umap-learn", "hdbscan", "scikit-learn"),
        strengths=("density-based syllables", "predictor can relabel new recordings"),
        caveats=("can over-fragment; output cadence differs from native fps"),
    ),
    ModuleSpec(
        name="keypoint-moseq",
        category="discovery",
        input_format="(T,K,D)",
        output_format="syllable sequence + latent state",
        module_path="behavior_lab.models.discovery.moseq.KeypointMoSeq",
        dependencies=("keypoint-moseq",),
        strengths=("explicit AR-HMM/SLDS temporal dynamics", "transition matrices are first-class outputs"),
        caveats=("heavier install/runtime; project directory state must be managed"),
    ),
    ModuleSpec(
        name="pca_hmm_fallback",
        category="discovery",
        input_format="(T,K,D)",
        output_format="HMM state labels",
        module_path="behavior_lab.models.discovery.moseq._PCAHMMFallback",
        dependencies=("hmmlearn",),
        strengths=("lightweight MoSeq-like temporal baseline",),
        caveats=("not a replacement for full keypoint-MoSeq SLDS"),
    ),
    ModuleSpec(
        name="SUBTLE",
        category="discovery",
        input_format="(T,K,D)",
        output_format="hierarchical motif labels",
        module_path="behavior_lab.models.discovery.subtle_wrapper.SUBTLE",
        dependencies=("subtle",),
        strengths=("time-frequency motifs", "strong for spontaneous 3D movement"),
        caveats=("macOS native package can be unstable; subprocess isolation recommended"),
    ),
    ModuleSpec(
        name="hBehaveMAE",
        category="discovery",
        input_format="windowed keypoints",
        output_format="hierarchical action/movement/activity clusters",
        module_path="behavior_lab.models.discovery.behavemae.BehaveMAE",
        dependencies=("torch",),
        strengths=("pretrained representation comparison", "hierarchical behavior discovery"),
        caveats=("dataset-specific input shape/checkpoint compatibility matters"),
    ),
    ModuleSpec(
        name="VAME",
        category="discovery",
        input_format="(T,K,D) egocentric-aligned pose",
        output_format="motif labels + RNN-VAE latent embedding",
        module_path="behavior_lab.models.discovery.vame.VAME",
        dependencies=("vame-py",),
        strengths=("self-supervised RNN-VAE representation", "hierarchical motif->community structure"),
        caveats=("cluster count not automatic; latent is a black box; seed/window sensitive; own multiprocessing may need subprocess isolation on macOS"),
    ),
)


def list_pose_sources() -> list[ModuleSpec]:
    return list(POSE_SOURCES)


def list_feature_blocks() -> list[ModuleSpec]:
    return list(FEATURE_BLOCKS)


def list_discovery_methods() -> list[ModuleSpec]:
    return list(DISCOVERY_METHODS)


def as_markdown_table(specs: tuple[ModuleSpec, ...] | list[ModuleSpec]) -> str:
    header = "| Name | Input | Output | Module | Strengths | Caveats |\n|---|---|---|---|---|---|"
    rows = [
        "| {name} | {inp} | {out} | `{mod}` | {strengths} | {caveats} |".format(
            name=s.name,
            inp=s.input_format,
            out=s.output_format,
            mod=s.module_path,
            strengths="<br>".join(s.strengths),
            caveats="<br>".join(s.caveats),
        )
        for s in specs
    ]
    return "\n".join([header, *rows])


__all__ = [
    "ModuleSpec",
    "POSE_SOURCES",
    "FEATURE_BLOCKS",
    "DISCOVERY_METHODS",
    "list_pose_sources",
    "list_feature_blocks",
    "list_discovery_methods",
    "as_markdown_table",
]
