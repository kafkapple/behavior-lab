# Behavior Analysis Workbench

This document is the integrated map for pose-derived behavior analysis in
`behavior-lab`. Keep source-video utilities in `behavior-tools`; keep reusable
analysis modules, dataset loaders, feature extraction, clustering, metrics, and
notebooks here.

> Project scope and operating rules: [Behavior Analysis PRD](behavior_analysis_prd.md)
> Theory appendix: [behavior_analysis_principles.md](behavior_analysis_principles.md)

## Pipeline

```text
video / dataset
  -> pose source or loader
  -> canonical BehaviorSequence(keypoints=(T,K,D), labels?, metadata)
  -> preprocessing / triangulation / confidence filtering
  -> feature block or learned embedding
  -> unsupervised discovery
  -> motif/syllable labels
  -> bout durations, transition matrix, ethogram, embedding plots
```

The canonical exchange format is always `(T,K,D)`: time, keypoint, coordinate.
Multi-animal recordings can either be flattened into `(T, M*K, D)` or kept as
separate tracks when a downstream method needs identity-specific handling.

## Pose And Feature Sources

Generated from `behavior_lab.data.features.catalog`:

| Name | Input | Output | Module | Strengths | Caveats |
|---|---|---|---|---|---|
| DeepLabCut | video or DLC h5/csv | (T,K,2/3) keypoints + likelihood | `scripts/*dlc*.sh; outputs/kp_benchmark/*.npz` | strong supervised/SuperAnimal ecosystem<br>good 2D animal keypoint tooling | project-specific bodypart order must be mapped to skeleton registry |
| SLEAP | .slp or SLEAP analysis .h5 | BehaviorSequence (T,K,D), flattened or per-track | `behavior_lab.pose.sleap; behavior_lab.data.loaders.sleap` | multi-animal tracking metadata<br>confidence-aware NaN masking | 3D requires external triangulation or calibrated multi-view export |
| Anipose/Triangulation | multi-view 2D keypoints + camera calibration | (T,K,3) triangulated sparse keypoints | `scripts/anipose_triangulate.py` | 3D reconstruction from DLC/SLEAP 2D tracks<br>RANSAC/linear comparison possible | accuracy dominated by calibration, sync, and 2D confidence filtering |
| Canonical NPZ loaders | CalMS21, MABe22, Rat7M, NTU, NW-UCLA, SUBTLE, Shank3KO npz/npy/mat/csv | BehaviorSequence list with (T,K,D) | `behavior_lab.data.loaders` | current local datasets use one common format<br>works without pose-estimator installs | dataset coordinate frames and units differ; normalize before cross-dataset comparison |

## Feature Blocks

| Name | Input | Output | Module | Strengths | Caveats |
|---|---|---|---|---|---|
| raw_keypoints | (T,K,D) | (T,K*D) | `numpy reshape` | preserves pose geometry<br>best baseline for PCA/UMAP/HMM | sensitive to camera frame, scale, and identity swaps |
| skeleton_kinematic | (T,K,D) | (T,4): speed, acceleration, spread, spatial variance | `behavior_lab.data.features.SkeletonBackend` | fast, interpretable summary<br>works for 2D and 3D | coarse; misses joint-specific motifs |
| dyadic_egocentric | CalMS21 raw (T,2,2,7) | (T,24) | `behavior_lab.data.features.dyadic.ego_centric_dyadic` | explicit social geometry<br>rotation/translation invariant | currently specialized to two-mouse MARS/CalMS21 layout |
| bsoid_spatiotemporal | (T,K,2/3) | 10 fps displacement + pairwise distance + angular-change matrix | `behavior_lab.models.discovery.bsoid._compute_bsoid_features` | classic behavior segmentation feature set<br>dimension-agnostic for 2D/3D | temporal binning changes label length; align before metrics |
| morlet_cwt | (T,K,D) | time-frequency spectral features | `behavior_lab.data.features.morlet_backend.MorletCWTBackend` | captures rhythmic motifs<br>SUBTLE-style spectral representation | feature dimension grows quickly with joints/frequencies |
| self_supervised_embedding | windows of (T,K,D) | latent vectors | `behavior_lab.models.ssl; behavior_lab.models.discovery.behavemae` | best when motifs are nonlinear or cross-species<br>supports frozen representation comparison | requires checkpoint/training protocol; less interpretable than hand features |

## Unsupervised Discovery Methods

| Name | Input | Output | Module | Strengths | Caveats |
|---|---|---|---|---|---|
| kmeans_pca_umap | feature matrix (N,F) | labels + 2D embedding | `behavior_lab.models.discovery.clustering.cluster_features` | fast baseline<br>fixed cluster count for controlled ablations | not temporal; cluster count is user-chosen |
| B-SOiD | (T,K,2/3) | 10 fps labels, embedding, RF classifier | `behavior_lab.models.discovery.bsoid.BSOiD` | density-based syllables<br>predictor can relabel new recordings | can over-fragment; output cadence differs from native fps |
| keypoint-moseq | (T,K,D) | syllable sequence + latent state | `behavior_lab.models.discovery.moseq.KeypointMoSeq` | explicit AR-HMM/SLDS temporal dynamics<br>transition matrices are first-class outputs | heavier install/runtime; project directory state must be managed |
| pca_hmm_fallback | (T,K,D) | HMM state labels | `behavior_lab.models.discovery.moseq._PCAHMMFallback` | lightweight MoSeq-like temporal baseline | not a replacement for full keypoint-MoSeq SLDS |
| SUBTLE | (T,K,D) | hierarchical motif labels | `behavior_lab.models.discovery.subtle_wrapper.SUBTLE` | time-frequency motifs<br>strong for spontaneous 3D movement | macOS native package can be unstable; subprocess isolation recommended |
| hBehaveMAE | windowed keypoints | hierarchical action/movement/activity clusters | `behavior_lab.models.discovery.behavemae.BehaveMAE` | pretrained representation comparison<br>hierarchical behavior discovery | dataset-specific input shape/checkpoint compatibility matters |

## Notebook Interface

Primary notebooks:

- `notebooks/behavior_analysis_workbench/00_end_to_end_overview.ipynb`
  - Loads an available local dataset.
  - Lists pose sources, feature blocks, and discovery methods.
  - Runs a lightweight comparable baseline.
  - Plots ethogram, bout durations, transition matrix, and embedding.
- `notebooks/behavior_analysis_workbench/01_sleap_import_and_triangulation.ipynb`
  - Demonstrates SLEAP analysis H5 import.
  - Shows where DLC/SLEAP 2D tracks feed triangulation.
  - Keeps 3D sparse keypoint evaluation separate from behavior discovery.
- `notebooks/behavior_analysis_workbench/02_method_comparison_matrix.ipynb`
  - Structured comparison template for dataset x feature x method sweeps.
  - Supports B-SOiD, SUBTLE, keypoint-MoSeq, hBehaveMAE, and lightweight baselines.

## Minimal API

```python
from behavior_lab.data.loaders import get_loader
from behavior_lab.experiments import compare_discovery_methods
from behavior_lab.evaluation import compute_behavior_metrics

seq = get_loader("calms21", data_dir="data/calms21").load_split("train")[0]
runs = compare_discovery_methods(
    seq.keypoints,
    methods=("kmeans_pca_umap", "bsoid"),
    feature="skeleton_kinematic",
    fps=seq.fps,
    max_frames=3000,
)
metrics = compute_behavior_metrics(runs[0].result.labels, fps=seq.fps)
```

SLEAP:

```python
from behavior_lab.pose import load_sleap_file

result = load_sleap_file(
    "predictions.analysis.h5",
    instance_mode="flatten",
    confidence_threshold=0.2,
)
seq = result.sequences[0]
```

## Dataset Comparison Notes

Use the same comparison axes for every dataset:

| Axis | What to record |
|---|---|
| Species / setting | mouse single, mouse dyad, triplet, human, fly |
| Keypoint source | manual 3D, DLC, SLEAP, triangulated 3D, benchmark NPZ |
| Geometry | 2D top view, 3D calibrated, egocentric dyadic, graph skeleton |
| Temporal scale | frame-level, 10 fps B-SOiD bins, 1 s windows, syllables |
| Discovery output | clusters, HMM states, MoSeq syllables, hierarchy levels |
| Metrics | silhouette, CH/DB, NMI/ARI if labels exist, bout duration, transition entropy |

For cross-dataset conclusions, separate pose quality from behavior discovery:
DLC/SLEAP/triangulation affect coordinate noise and missingness; B-SOiD,
SUBTLE, keypoint-MoSeq, and hBehaveMAE affect representation and temporal
segmentation. Do not compare method quality without reporting both layers.

## Theory Appendix

Detailed principles, metric interpretation, and the CEBRA/behavior-segmentation
discussion live in [`behavior_analysis_principles.md`](behavior_analysis_principles.md).
Use it when you need the why behind the workflow rather than the workflow
itself.
