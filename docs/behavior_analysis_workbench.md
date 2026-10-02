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

## Two-Stage View

Pose estimation is Stage 0; representation (Stage 1) and discovery and analysis (Stage 2) are compared on the shared `(T,K,D)` contract.
Canonical text lives in the vault note `30_Projects/Behavior-Lab/_Agent/261001_Behavior_two_stage_concept.md` (method placement table included); kept there to avoid two copies.
Only the `kmeans_pca_umap` path honors `feature=` (`experiments/discovery.py`); other methods compute Stage 1 internally.

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
| VAME | (T,K,D) egocentric-aligned pose | motif labels + RNN-VAE latent embedding | `behavior_lab.models.discovery.vame.VAME` | self-supervised RNN-VAE representation<br>hierarchical motif→community structure | cluster count not automatic; latent is a black box; seed/window sensitive |

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
- `notebooks/behavior_analysis_workbench/03_batch_results_all_methods.ipynb`
  - Reads the dataset x method grid written by `scripts/run_behavior_workbench_batch.py`.
  - Tables and heatmaps per cell, then label agreement on one clip (ethogram per method, ARI between methods, ARI between dataset slices for one method).

## Dataset x Method Grid

One script fills the grid, run once per environment or machine; rows are merged by `(dataset, method)` into `outputs/behavior_analysis_workbench/<out>/`.

```bash
# light methods (main env; the HMM fallback needs --extra moseq-fallback)
uv run python scripts/run_behavior_workbench_batch.py --out long --max-frames 30000 --repeats 3 \
    --datasets avatar,subtle,shank3ko --methods kmeans_pca_umap,B-SOiD,pca_hmm_moseq_fallback
# SUBTLE (own env)
UV_PROJECT_ENVIRONMENT=.venv-subtle uv run --extra subtle --extra viz --extra clustering \
    python scripts/run_behavior_workbench_batch.py --out long --max-frames 30000 --repeats 3 \
    --datasets avatar,subtle,shank3ko --methods SUBTLE
# keypoint-MoSeq: Linux only (own env, CPU works); then copy its folder back and merge
UV_PROJECT_ENVIRONMENT=.venv-moseq uv sync --python 3.11 --extra moseq --extra viz --extra clustering
JAX_PLATFORMS=cpu .venv-moseq/bin/python scripts/run_behavior_workbench_batch.py --out long \
    --max-frames 30000 --repeats 3 --datasets avatar,subtle,shank3ko --methods keypoint_moseq
uv run python scripts/run_behavior_workbench_batch.py --out long --merge <copied folder>
# shareable page (plain HTML source; the vault's build_page.py adds theme and back link)
uv run python -m behavior_lab.visualization.grid_report outputs/behavior_analysis_workbench/long  # report, gallery, playback
```

Slices and inputs

- A slice is one recording or one pose variant, fitted on its own. SUBTLE recordings (`y5a5_*`, 9 keypoints) and two same-date Shank3KO recordings (16 keypoints) run at full length; AVATAR slices come from `$BEHAVIOR_LAB_AVATAR_DIR` (default `~/data/avatar_gslrm/keypoints/*_gslrm.npz`), one per pose post-processing variant, in the file's native 11-point layout (see `architecture.md` "Pose Output Contract").
- Recordings are never concatenated: a splice is a fake transition for temporal models.
- Missing keypoints are interpolated over time before any method runs; `nan_frac` and `max_gap_frames` are stored per slice.

Repeats and agreement

- `--repeats N` runs seeds `seed..seed+N-1`; the seed reaches kmeans/UMAP, B-SOiD, the HMM fallback and keypoint-MoSeq. SUBTLE has no seed upstream, so its repeats differ by design. `repeat_ari_mean` per cell is the noise floor for every other comparison.
- `behavior_lab.visualization.agreement` (used by notebook 03 and the HTML report) reports ARI with a circular-shift null, AMI, and homogeneity in both directions, because ARI alone drops when one method merely splits another's labels more finely.
- B-SOiD labels are 10 Hz bins; bout durations use that rate. Bins are whole frames starting at frame 0 with a short tail dropped, so `stretch_labels` maps frame `i` to bin `i // bin_size` (the last bin covers the tail); other length ratios fall back to `floor(i * L / T)`.

Reading the page

- Result grid: each method has one fixed color. Green marks the most repeatable method per slice (highest repeat ARI among unflagged rows); it is a stability mark, not a quality ranking, and a seeded kmeans on a fixed feature set will usually take it. Rows are flagged, and left out of that ranking, when the segmentation is degenerate: fewer than 3 clusters, a median bout of one label step, or noise above 30% (own loose rule).
- Page order: keypoint layouts, run setup, result grid over all cells, then one section per dataset family (name prefix) with one subsection per slice. Each slice has the same foldable blocks (`visualization.grid_slice`): label sequences and agreement, cluster sizes, cluster maps on shared axes, SUBTLE's own map, cluster correspondence.
- Keypoint layouts (`visualization.keypoint_schema`): one real frame per family (the one closest to the median pairwise joint distances, not a mean pose) with joint indices, names and bones. Methods take any `(T, K, D)` layout; bones are only needed for drawing and come from the skeleton registry, and keypoint-MoSeq finds nose and tail base by joint name. A layout without a registered bone list is drawn as points.
- Shared cluster maps (`cluster_map.pose_embedding`, `plot_method_maps`): a 2D PCA of pairwise joint distances, joint heights and centroid speed, the same axes for every method. Chosen because no method clusters on it: SUBTLE's own UMAP as common axes would make SUBTLE look clean and the others scattered. Clusters that overlap on it can still differ in dynamics.
- Cluster correspondence (`agreement.match_clusters`): one-to-one pairs between two methods by Jaccard overlap of frames (Hungarian assignment, noise left out), sorted by Jaccard. Jaccard grows with cluster size, so each pair shows the Jaccard expected for independent clusters of those sizes. `p` comes from 200 circular shifts and compares against the best Jaccard over all cluster pairs of the shifted data, so picking the best of many pairs is accounted for. A group of each method's largest clusters is labelled as such: it is expected from size alone. Matching needs the same frames, so it is never done across recordings.
- Playback (`batch_player.html`, `visualization.player`): skeleton, position on the shared map and every method's label row in sync at 10 Hz. Pose, map point and labels are subsampled with one index vector. Recent positions are fading dots, not a line, because the plane is a projection.
- SUBTLE cluster map (`visualization.cluster_map`): one run's UMAP embedding colored by subcluster and supercluster, with transition arrows. The batch script stores `subtle_map_seed<seed>.npz` per SUBTLE run so embedding and labels always come from the same run; `--subtle-map` adds one extra run without touching the result rows.
- Cluster gallery (`batch_gallery.html`): per method, skeleton GIFs of the 6 most frequent clusters on the slices that have a map. Below them, the 6 best matched cluster pairs across methods (p < 0.05, largest-cluster pairs left out), each drawn from frames both methods assign to the pair. Each GIF is one real bout of median length (at most 2 s) with 0.5 s of context, never several bouts stitched. AVATAR is drawn as points because its bone list is not registered in behavior-lab.

keypoint-MoSeq recipe and install

- Fit = the modeling tutorial's two stages: AR-HMM only for 50 iterations, then the full model for 500 with kappa 1e4; tail keypoints excluded; `latent_dim = min(10, dims for 90% variance)`. kappa is not tuned to a target syllable duration, so read `median_bout_sec` before its agreement numbers.
- It installs on Linux with Python < 3.13 only: `keypoint-moseq>=0.6` depends on `jax-cuda12-pjrt` (Linux wheels), and 0.4.x does not import against current `dynamax`. The lock pins `jax 0.6.x` and `tfp-nightly==0.26.0.dev20260704` (the set in `env_snapshots/kpms.yml`); a newer nightly breaks `dynamax`.
- Measured on a 1-CPU WSL box: 20 iterations on 12,010 frames in 61.5 s including compilation.

Results and their numbers live in the vault experiment note `30_Projects/Behavior-Lab/_Agent/Experiment/261002_behaviorlab_discovery_grid_long_run.md` and the page built from the grid; they are not copied here.

SUBTLE wrapper labels written before 2026-10-01 are not in time order: `SUBTLE.fit()` returned `Mapper.y`, which is in the shuffled training order, and flattened the `(T, n_levels)` supercluster array (label length `T * n_levels`). Measured on 1,200 SUBTLE frames: mean bout 1.07 frames in the returned order against 9.02 frames in time order. Fixed in `subtle_wrapper.py`.

- Affected: anything temporal computed from `SUBTLE.fit()` / `fit_predict()` labels (bout durations, transitions, ethograms, ARI against frame labels), i.e. the `SUBTLE` rows of the June `batch/batch_results.csv`.
- Not affected: cluster counts and UMAP scatter plots (order-independent), which is all `phase4_report.md` and `subtle_pipeline_reference.md` report for SUBTLE; and `notebooks/calms21_behavior_discovery/01_subtle_baseline.ipynb`, which calls the upstream API and reads the per-session, time-ordered `out.y`.

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
