# behavior-lab: Commands & Quick Reference

---

## Environment Setup

```bash
# Create conda environment
conda create -n behavior-lab python=3.12 -y
conda activate behavior-lab

# Install with all optional dependencies
pip install -e ".[viz,clustering,bsoid,moseq-fallback,subtle,torch]"

# macOS: Required env vars (prevents SIGSEGV in UMAP/HDBSCAN)
export LOKY_MAX_CPU_COUNT=1
export OMP_NUM_THREADS=1
```

---

## Scripts

### 1. Feature Backend Visualization

**What**: SkeletonBackend feature extraction pipeline analysis on CalMS21.
Generates 5 diagnostic panels (distributions, embeddings, clusters, ethogram, temporal methods).

```bash
LOKY_MAX_CPU_COUNT=1 python scripts/viz_feature_backends.py
```

**Output**: `outputs/feature_backend_viz/fig{1-5}_*.png` (~1 MB total)
**Runtime**: ~30 seconds

---

### 2. Method Comparison + Report (canonical)

> `compare_clustering.py` / `generate_cluster_report.py` were **removed (260705)** — the
> multi-method benchmark + HTML report now live on the shared modules (single common path):
> `experiments.compare_discovery_methods` + `visualization.render_comparison_report`.
> Heavy methods (keypoint-MoSeq/VAME) run in isolated conda envs via `scripts/isolated_run.py`.

```python
from behavior_lab.data import ingest
from behavior_lab.experiments.discovery import compare_discovery_methods
from behavior_lab.visualization import render_comparison_report

seqs = ingest("path/to/keypoints.npz")              # DLC/SLEAP/npz -> BehaviorSequence
runs = compare_discovery_methods(                    # one comparable API
    seqs[0].keypoints, methods=("kmeans_pca_umap", "bsoid", "pca_hmm_fallback"))
render_comparison_report(runs, "outputs/comparison.html", fps=30.0)   # ONE comparable HTML
```

**Isolated heavy method** (own env, no torch conflict):
```bash
conda run -n kpms python scripts/isolated_run.py --method keypoint_moseq --npz <file.npz> --out outputs/iso/kpms
```

**Workbench batch runner** (over local dataset slices): `python scripts/run_behavior_workbench_batch.py`

---

### 4. End-to-End Pipeline Test

**What**: Full pipeline verification (data load → preprocess → discover → evaluate → visualize → report).
Tests CalMS21, SUBTLE, Shank3KO with all modules.

```bash
LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 python scripts/test_e2e.py
```

**Output**: `outputs/e2e_test/` (per-dataset subdirs + `pipeline_report.html`)
**Runtime**: ~30-60 minutes

---

## Typical Workflow

```python
# One config-driven call — works for ANY species (mouse/rat/human).
from behavior_lab.experiments import run_comparison

# Light methods in-process; heavy methods (keypoint-MoSeq/VAME) precomputed in
# isolated conda envs via scripts/isolated_run.py and passed as extra_labels.
run_comparison(
    "data/keypoints.npz", "calms21",              # source + DatasetSpec key
    "outputs/comparison.html",                    # metrics + ARI/NMI + ethogram + agreement
    extra_labels={"keypoint_moseq": kpms_labels, "VAME": vame_labels},
    ground_truth=labels,                          # optional gold-standard eval
    gallery_html="outputs/cluster_gallery.html")  # optional per-cluster skeleton GIFs
```

```bash
# Heavy methods run in isolated envs first (no torch/jax conflict):
conda run -n kpms python scripts/isolated_run.py --method keypoint_moseq --npz data.npz --out outputs/iso/kpms
conda run -n vame python scripts/isolated_run.py --method vame          --npz data.npz --out outputs/iso/vame
```

---

## Python API

### Data Loading

```python
from behavior_lab.data.loaders import get_loader

loader = get_loader("calms21", data_dir="data/calms21")
sequences = loader.load_split("train")
kp = sequences[0].keypoints  # (T, 7, 2)
```

### Skeleton Registry

```python
from behavior_lab.core.skeleton import get_skeleton, list_skeletons

skeleton = get_skeleton("calms21")  # SkeletonDefinition
print(skeleton.joint_names)         # ['nose', 'left_ear', ...]
print(skeleton.edges)               # [(0,3), (1,3), ...]

print(list_skeletons())  # All available skeleton keys
```

### Preprocessing

```python
from behavior_lab.data.preprocessing.pipeline import (
    PreprocessingPipeline, Interpolator, OutlierRemover,
    TemporalSmoother, Normalizer,
)

pipeline = PreprocessingPipeline([
    Interpolator(max_gap=10),
    OutlierRemover(velocity_threshold=50),
    TemporalSmoother(window_size=5),
    Normalizer(center_joint=0),
])
clean_kp = pipeline(keypoints)
```

### Feature Extraction

```python
from behavior_lab.data.features import SkeletonBackend
from behavior_lab.data.features.cebra_backend import CEBRABackend

# Kinematic features
backend = SkeletonBackend(fps=30.0)
features = backend.extract(keypoints)  # (T, K, D) → (T, 4)

# CEBRA temporal embeddings
cebra = CEBRABackend(output_dim=32, max_iterations=5000)
embeddings = cebra.extract(keypoints)  # (T, K, D) → (T, 32)
```

### Clustering

```python
from behavior_lab.models.discovery.clustering import cluster_features

result = cluster_features(features, n_clusters=8, use_umap=True)
# result["labels"]:       (N,) cluster assignments
# result["embedding_2d"]: (N, 2) UMAP projection
# result["n_clusters"]:   int
```

### B-SOiD

```python
from behavior_lab.models.discovery.bsoid import BSOiD

model = BSOiD(fps=30, min_cluster_size=50)
result = model.fit(keypoints)  # (T, K, 2) → ClusteringResult
# result.labels, result.embeddings, result.n_clusters
```

### Evaluation

```python
from behavior_lab.evaluation import (
    compute_cluster_metrics, compute_behavior_metrics, linear_probe,
)

# Clustering quality
cm = compute_cluster_metrics(features, labels, true_labels)
# cm.silhouette, cm.nmi, cm.ari, cm.hungarian_accuracy

# Behavior analysis
bm = compute_behavior_metrics(labels, fps=30.0)
# bm.bout_durations, bm.transition_matrix, bm.temporal_consistency
```

### Visualization

```python
from behavior_lab.visualization import (
    plot_embedding, plot_skeleton, animate_skeleton,
    plot_transition_matrix, plot_temporal_raster,
)
from behavior_lab.visualization.html_report import (
    generate_pipeline_report, generate_cluster_animations,
    image_to_base64,
)

# Static skeleton plot
fig, ax = plot_skeleton(keypoints, skeleton=skeleton, frame=0)

# Animated skeleton GIF
anim = animate_skeleton(keypoints, skeleton=skeleton, fps=15, save_path="anim.gif")

# Per-cluster GIFs
gifs = generate_cluster_animations(
    keypoints, labels, skeleton, out_dir="gifs/",
    n_frames=480, fps=15.0, max_clusters=8,
)

# HTML report
generate_pipeline_report(report_data, "report.html", title="My Report")
```

---

## Data Directory Structure

```
data/
├── calms21/
│   └── calms21_aligned.npz        # (19144, 2, 64, 7, 2)
├── raw/
│   ├── subtle/
│   │   └── y5a5_adult_*.csv       # (T, 27) = 9 joints × 3D
│   ├── shank3ko/
│   │   └── Shank3KO_mice_slk3D.mat  # struct: CoordX/Y/Z (T, 16)
│   └── mabe22/
│       └── mouse_user_train.npy   # dict of sequences (T, 3, 12, 2)
├── ntu_rgbd/                      # NTU RGB+D .npz files
└── nw_ucla/                       # NW-UCLA .npz files
```

---

## Output Directory Structure

```
outputs/
├── feature_backend_viz/           # SkeletonBackend analysis
│   └── fig{1-5}_*.png
├── clustering_comparison/         # Model comparison benchmark
│   ├── comparison_*.png           # Per-dataset comparison figures
│   ├── cache/*.npz                # Cached model results
│   ├── gifs/                      # Per-cluster animation GIFs
│   │   └── {dataset}/{model}/cluster_*.gif
│   └── report.html                # Self-contained HTML report
└── e2e_test/                      # Full pipeline verification
    ├── calms21/
    ├── subtle/
    ├── shank3ko/
    └── pipeline_report.html
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| SIGSEGV (exit code 139) | macOS + loky multiprocessing | `LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1` |
| SUBTLE "0 clusters" on 2D data | SUBTLE requires 3D skeleton | Expected behavior — model skips 2D datasets |
| "No checkpoint" for hBehaveMAE | Missing pretrained weights | Download `hBehaveMAE_MABe22.pth` to `checkpoints/behavemae/` |
| UMAP import error | Optional dependency | `pip install umap-learn` |
| scipy CWT deprecation | scipy >= 1.12 removed `signal.cwt` | Auto-patched in `subtle_wrapper.py` |

---

*behavior-lab v0.1.0 | Commands Reference | 2026-02-09*
