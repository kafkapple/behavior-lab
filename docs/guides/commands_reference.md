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

### 2. Clustering Model Comparison

**What**: Runs 6 discovery models on 4 datasets, generates comparison figures + caches results.

```bash
LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 python scripts/compare_clustering.py
```

**Output**:
- `outputs/clustering_comparison/comparison_{dataset}.png` (4 files)
- `outputs/clustering_comparison/cache/{dataset}.npz` (4 cached results)

**Runtime**: ~20-30 minutes (all models)

**Models**: Clustering, B-SOiD, MoSeq (HMM), SUBTLE, hBehaveMAE, CEBRA
**Datasets**: CalMS21, SUBTLE, Shank3KO, MABe22

---

### 3. HTML Report Generation

**What**: Generates self-contained HTML report with comparison PNGs + per-cluster GIF animations.
Uses cached results from `compare_clustering.py`.

```bash
# Default: top 2 models by Silhouette, max 8 clusters per model
python scripts/generate_cluster_report.py

# Specific models only
python scripts/generate_cluster_report.py --models clustering,cebra

# More GIFs: top 3 models, max 6 clusters
python scripts/generate_cluster_report.py --top-n 3 --max-clusters 6

# Custom GIF settings
python scripts/generate_cluster_report.py --n-frames 240 --fps 10

# Force re-run models (no cache)
python scripts/generate_cluster_report.py --rerun

# Custom output path
python scripts/generate_cluster_report.py --output my_report.html
```

**Output**: `outputs/clustering_comparison/report.html` (~10-20 MB)
**Runtime**: ~10 minutes (cached) / ~35 minutes (rerun)

**CLI Options**:

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | all | Comma-separated model names |
| `--rerun` | false | Force re-run (ignore cache) |
| `--max-clusters` | 8 | Max clusters for GIF generation |
| `--top-n` | 2 | Top N models (by Silhouette) for GIFs |
| `--n-frames` | 480 | Frames per GIF animation |
| `--fps` | 15.0 | GIF playback speed |
| `--output` | report.html | Output HTML path |

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

```bash
# Step 1: Run model comparison (generates PNGs + cache)
LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 python scripts/compare_clustering.py

# Step 2: Generate HTML report (uses cache, adds per-cluster GIF animations)
python scripts/generate_cluster_report.py

# Step 3: View in browser
open outputs/clustering_comparison/report.html
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
