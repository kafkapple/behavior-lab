# Quick Start Guide: Experiment Workflow

> [← MoC](../README.md) | [E2E Verification](../e2e_verification.md) | [Architecture](../architecture.md)

Step-by-step guide for running behavior analysis experiments with behavior-lab.

---

## 1. Installation

```bash
# Clone
git clone https://github.com/kafkapple/behavior-lab.git
cd behavior-lab

# Create conda environment (recommended)
conda create -n behavior-lab python=3.10
conda activate behavior-lab

# Install with visualization support
pip install -e ".[viz,clustering,bsoid,loaders]"

# Or full installation
pip install -e ".[all]"
```

### Verify Installation

```python
python -c "
from behavior_lab.models import list_models
from behavior_lab.core.skeleton import list_skeletons
print('Models:', list_models())
print('Skeletons:', list_skeletons())
"
```

---

## 2. Data Preparation

Place data files under `data/{dataset_name}/`:

```
data/
├── calms21/
│   └── calms21_task1_train.npz    # NPZ: x_train (N,2,T,7,2), y_train (N,4)
├── ntu/
│   └── demo_CS_aligned.npz        # NPZ: x_train (N,300,150), y_train (N,60)
└── nwucla/
    └── nwucla_aligned.npz          # NPZ: x_train (N,300,60), y_train (N,10)
```

### Dataset Details

#### CalMS21 (Mouse Social Behavior)

- **Species**: Mouse (top-view, 2 animals)
- **Skeleton**: 7 joints × 2 mice = 14 keypoints per frame
- **Joints**: nose, left_ear, right_ear, neck, left_hip, right_hip, tail_base
- **Dimensions**: 2D (x, y pixel coordinates)
- **Classes**: `other(0)`, `attack(1)`, `investigation(2)`, `mount(3)`
- **Format**: Each sequence = 64 frames, labels = per-sequence (uniform)
- **Use case**: Unsupervised behavior discovery, social interaction analysis

#### NTU RGB+D 60 (Human Action Recognition)

- **Species**: Human (Kinect depth sensor, 1-2 persons)
- **Skeleton**: 25 joints per person (up to 50 joints total)
- **Joints**: base_spine, mid_spine, neck, head, shoulders, elbows, wrists, hands, hips, knees, ankles, feet, thumbs
- **Dimensions**: 3D (x, y, z world coordinates)
- **Classes**: 60 actions (drink water, eat meal, brush teeth, clap, ...)
- **Format**: Each sequence = 300 frames (zero-padded)
- **Use case**: Supervised action recognition, SSL pre-training

#### NW-UCLA (Action Recognition)

- **Species**: Human (multi-view Kinect, 1-10 persons)
- **Skeleton**: 20 joints per person
- **Joints**: hip_center, spine, shoulder_center, head, shoulders, elbows, wrists, hands, hips, knees, ankles, feet
- **Dimensions**: 3D (x, y, z)
- **Classes**: 10 actions (pick up, throw, sit down, stand up, ...)
- **Format**: Each sequence = 300 frames (zero-padded)
- **Use case**: Cross-view action recognition, linear probe baseline

---

## 3. Data Loading

```python
from behavior_lab.data.loaders import get_loader

# CalMS21
loader = get_loader("calms21", data_dir="data/calms21")
train_seqs = loader.load_split("train")
test_seqs = loader.load_split("test")

# NTU RGB+D (NPZ format)
loader = get_loader("ntu", data_dir="data/ntu")
train_seqs = loader.load_npz("data/ntu/demo_CS_aligned.npz", split="train")

# NW-UCLA
loader = get_loader("nwucla", data_dir="data/nwucla")
train_seqs = loader.load_split("train")
```

Each sequence is a `BehaviorSequence`:

```python
seq = train_seqs[0]
print(seq.keypoints.shape)    # (T, K, D) — e.g. (64, 14, 2)
print(seq.labels.shape)       # (T,) — per-frame labels
print(seq.skeleton_name)      # 'calms21_mouse'
print(seq.fps)                # 30.0
```

---

## 4. Preprocessing

```python
from behavior_lab.data.preprocessing.pipeline import (
    PreprocessingPipeline, Interpolator, OutlierRemover,
    TemporalSmoother, Normalizer,
)

pipeline = PreprocessingPipeline([
    Interpolator(max_gap=10),             # Fill NaN gaps (linear)
    OutlierRemover(velocity_threshold=50.0),  # Remove velocity spikes
    TemporalSmoother(window_size=5),      # Moving average smoothing
    Normalizer(center_joint=0),           # Body-size normalization
])

raw_kp = train_seqs[0].keypoints.copy()
cleaned_kp = pipeline(raw_kp)

print(f"Before: [{raw_kp.min():.1f}, {raw_kp.max():.1f}]")
print(f"After:  [{cleaned_kp.min():.3f}, {cleaned_kp.max():.3f}]")
```

### Pipeline Steps Explained

| Step | Purpose | Parameters |
|------|---------|------------|
| `Interpolator` | Fill missing (NaN) keypoints by linear interpolation | `max_gap`: max consecutive NaN frames to fill |
| `OutlierRemover` | Remove physically impossible jumps (velocity spikes) | `velocity_threshold`: max pixels/frame |
| `TemporalSmoother` | Reduce jitter with moving average | `window_size`: smoothing window |
| `Normalizer` | Normalize to body-size-invariant coordinates | `center_joint`: root joint index |

---

## 5. Skeleton Visualization

```python
import matplotlib
matplotlib.use("Agg")  # For non-interactive (server) environments

from behavior_lab.core.skeleton import get_skeleton
from behavior_lab.visualization import plot_skeleton, animate_skeleton, plot_skeleton_comparison

skeleton = get_skeleton("calms21")

# Static plot (body-part colored, multi-person auto-detected)
fig, ax = plot_skeleton(
    train_seqs[0].keypoints,
    skeleton=skeleton,
    frame=0,
    show_labels=True,           # Joint name abbreviations
    save_path="skeleton.png",
)

# Animation (GIF)
anim = animate_skeleton(
    train_seqs[0].keypoints[:60],   # First 60 frames
    skeleton=skeleton,
    fps=10.0,
    save_path="animation.gif",
)

# Side-by-side comparison (raw vs preprocessed)
fig, axes = plot_skeleton_comparison(
    [raw_kp, cleaned_kp],
    ["Raw", "Preprocessed"],
    skeleton=skeleton,
    frame=0,
    save_path="comparison.png",
)
```

### Color System

- **Body-part colors**: Each body part (head=red, torso=blue, left_arm=green, etc.) gets a distinct color, automatically derived from `skeleton.body_parts`
- **Multi-person**: When `skeleton.num_persons > 1` (CalMS21, NTU), each individual is rendered with a different color scheme
- **Labels**: `show_labels=True` overlays joint name abbreviations (e.g., "Nos" for nose, "LSh" for left_shoulder)

---

## 6. Behavior Discovery (B-SOiD)

B-SOiD discovers behavioral motifs from unlabeled pose data using a two-space approach: UMAP embedding + HDBSCAN clustering + Random Forest classification.

```python
import numpy as np
from behavior_lab.models.discovery.bsoid import BSOiD

# Concatenate all sequences
all_kp = np.concatenate([s.keypoints for s in train_seqs], axis=0)
print(f"Total frames: {all_kp.shape}")

# Fit and predict
bsoid = BSOiD(fps=30, min_cluster_size=50)
result = bsoid.fit_predict(all_kp)

print(f"Clusters: {result.n_clusters}")
print(f"Labels: {result.labels.shape}")
print(f"Embeddings: {result.embeddings.shape}")

# Save model for later use
bsoid.save("bsoid_model.pkl")
```

### B-SOiD Pipeline Internal Steps

1. **Feature extraction**: Inter-joint distances + angles at 10 fps (binned from original fps)
2. **UMAP embedding**: High-dimensional features → 3D embedding
3. **HDBSCAN clustering**: Density-based clustering on embedding
4. **Random Forest**: Train classifier on clusters → apply to all frames

---

## 7. Evaluation Metrics

```python
from behavior_lab.evaluation import (
    compute_cluster_metrics,
    compute_behavior_metrics,
    linear_probe,
)

# --- Cluster metrics (unsupervised) ---
cluster_metrics = compute_cluster_metrics(
    result.features,     # (N, D) feature matrix
    result.labels,       # (N,) predicted cluster labels
    true_labels=gt_labels,  # (N,) ground truth (optional)
)
print(f"Silhouette: {cluster_metrics.silhouette:.4f}")
print(f"NMI: {cluster_metrics.nmi:.4f}")
print(f"ARI: {cluster_metrics.ari:.4f}")
print(f"Hungarian Accuracy: {cluster_metrics.hungarian_accuracy:.4f}")

# --- Behavior metrics ---
beh_metrics = compute_behavior_metrics(result.labels, fps=10.0)
print(f"Temporal consistency: {beh_metrics.temporal_consistency:.4f}")
print(f"Num bouts: {beh_metrics.num_bouts}")
print(f"Entropy rate: {beh_metrics.entropy_rate:.4f}")
# beh_metrics.transition_matrix: (C, C) state transition probabilities
# beh_metrics.bout_durations: {cluster_id: mean_duration_sec}

# --- Linear probe (supervised baseline) ---
train_feat = np.array([s.keypoints.mean(axis=0).flatten() for s in train_seqs])
test_feat = np.array([s.keypoints.mean(axis=0).flatten() for s in test_seqs])
probe = linear_probe(train_feat, train_labels, test_feat, test_labels)
print(f"Accuracy: {probe.accuracy:.4f}, F1: {probe.f1_macro:.4f}")
```

### Metrics Reference

| Metric | Range | Meaning |
|--------|-------|---------|
| **Silhouette** | [-1, 1] | Cluster separation quality. >0.5 = good |
| **NMI** | [0, 1] | Agreement with ground truth labels (normalized) |
| **ARI** | [-1, 1] | Agreement adjusted for chance. >0.3 = reasonable |
| **Hungarian Accuracy** | [0, 1] | Best 1:1 cluster-to-label mapping accuracy |
| **Temporal Consistency** | [0, 1] | Fraction of consecutive same-label frame pairs |
| **Entropy Rate** | [0, +inf] | Behavioral diversity/switching rate |
| **Bout Duration** | seconds | Mean uninterrupted behavior episode length |

---

## 8. Analysis Visualization

```python
from behavior_lab.visualization import (
    plot_embedding,
    plot_transition_matrix,
    plot_bout_duration,
    plot_temporal_raster,
)

# UMAP embedding scatter plot
plot_embedding(
    result.embeddings, result.labels,
    title="B-SOiD Embedding",
    save_path="embedding.png",
)

# State transition heatmap
plot_transition_matrix(
    beh_metrics.transition_matrix,
    title="Behavior Transitions",
    save_path="transitions.png",
)

# Mean bout duration per cluster
plot_bout_duration(
    beh_metrics.bout_durations,
    title="Bout Durations",
    save_path="bout_duration.png",
)

# Ethogram (temporal raster)
plot_temporal_raster(
    result.labels[:5000],
    fps=10.0,
    title="Ethogram (First 5000 frames)",
    save_path="ethogram.png",
)
```

---

## 9. HTML Report Generation

Generate a self-contained HTML report with all figures embedded as base64:

```python
from behavior_lab.visualization import fig_to_base64
from behavior_lab.visualization.html_report import (
    generate_pipeline_report,
    image_to_base64,
)

# Build report data
report_data = {
    "title": "My Experiment Report",
    "datasets": {
        "calms21": {
            "data": {"n_train": 19144, "n_test": 4787, "shape": [64, 14, 2]},
            "figures": {
                "skeleton_static": fig_to_base64(fig_skeleton),
                "skeleton_gif": image_to_base64("animation.gif"),
                "embedding": fig_to_base64(fig_embedding),
                "transition": fig_to_base64(fig_transition),
            },
            "cluster_metrics": {
                "silhouette": 0.59,
                "nmi": 0.007,
                "n_clusters": 2,
            },
            "behavior_metrics": {
                "temporal_consistency": 0.897,
                "num_bouts": 2183,
            },
        }
    }
}

# Generate HTML
path = generate_pipeline_report(report_data, "report.html")
print(f"Report: {path}")
# Open in browser — single file, no external dependencies
```

### HTML Report Features

- **Self-contained**: All images base64-embedded, no external files needed
- **Tab navigation**: Overview + per-dataset tabs (click to switch)
- **Metric cards**: Hover-interactive value display
- **Responsive**: Works on desktop and mobile browsers
- **GIF support**: Animated skeletons embedded inline

---

## 10. Full Pipeline (E2E Test)

Run the complete pipeline on all 3 datasets:

```bash
python scripts/test_e2e.py
```

This executes:

1. **CalMS21**: Load → Preprocess → Skeleton viz → B-SOiD discovery → Cluster metrics → Behavior metrics → Embedding/transition/ethogram plots
2. **NTU RGB+D**: Load → Skeleton viz → Linear probe → Confusion matrix
3. **NW-UCLA**: Load → Skeleton viz → Animation GIF → Linear probe → Confusion matrix
4. **Report generation**: JSON + Markdown + HTML

### Outputs

```
outputs/e2e_test/
├── report.html                 # Open in browser
├── report.md                   # Markdown summary
├── report.json                 # Machine-readable
├── calms21/
│   ├── sample_skeleton.png     # Colored 2-mice skeleton
│   ├── sample_animation.gif    # Multi-person animated GIF
│   ├── skeleton_comparison.png # Raw vs preprocessed side-by-side
│   ├── preprocessing_comparison.png
│   ├── bsoid_embedding.png     # UMAP scatter
│   ├── bsoid_transition_matrix.png
│   ├── bsoid_bout_duration.png
│   ├── bsoid_ethogram.png
│   └── bsoid_model.pkl         # Saved B-SOiD model
├── ntu/
│   ├── sample_skeleton.png     # Body-part colored human skeleton
│   └── linear_probe_confusion.png
└── nwucla/
    ├── sample_skeleton.png     # 20-joint colored skeleton
    ├── sample_animation.gif    # Animated GIF
    └── linear_probe_confusion.png
```

---

## 11. Hydra CLI Experiments

### Supervised Training

```bash
# InfoGCN on NTU RGB+D
python scripts/train.py model=infogcn dataset=ntu60_xsub skeleton=ntu25

# Fast debug (small subset)
python scripts/train.py model=stgcn training=fast_debug

# Multi-run sweep
python scripts/train.py -m model=infogcn,stgcn,agcn dataset=ntu60_xsub
```

### SSL Pre-training

```bash
python scripts/train_ssl.py model=infogcn_dino dataset=calms21
python scripts/train_ssl.py model=gcn_mae dataset=ntu60_xsub
```

### Behavior Discovery

```bash
python scripts/discover.py model=bsoid dataset=calms21
python scripts/discover.py model=moseq dataset=calms21
```

### Evaluation

```bash
python scripts/evaluate.py model=lstm dataset=calms21
python scripts/benchmark.py  # Compare multiple models
```

---

## 12. Interpreting Results

### B-SOiD Discovery Results

**Expected behavior**: B-SOiD discovers *behavioral motifs* (movement patterns), not necessarily matching annotation categories. A 4-class annotated dataset may yield 2-5 B-SOiD clusters.

| Metric | Good Sign | Concern |
|--------|-----------|---------|
| Silhouette > 0.3 | Well-separated clusters | < 0.1 = no structure |
| Temporal consistency > 0.8 | Stable behavior bouts | < 0.5 = noisy switching |
| Low NMI with GT | Normal — motifs ≠ annotations | High NMI may indicate overfitting to labels |
| 2-10 clusters | Meaningful behavioral repertoire | 1 = no discrimination, >20 = over-segmentation |

### Linear Probe Results

**Purpose**: Measure how much discriminative information exists in raw features. This is an intentionally naive baseline.

| Dataset | Expected Range | Why |
|---------|---------------|-----|
| CalMS21 | ~60-70% (4 classes) | Simple movements, raw coordinates informative |
| NTU RGB+D | ~5-15% (60 classes) | Too many classes, needs temporal + graph modeling |
| NW-UCLA | ~30-45% (10 classes) | Fewer classes, some spatial patterns sufficient |

---

> [← MoC](../README.md) | [E2E Verification](../e2e_verification.md) | [Architecture](../architecture.md) | [Model Taxonomy](../model_taxonomy.md)

*behavior-lab v0.1 | Quick Start Guide | 2026-02-08*
