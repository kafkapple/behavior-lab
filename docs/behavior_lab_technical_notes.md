# behavior-lab: Technical Research Notes

> Modular skeleton-based behavior analysis platform for humans and animals.
> Supports supervised, self-supervised, and unsupervised paradigms via unified factory API.

---

## 1. Platform Overview

### 1.1 Design Philosophy

- **Canonical tensor**: `(T, K, D)` — all data normalized to this format
  - T = temporal frames, K = keypoints/joints, D = spatial dims (2 or 3)
- **Graph tensor**: `(N, C, T, V, M)` — auto-converted for GCN models only
- **Single factory**: `get_model('name')` dispatches across all paradigms
- **Skeleton-agnostic**: 10+ skeleton definitions, same pipeline for human/animal
- **Self-contained HTML reports**: base64-embedded figures, zero external dependencies

### 1.2 Module Architecture

```
behavior-lab/
├── src/behavior_lab/
│   ├── core/              # Skeleton registry, graph adjacency, tensor formats
│   │   ├── skeleton.py    #   10+ SkeletonDefinition (NTU, COCO, CalMS21, SUBTLE, ...)
│   │   ├── graph.py       #   Spatial partition (3, V, V): identity/inward/outward
│   │   ├── types.py       #   BehaviorSequence, ClusteringResult, Metrics
│   │   └── tensor_format.py
│   ├── data/
│   │   ├── loaders/       # 7 dataset loaders (CalMS21, NTU, UCLA, SUBTLE, Shank3KO, Rat7M, MABe22)
│   │   ├── features/      # SkeletonBackend (kinematic), CEBRABackend (contrastive), temporal agg
│   │   └── preprocessing/ # Pipeline: Confidence→Interpolate→Outlier→Smooth→Normalize
│   ├── models/
│   │   ├── graph/         # GCN family: InfoGCN, ST-GCN, AGCN + PySKL wrappers
│   │   ├── sequence/      # MLP, LSTM, Transformer, RuleBased
│   │   ├── ssl/           # {GCN, InfoGCN, Interaction} × {MAE, JEPA, DINO}
│   │   └── discovery/     # B-SOiD, MoSeq, Keypoint-MoSeq, VAME, SUBTLE, BehaveMAE, CEBRA, KMeans
│   ├── experiments/       # pipeline.run_comparison (unified cross-species) + discovery
│   ├── evaluation/        # Classification, clustering, behavior metrics
│   └── visualization/     # Skeleton plots/GIFs, embeddings, comparison.render_comparison_report
│                          #   (+ ground-truth ARI/NMI, agreement matrix) + render_cluster_gallery
└── scripts/
    ├── isolated_run.py            # run one heavy method (keypoint-MoSeq/VAME) in its own conda env
    ├── viz_feature_backends.py    # SkeletonBackend visualization (5 panels)
    └── test_e2e.py                # End-to-end pipeline verification
```
> Benchmark + report are now library calls (`experiments.run_comparison` +
> `visualization.render_comparison_report`); the old `compare_clustering.py` /
> `generate_cluster_report.py` scripts were removed (260705).

---

## 2. Datasets

### 2.1 Dataset Catalog

| Dataset | Species | Joints | Dim | Classes | Frames | Source Format | Skeleton Key |
|---------|---------|--------|-----|---------|--------|---------------|--------------|
| **CalMS21** | 2 mice (resident-intruder) | 7 × 2 animals | 2D | 4 (attack, investigation, mount, other) | 19K sequences × 64f | NPZ | `calms21` |
| **SUBTLE** | 1 mouse (spontaneous) | 9 | 3D | Unsupervised | ~60K frames/file | CSV | `subtle` |
| **Shank3KO** | 1 mouse (KO vs WT) | 16 | 3D | 2 genotypes (11 behaviors) | ~50K frames | MAT (SLEAP) | `shank3ko` |
| **MABe22** | 3 mice (triplet) | 12 × 3 = 36 | 2D | Unsupervised | ~180K frames | NPY | `mabe22` |
| **NTU RGB+D** | 1-2 humans | 25 × 2 persons | 3D | 60/120 actions | 56K/114K samples | .skeleton/.npz | `ntu` |
| **NW-UCLA** | 1 human, 3 views | 20 | 3D | 10 actions | 1.5K samples | NPZ | `ucla` |
| **Rat7M** | 1 rat (mocap) | 20 | 3D | Unlabeled | ~7M frames | MAT/HDF5 | `rat7m` |

### 2.2 Data Loading Interface

```python
loader = get_loader("calms21", data_dir="data/calms21", skeleton_name="calms21")
sequences = loader.load_split("train")  # → list[BehaviorSequence]
# BehaviorSequence.keypoints: (T, K, D)
# BehaviorSequence.labels: (T,) or None
```

### 2.3 Coordinate Systems

| Convention | Datasets | Matplotlib Display |
|------------|----------|--------------------|
| **Y-up** (Kinect) | NTU, UCLA, DLC, CalMS21, MABe22 | Swapped to Z-up: `(x,y,z) → (x,z,y)` |
| **Z-up** (mocap) | SUBTLE, Shank3KO, Rat7M | Native — no swap needed |

- Auto-detected via `_infer_coord_up(skeleton)` based on skeleton name

---

## 3. Skeleton Registry

### 3.1 SkeletonDefinition Structure

```python
@dataclass
class SkeletonDefinition:
    name: str                          # Unique ID
    num_joints: int                    # K
    joint_names: list[str]             # Ordered names
    joint_parents: list[int]           # Parent idx (-1 for root)
    edges: list[tuple[int, int]]       # Skeleton graph
    symmetric_pairs: list[tuple]       # L-R pairs (augmentation)
    num_channels: int                  # 2 or 3
    body_parts: dict[str, list[int]]   # Named joint groups
    center_joint: int                  # Normalization reference
    num_persons: int                   # Max agents per scene
```

### 3.2 Key Methods

- `get_adjacency_matrix()` → `(V, V)` binary with self-loops
- `get_normalized_adjacency()` → `D^{-1/2} A D^{-1/2}`
- `subset(joint_names)` → Sub-skeleton with remapped indices
- `get_inward_edges()` / `get_outward_edges()` → Directed edge lists

### 3.3 Graph Spatial Partition (for GCN)

```python
graph = Graph("ntu", labeling_mode="spatial")
# A: (3, V, V) = [Identity | Centripetal (inward) | Centrifugal (outward)]
```

- Based on ST-GCN (Yan et al., AAAI 2018) spatial partition strategy
- Used by all graph-based models (InfoGCN, ST-GCN++, CTR-GCN, etc.)

---

## 4. Preprocessing Pipeline

### 4.1 Pipeline Steps

```
Raw keypoints (T, K, D)
  → ConfidenceFilter  — Replace low-confidence joints with NaN (threshold=0.3)
  → Interpolator      — Linear interpolation across NaN gaps (max_gap=10)
  → OutlierRemover    — Remove velocity spikes (threshold=50)
  → TemporalSmoother  — Moving average or Savitzky-Golay (window=5)
  → Normalizer        — Center on reference joint, optional scaling
  → Clean (T, K, D)
```

### 4.2 Augmentation (SSL Training)

| Mode | Rotation | Scale | Noise |
|------|----------|-------|-------|
| weak | ±15° | ±10% | σ=0.01 |
| medium | ±30° | ±20% | σ=0.02 |
| strong | ±45° | ±30% | σ=0.03 |

---

## 5. Feature Extraction

### 5.1 SkeletonBackend (Kinematic Features)

```python
backend = SkeletonBackend(fps=30.0, smooth_window=5)
features = backend.extract(keypoints)  # (T, K, D) → (T, 4)
```

| Feature | Description | Dim |
|---------|-------------|-----|
| velocity | Centroid displacement per frame | 1 |
| acceleration | Velocity derivative | 1 |
| body_spread | Max std of keypoint positions | 1 |
| spatial_variance | Variance of all coordinates | 1 |

### 5.2 CEBRABackend (Temporal Contrastive)

```python
backend = CEBRABackend(output_dim=32, max_iterations=5000, time_offsets=10)
embeddings = backend.extract(keypoints)  # (T, K, D) → (T, 32)
```

- Self-supervised: Temporal contrastive loss
- Preserves temporal structure via offset-based negative sampling
- Cosine distance metric in embedding space

### 5.3 Temporal Aggregation

```python
aggregate_temporal(features, window_size=30, stride=15, method="mean")
# (T, D) → (N_seg, D)           if method="mean"
# (T, D) → (N_seg, 4D)          if method="concat_stats" (mean|std|min|max)
```

---

## 6. Discovery Models (Unsupervised / Self-Supervised)

### 6.1 Model Comparison Matrix

| Model | Algorithm | Input | Temporal | Train Type | Key Dep |
|-------|-----------|-------|----------|------------|---------|
| **Clustering** | PCA → UMAP → KMeans | (N, D) features | None | Unsupervised | sklearn, umap |
| **B-SOiD** | Spatiotemporal → UMAP → HDBSCAN → RF | (T, K, **2**) 2D only | Binning (10fps) | Unsupervised + RF | umap, hdbscan |
| **MoSeq (HMM)** | PCA → Gaussian HMM → Viterbi | (T, K, D) any | Native (AR-HMM) | Self-supervised | hmmlearn |
| **SUBTLE** | Morlet CWT → UMAP → Phenograph | List[(T, K, **3**)] 3D only | Wavelet spectral | Unsupervised | subtle, scipy |
| **hBehaveMAE** | Hierarchical Masked Autoencoder | (T, A, J, D) multi-agent | Patch masking (3f stride) | Self-supervised (pretrained) | torch |
| **CEBRA** | Temporal Contrastive Learning | (T, D) any | Offset-based contrast | Self-supervised | cebra |

### 6.2 Detailed Model Specs

#### 6.2.1 Clustering Baseline

```
Pipeline: Features → StandardScaler → PCA (95% var, max 50) → UMAP(2D) → KMeans(k)
```

- **Input**: `(N, D)` pre-extracted feature matrix
- **Output**: `{"labels": (N,), "embedding_2d": (N,2), "n_clusters": int}`
- **Params**: `n_clusters=5`, `pca_variance=0.95`, `use_umap=True`
- **Limitation**: Requires pre-extracted features; no temporal modeling

#### 6.2.2 B-SOiD (Behavioral Segmentation of Deeplabcut)

```
Pipeline: Keypoints → Displacement + Pairwise Dist + Angular Change
        → Boxcar Smooth (60ms) → Temporal Binning (10fps)
        → StandardScaler → UMAP(11D) → HDBSCAN → RF Classifier
```

- **Input**: `(T, K, 2)` — **2D skeleton only**
- **Output**: `ClusteringResult` with labels, embeddings, n_clusters, features
- **Feature extraction** (3 types):
  - Displacement: `||p_{t+1} - p_t||` per keypoint → (T-1, K)
  - Pairwise distances: `||p_i - p_j||` for all pairs → (T, K*(K-1)/2)
  - Angular change: `arccos(dot(edge_t, edge_{t+1}))` → (T-1, K*(K-1)/2)
- **Transfer learning**: RF classifier trained on discovered labels
- **Params**: `fps=30, n_neighbors=60, min_dist=0.0, umap_dim=11, min_cluster_size=30`
- **Ref**: Hsu & Bhatt, "B-SOiD" (Nature Communications 2021)

#### 6.2.3 Keypoint-MoSeq / PCA-HMM Fallback

**Full MoSeq** (when keypoint-moseq installed):
```
Pipeline: (T,K,D) → Format dict → PCA(10D) → AR-HMM fit → SLDS fit → Syllable extract
```

**Fallback** (PCA-HMM):
```
Pipeline: (T,K,D) → Flatten(T, K*D) → PCA(10D) → Gaussian HMM(20 states) → Viterbi
```

- **Input**: `(T, K, D)` — any dimensionality
- **Output**: `ClusteringResult` with syllable/state labels
- **Params (full)**: `latent_dim=10, num_iters=50, kappa=1e6`
- **Params (fallback)**: `n_components=10, n_states=20, n_iter=50`
- **Ref**: Wiltschko et al., "MoSeq" (Neuron 2015); Datta lab

#### 6.2.4 SUBTLE (Spectrogram-UMAP Temporal Link Embedding)

```
Pipeline: List[(T_i,K,3)] → Center → Morlet CWT (spectral features)
        → UMAP → Phenograph (sub/superclusters)
        → Transition probabilities + Retention metrics
```

- **Input**: `List[(T_i, K, 3)]` — **3D skeleton sequences**
- **Output**: embeddings, subclusters, superclusters, transition matrix, retention
- **Params**: `fps=20, n_train_frames=120000, embedding_method='umap'`
- **Note**: scipy >= 1.12 removed `signal.cwt` — monkey-patch applied
- **Ref**: Kwon et al., "SUBTLE" (github.com/jeakwon/subtle)

#### 6.2.5 hBehaveMAE (Hierarchical Masked Autoencoder)

```
Architecture: (B,1,T,A,F) → Conv3d patches → 3-level Transformer encoder
            → Masked token reconstruction (self-supervised)
            → Multi-scale feature extraction

Level 0: embed_dim=128, 3 layers, 2 heads
Level 1: embed_dim=192, 4 layers, 3 heads
Level 2: embed_dim=256, 5 layers, 4 heads

Patch: kernel=(3,1,24), stride=(3,1,24) → 3-frame temporal patches
```

- **Input**: `(T, n_agents, J, D)` — multi-agent keypoints
  - MABe22: `(900, 3, 12, 2)` → reshape to `(B, 1, 900, 3, 24)`
- **Output (encode)**: `(N_tokens, 256)` — final-level features
- **Output (hierarchical)**: `{"level_0": (N,128), "level_1": (N,192), "level_2": (N,256)}`
- **Pre-trained only**: Checkpoint required (`hBehaveMAE_MABe22.pth`)
- **Sliding window**: 900 frames, stride 450 → pool to per-window (256,) vector
- **Ref**: Azabou et al., "BehaveMAE" (NeurIPS 2024)

#### 6.2.6 CEBRA (Consistent Embeddings of high-dimensional Recordings using Auxiliary variables)

```
Pipeline: (T, K*D) → Temporal Contrastive Model
        → Offset-based positive pairs (|Δt| ≤ time_offsets)
        → Cosine distance loss
        → (T, output_dim) embeddings
```

- **Input**: `(T, D)` or `(T, K, D)` auto-flattened — **any dimensionality**
- **Output**: `(T, output_dim)` temporal embeddings
- **Params**: `output_dim=32, max_iterations=5000, time_offsets=10, batch_size=512, temperature=1.0`
- **Key insight**: Temporally nearby frames → similar embeddings (preserves dynamics)
- **Ref**: Schneider et al., "CEBRA" (Nature 2023)

### 6.3 Shared Interface (BehaviorClusterer Protocol)

```python
# All discovery models implement:
model.fit(data)           → ClusteringResult / fitted state
model.predict(data)       → (T,) labels
model.fit_predict(data)   → ClusteringResult
model.get_embeddings()    → (T, D_embed) array
model.save(path)          → serialized state
model.load(path)          → restored model
```

### 6.4 Model Selection Guide

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| **2D mouse, fast** | B-SOiD | Spatiotemporal features + HDBSCAN, no training |
| **3D tracking, spectral** | SUBTLE | Wavelet captures frequency dynamics |
| **Temporal dynamics, any dim** | CEBRA | Contrastive learning preserves temporal structure |
| **Multi-agent** | hBehaveMAE | Hierarchical + multi-agent native |
| **Quick baseline** | Clustering | PCA→UMAP→KMeans, no dependencies |
| **Syllable discovery** | MoSeq (HMM) | AR-HMM models transitions natively |

---

## 7. Evaluation Metrics

### 7.1 Classification Metrics

```python
compute_classification_metrics(y_true, y_pred, class_names)
→ ClassificationMetrics(accuracy, f1_macro, f1_per_class, confusion_matrix)
```

### 7.2 Clustering Metrics

| Metric | Type | Range | Needs GT |
|--------|------|-------|----------|
| **Silhouette** | Internal | [-1, 1] ↑ | No |
| **Calinski-Harabasz** | Internal | [0, ∞) ↑ | No |
| **Davies-Bouldin** | Internal | [0, ∞) ↓ | No |
| **NMI** | External | [0, 1] ↑ | Yes |
| **ARI** | External | [-1, 1] ↑ | Yes |
| **V-measure** | External | [0, 1] ↑ | Yes |
| **Hungarian Accuracy** | External | [0, 1] ↑ | Yes |

### 7.3 Behavior Metrics

| Metric | Description |
|--------|-------------|
| **Bout durations** | Mean duration per behavior class (seconds) |
| **Transition matrix** | (C, C) empirical transition probabilities |
| **Temporal consistency** | Fraction of same-label consecutive frames |
| **Entropy rate** | Information-theoretic label entropy |

---

## 8. Visualization & Reporting

### 8.1 Skeleton Visualization

- `plot_skeleton()` — Static frame, body-part coloring, 2D/3D
- `animate_skeleton()` — GIF/MP4 animation via FuncAnimation
- `plot_skeleton_comparison()` — Side-by-side (e.g., raw vs preprocessed)
- Auto coordinate conversion (Y-up ↔ Z-up)
- Per-joint IQR outlier clipping (Tukey's fences, factor=3.0)

### 8.2 Analysis Plots

- `plot_embedding()` — UMAP/PCA 2D scatter, colored by label
- `plot_transition_matrix()` — Behavior transition heatmap
- `plot_temporal_raster()` — Timeline ethogram
- `plot_bout_duration()` — Bar chart of behavior durations
- `plot_behavior_dendrogram()` — Hierarchical behavior similarity

### 8.3 HTML Report Generator

```python
generate_pipeline_report(report_data, output_path, title)
# Self-contained HTML: base64-embedded images, tab navigation
# Supports: data tables, metric cards, comparison images, per-cluster GIFs
```

### 8.4 Video Overlay

```python
overlay_keypoints_on_video(video_path, keypoints, skeleton, output_path)
# Renders skeleton on raw video frames (OpenCV backend)
```

---

## 9. Clustering Comparison Benchmark

### 9.1 Benchmark Design

- **6 models** × **4 datasets** = 24 model-dataset combinations
- Per combination: UMAP embedding plot, ethogram, transition matrix
- Summary: Silhouette, ARI, NMI, runtime

### 9.2 Results Summary (from Phase 4)

| Dataset | Best Model | Silhouette | Note |
|---------|-----------|------------|------|
| CalMS21 | Clustering / CEBRA | ~0.15-0.25 | 4 GT classes, 2D |
| SUBTLE | Clustering / CEBRA | ~0.10-0.30 | No GT, 3D |
| Shank3KO | Clustering / CEBRA | ~0.15-0.20 | KO/WT genotypes, 3D |
| MABe22 | CEBRA / hBehaveMAE | ~0.10-0.25 | 3 mice, 2D |

### 9.3 Model Compatibility Matrix

| Model | CalMS21 (2D) | SUBTLE (3D) | Shank3KO (3D) | MABe22 (2D) |
|-------|:---:|:---:|:---:|:---:|
| Clustering | Y | Y | Y | Y |
| B-SOiD | Y | Skip (3D) | Skip (3D) | Y |
| MoSeq HMM | Y | Y | Y | Y |
| SUBTLE | Skip (2D) | Y | Y | Skip (2D) |
| hBehaveMAE | Skip | Skip | Skip | Y (checkpoint) |
| CEBRA | Y | Y | Y | Y |

---

## 10. Output Artifacts

### 10.1 feature_backend_viz/

| File | Content |
|------|---------|
| `fig1_feature_distributions.png` | Box plots: Speed, Acceleration, Body Spread, Spatial Variance × 4 classes |
| `fig2_embedding_comparison.png` | Frame-level vs Segment-level UMAP |
| `fig3_umap_clusters.png` | Unsupervised clusters vs GT labels |
| `fig4_behavior_analysis.png` | Ethogram + Transition Matrix + Dendrogram |
| `fig5_temporal_methods.png` | Temporal aggregation comparison (mean/max/concat_stats) |

### 10.2 clustering_comparison/

| File | Content |
|------|---------|
| `comparison_calms21.png` | 6-model comparison (UMAP + ethogram + transitions) |
| `comparison_subtle.png` | Same for SUBTLE dataset |
| `comparison_shank3ko.png` | Same for Shank3KO dataset |
| `comparison_mabe22.png` | Same for MABe22 dataset |
| `cache/*.npz` | Cached model labels/embeddings for report generation |
| `gifs/` | Per-cluster skeleton animation GIFs |
| `report.html` | Self-contained HTML report with all above |

---

## 11. Key References

| Model/Method | Reference |
|-------------|-----------|
| ST-GCN | Yan et al., "Spatial Temporal Graph Convolutional Networks" (AAAI 2018) |
| InfoGCN | Chi et al., "InfoGCN: Representation Learning for Human Skeleton-based Action Recognition" (CVPR 2022) |
| B-SOiD | Hsu & Bhatt, "B-SOiD: An Open-Source Unsupervised Algorithm for Identification and Fast Prediction of Behaviors" (Nature Communications 2021) |
| MoSeq | Wiltschko et al., "Mapping Sub-Second Structure in Mouse Behavior" (Neuron 2015) |
| SUBTLE | Kwon et al., "SUBTLE: An Unsupervised Platform with Temporal Link Embedding that Maps Animal Behavioral Repertoire" |
| BehaveMAE | Azabou et al., "BehaveMAE: Hierarchical Masked Autoencoders for Multi-Agent Behavior" (NeurIPS 2024) |
| CEBRA | Schneider et al., "Learnable Latent Embeddings for Joint Behavioural and Neural Analysis" (Nature 2023) |
| CalMS21 | Sun et al., "CalMS21: Multi-Agent Mouse Social Behavior" (NeurIPS 2021 D&B) |
| MABe22 | Sun et al., "MABe22: Multi-Agent Behavior Challenge" (ICML 2023) |
| NTU RGB+D | Shahroudy et al., "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis" (CVPR 2016) |

---

*behavior-lab v0.1.0 | Technical Research Notes | 2026-02-09*
