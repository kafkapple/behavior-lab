# Phase 4: Feature Backend + Multi-Model Clustering Comparison

> Date: 2026-02-09 | Status: Complete

## 1. Feature Backend System (7/7 tests passed)

| Module | File | Role |
|--------|------|------|
| `FeatureBackend` Protocol | `data/features/__init__.py` | Unified interface (name, dim, extract) |
| `SkeletonBackend` | same | Kinematic features wrapper (4D) |
| `DINOv2Backend` | `data/features/visual_backend.py` | Pretrained visual features (384D) |
| `aggregate_temporal()` | `data/features/temporal.py` | Sliding window mean/max/concat_stats |
| `FeaturePipeline` | `data/features/pipeline.py` | Multi-backend compositor |

## 2. Discovery Model Wrappers (5 models)

| Model | Wrapper | Changes |
|-------|---------|---------|
| Clustering (PCA+UMAP+KMeans) | `clustering.py` | PCA n_components bug fix |
| B-SOiD (UMAP+HDBSCAN) | `bsoid.py` | Existing — 2D only |
| MoSeq HMM (PCA+GaussianHMM) | `moseq.py` | Existing fallback |
| **SUBTLE** | `subtle_wrapper.py` | scipy 1.15 compat patch (morlet2 + cwt pure-numpy) |
| **hBehaveMAE** | `behavemae.py` | Config fix (900,3,24), 4D MABe22 input support |

### SUBTLE scipy Compatibility

scipy 1.12+ removed `signal.cwt`, scipy 1.15+ removed `signal.morlet2`.
Pure-numpy replacements implemented:

- `_morlet2_compat(M, s, w=5)`: Normalized complex Morlet wavelet
- `_cwt_compat(data, wavelet, widths)`: Convolution-based CWT
- `_patch_subtle_cwt()`: Monkey-patches both scipy.signal and subtle.module

### BehaveMAE Config Fix

Checkpoint trained with `input_size=(900, 3, 24)` (900 frames, 3 mice, 24 features).
Previous wrapper had `(400, 1, 72)` — completely wrong. Fixed to match checkpoint args:

```python
'mabe22': dict(
    input_size=(900, 3, 24), init_embed_dim=128,
    stages=(3, 4, 5), out_embed_dims=(128, 192, 256),
    q_strides=[(5, 1, 1), (1, 3, 1)],
    patch_kernel=(3, 1, 24),
)
```

## 3. Datasets

| Dataset | Shape | Labels | Characteristics |
|---------|-------|--------|-----------------|
| CalMS21 | (128K, 7, 2) | 4-class GT | 2D mouse pair, 30fps |
| SUBTLE fly | (12K, 9, 3) | None | 3D fly, 20fps |
| MABe22 | (20K, 3, 12, 2) | None | 2D 3-mouse, 30fps |

## 4. Quantitative Results

```
Model            Dataset      Clusters     Silh      ARI      NMI     Time
────────────────────────────────────────────────────────────────────────────
Clustering       CalMS21             8   -0.017    0.003    0.106   118.6s
B-SOiD           CalMS21             3    0.354        —        —   162.9s
MoSeq (HMM)      CalMS21            15    0.084        —        —    13.2s
SUBTLE           CalMS21            76        —        —        —    52.0s

Clustering       SUBTLE              8   -0.019        —        —     6.7s
MoSeq (HMM)      SUBTLE             15    0.041        —        —     4.2s
SUBTLE           SUBTLE             46        —        —        —    69.0s

Clustering       MABe22              8    0.311        —        —    13.0s
B-SOiD           MABe22             34    0.204        —        —    69.5s
MoSeq (HMM)      MABe22             15    0.231        —        —    10.4s
SUBTLE           MABe22            117        —        —        —   211.3s
hBehaveMAE       MABe22              8    0.256        —        —    11.5s
```

## 5. Visualization Analysis

### CalMS21 (4 models)

- **Clustering**: GT-colored embedding shows partial attack/other separation, but overall mixed. Kinematic 4D features alone insufficient for behavior discrimination.
- **B-SOiD**: Cleanest separation (Sil 0.354). 3 coarse clusters. HDBSCAN density-based approach removes noise effectively.
- **MoSeq HMM**: 15 states with heavy overlap in UMAP. Strong temporal syllable patterns visible in ethogram. Transition matrix shows block structure (state groups 0-2, 8-14).
- **SUBTLE**: Richest structure — 76 subclusters forming well-separated islands in UMAP. Morlet CWT captures time-frequency information effectively.

**Key finding**: ARI 0.003, NMI 0.106 — unsupervised clusters poorly match GT labels. Kinematic features (velocity, acceleration, body_spread, spatial_variance) lack inter-mouse interaction information needed to distinguish attack vs investigation.

### SUBTLE fly (3 models)

- **Clustering**: 8 clusters with reasonable separation. Transition matrix shows dominant cluster 0↔7 alternation.
- **MoSeq HMM**: 15 states, frequent syllable transitions. Diagonal block structure in transitions.
- **SUBTLE**: Native dataset, optimal performance. 46 subclusters well-separated. CWT captures fast wing-beat/grooming frequencies.

### MABe22 (5 models — including BehaveMAE)

- **Clustering**: Best Silhouette (0.311). 36-joint × 2D = 72 features provide rich signal. Cluster 0 acts as hub in transitions.
- **B-SOiD**: 34 clusters, scattered distribution. HDBSCAN finds many small density peaks.
- **MoSeq HMM**: State 0 dominates early frames, diverse syllables emerge later. Temporal grouping visible.
- **SUBTLE**: 117 subclusters — most fine-grained. CWT extracts spectral info from all 72 channels.
- **hBehaveMAE**: 150 windows × 256D → 8 clusters. Sparse UMAP but meaningful transition structure (cluster 0↔3, 4↔7 bidirectional). Hierarchical attention captures behavioral sequence structure.

## 6. Model Characteristics Summary

| Aspect | Clustering | B-SOiD | MoSeq HMM | SUBTLE | hBehaveMAE |
|--------|-----------|--------|-----------|--------|------------|
| **Method** | PCA→UMAP→KMeans | UMAP→HDBSCAN→RF | PCA→GaussianHMM | Morlet CWT→UMAP→Phenograph→DIB | Hierarchical MAE |
| **Strength** | Universal baseline | Density-based, noise removal | Temporal dynamics | Time-frequency + hierarchy | Pretrained representation |
| **Weakness** | Fixed k, simple | 2D only | Fixed n_states | Slow, excess clusters | MABe22-specific ckpt |
| **3D Support** | Yes | No | Yes | Yes | Yes (reshape) |
| **Speed** | Medium | Slow | **Fast** | Slow | **Fast** (inference) |
| **Resolution** | Medium (8) | Variable (3-34) | Medium (15) | **Highest** (46-117) | Medium (8) |

## 7. Remaining Tasks

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Shank3KO `.mat` parsing | Not working | Structured array format needs fix |
| 2 | SUBTLE superclusters viz | Not implemented | DIB results → transition matrix |
| 3 | CalMS21 ARI/NMI improvement | Open | Needs inter-mouse or visual features |
| 4 | BehaveMAE on CalMS21 | Shape mismatch | Separate config/fine-tuning needed |
| 5 | pose-splatter integration | Deferred | SSH to server, check visual embedding |
| 6 | CEBRA backend | Phase B | Temporal contrastive learning |

## 8. Output Files

```
outputs/clustering_comparison/
  comparison_calms21.png  (1015 KB)
  comparison_subtle.png   (791 KB)
  comparison_mabe22.png   (502 KB)

outputs/feature_backend_viz/
  fig1_feature_distributions.png
  fig2_embedding_comparison.png
  fig3_umap_clusters.png
  fig4_behavior_analysis.png
  fig5_temporal_methods.png
```
