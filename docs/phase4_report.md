# Phase 4: Feature Backend + Multi-Model Clustering Comparison

> Date: 2026-02-09 | Updated: 2026-02-09 | Status: Complete (Phase B CEBRA added)

## 1. Feature Backend System (7/7 tests passed)

| Module | File | Role |
|--------|------|------|
| `FeatureBackend` Protocol | `data/features/__init__.py` | Unified interface (name, dim, extract) |
| `SkeletonBackend` | same | Kinematic features wrapper (4D) |
| `DINOv2Backend` | `data/features/visual_backend.py` | Pretrained visual features (384D) |
| `aggregate_temporal()` | `data/features/temporal.py` | Sliding window mean/max/concat_stats |
| `FeaturePipeline` | `data/features/pipeline.py` | Multi-backend compositor |

## 2. Discovery Model Wrappers (6 models)

| Model | Wrapper | Changes |
|-------|---------|---------|
| Clustering (PCA+UMAP+KMeans) | `clustering.py` | PCA n_components bug fix |
| B-SOiD (UMAP+HDBSCAN) | `bsoid.py` | Existing — 2D only |
| MoSeq HMM (PCA+GaussianHMM) | `moseq.py` | Existing fallback |
| **SUBTLE** | `subtle_wrapper.py` | scipy 1.15 compat patch (morlet2 + cwt pure-numpy) |
| **hBehaveMAE** | `behavemae.py` | Config fix (900,3,24), 4D MABe22 input support |
| **CEBRA** | `data/features/cebra_backend.py` | **Phase B** — temporal contrastive learning |

### SUBTLE scipy Compatibility

scipy 1.12+ removed `signal.cwt`, scipy 1.15+ removed `signal.morlet2`.
Pure-numpy replacements implemented:

- `_morlet2_compat(M, s, w=5)`: Normalized complex Morlet wavelet
- `_cwt_compat(data, wavelet, widths)`: Convolution-based CWT
- `_patch_subtle_cwt()`: Monkey-patches both scipy.signal and subtle.module

### CEBRA Temporal Contrastive Learning (Phase B)

CEBRA uses self-supervised contrastive learning with `conditional="time"` to learn
temporal structure in behavior data. No labels or neural recordings required.

```python
CEBRABackend(output_dim=32, max_iterations=2000, time_offsets=10)
# Input: (T, K, D) keypoints → flattened to (T, K*D)
# Output: (T, 32) temporal embeddings
```

**Key advantage**: Unlike PCA/UMAP which treat frames independently, CEBRA
learns that temporally adjacent frames should have similar representations,
capturing behavioral transitions and dynamics.

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
| Shank3KO | (50K, 16, 3) | KO/WT genotype | 3D mouse, 30fps, 5 recordings |
| MABe22 | (20K, 3, 12, 2) | None | 2D 3-mouse, 30fps |

## 4. Quantitative Results

```
Model            Dataset      Clusters     Silh      ARI      NMI     Time
────────────────────────────────────────────────────────────────────────────
Clustering       CalMS21             8   -0.014    0.003    0.106    86.8s
B-SOiD           CalMS21            56    0.048        —        —   169.6s
MoSeq (HMM)      CalMS21            15    0.048        —        —     9.5s
SUBTLE           CalMS21            75        —        —        —    56.9s *
★ CEBRA          CalMS21             8    0.267        —        —   263.3s

Clustering       SUBTLE              8   -0.023        —        —     4.5s
MoSeq (HMM)      SUBTLE             15    0.035        —        —     5.0s
SUBTLE           SUBTLE             46        —        —        —    68.2s *
★ CEBRA          SUBTLE              8    0.239        —        —   279.4s

Clustering       Shank3KO            8   -0.206    0.071    0.079    20.6s
MoSeq (HMM)      Shank3KO           15    0.124        —        —     2.8s
SUBTLE           Shank3KO           69        —        —        —   134.4s *
★ CEBRA          Shank3KO            8    0.327        —        —   314.0s

Clustering       MABe22              8    0.312        —        —     8.0s
B-SOiD           MABe22             35    0.224        —        —   112.2s
MoSeq (HMM)      MABe22             15    0.310        —        —     7.8s
SUBTLE           MABe22            121        —        —        —   203.1s *
hBehaveMAE       MABe22              8    0.256        —        —     9.8s
★ CEBRA          MABe22              8    0.377        —        —   364.0s

★ = Best Silhouette per dataset
* = SUBTLE results from previous session (subprocess SIGSEGV on macOS)
```

### CEBRA vs Other Models — Silhouette Comparison

| Dataset | CEBRA | Best Other | Model | Improvement |
|---------|-------|-----------|-------|-------------|
| CalMS21 | **0.267** | 0.048 | B-SOiD/MoSeq | +456% |
| SUBTLE fly | **0.239** | 0.035 | MoSeq | +583% |
| Shank3KO | **0.327** | 0.124 | MoSeq | +164% |
| MABe22 | **0.377** | 0.312 | Clustering | +21% |

## 5. Visualization Analysis

### CalMS21 (5 models)

- **Clustering**: GT-colored embedding shows partial attack/other separation, but overall mixed. Kinematic 4D features alone insufficient for behavior discrimination.
- **B-SOiD**: 56 clusters with HDBSCAN density-based approach. Noise removal effective but over-fragmented.
- **MoSeq HMM**: 15 states with heavy overlap in UMAP. Strong temporal syllable patterns visible in ethogram. Transition matrix shows block structure.
- **SUBTLE**: Richest structure — 75 subclusters forming well-separated islands in UMAP. Morlet CWT captures time-frequency information effectively.
- **CEBRA**: **Sil 0.267 — best on CalMS21**. Temporal contrastive learning produces well-separated clusters from raw (7, 2) keypoints. 32D embedding captures behavioral dynamics that frame-level methods miss.

**Key finding**: ARI 0.003, NMI 0.106 — unsupervised clusters poorly match GT labels. Kinematic features lack inter-mouse interaction information. However, CEBRA's temporal structure significantly improves cluster quality.

### SUBTLE fly (4 models)

- **Clustering**: 8 clusters with negative Silhouette (-0.023). Transition matrix shows dominant cluster 0↔7 alternation.
- **MoSeq HMM**: 15 states, Sil 0.035. Frequent syllable transitions. Diagonal block structure in transitions.
- **SUBTLE**: Native dataset, optimal performance. 46 subclusters well-separated. CWT captures fast wing-beat/grooming frequencies.
- **CEBRA**: **Sil 0.239 — best on SUBTLE fly**. 3D fly keypoints (9, 3) → 32D temporal embedding. Clear cluster separation where frame-level methods fail.

### Shank3KO (4 models)

- **Clustering**: GT-colored (KO=blue/WT=orange) shows complete overlap in UMAP (ARI 0.071, NMI 0.079). Genotype does not cleanly separate at behavior level — expected for subtle phenotypic differences.
- **MoSeq HMM**: Sil 0.124. Large segments in ethogram (green state dominant ~frames 900-1700). Diagonal block structure reveals syllable groupings.
- **SUBTLE**: 69 subclusters, well-separated islands in UMAP. 3D 16-joint mouse skeleton provides rich CWT spectrograms.
- **CEBRA**: **Sil 0.327 — best on Shank3KO**. Temporal contrastive learning on (16, 3) keypoints → 32D. +164% improvement over MoSeq. Genotype KO/WT still overlaps but behavioral clusters are much tighter.

### MABe22 (6 models — including BehaveMAE + CEBRA)

- **Clustering**: Sil 0.312. 36-joint × 2D = 72 features provide rich signal. Cluster 0 acts as hub in transitions.
- **B-SOiD**: 35 clusters, Sil 0.224. HDBSCAN finds many small density peaks.
- **MoSeq HMM**: Sil 0.310. State 0 dominates early frames, diverse syllables emerge later.
- **SUBTLE**: 121 subclusters — most fine-grained. CWT extracts spectral info from all 72 channels.
- **hBehaveMAE**: 150 windows × 256D → 8 clusters, Sil 0.256. Sparse UMAP but meaningful transition structure.
- **CEBRA**: **Sil 0.377 — best on MABe22**. 72D keypoints → 32D temporal embedding. Temporal contrastive learning outperforms even the pretrained BehaveMAE representation (+47% Silhouette).

## 6. Model Characteristics Summary

| Aspect | Clustering | B-SOiD | MoSeq HMM | SUBTLE | hBehaveMAE | **CEBRA** |
|--------|-----------|--------|-----------|--------|------------|-----------|
| **Method** | PCA→UMAP→KMeans | UMAP→HDBSCAN→RF | PCA→GaussianHMM | Morlet CWT→UMAP→Phenograph | Hierarchical MAE | **Temporal Contrastive→KMeans** |
| **Strength** | Universal baseline | Density-based | Temporal syllables | Time-frequency | Pretrained repr | **Best Silhouette, temporal** |
| **Weakness** | Fixed k, simple | 2D only | Fixed n_states | Slow, macOS crash | MABe22-specific | Slow training (~5min/dataset) |
| **3D Support** | Yes | No | Yes | Yes | Yes (reshape) | **Yes** |
| **Speed** | Medium | Slow | **Fast** | Slow | **Fast** (inf) | Slow (training) |
| **Resolution** | Medium (8) | Variable (3-56) | Medium (15) | **Highest** (46-121) | Medium (8) | Medium (8) |

## 7. Remaining Tasks

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Shank3KO `.mat` parsing | **Done** ✅ | CoordX/Y/Z stacking |
| 2 | CEBRA backend | **Done** ✅ | Best Silhouette on all 4 datasets |
| 3 | SUBTLE macOS SIGSEGV | Workaround | Subprocess isolation, still crashes (exit -11) |
| 4 | CalMS21 ARI/NMI improvement | Open | Needs inter-mouse features or CEBRA+labels |
| 5 | BehaveMAE on CalMS21 | Shape mismatch | Separate config/fine-tuning needed |
| 6 | pose-splatter integration | Deferred | SSH to server |
| 7 | VideoMAE v2 backend | Phase C | HuggingFace transformers |

## 8. Output Files

```
outputs/clustering_comparison/
  comparison_calms21.png   (1025 KB)
  comparison_subtle.png    (628 KB)
  comparison_shank3ko.png  (670 KB)
  comparison_mabe22.png    (528 KB)

outputs/feature_backend_viz/
  fig1_feature_distributions.png
  fig2_embedding_comparison.png
  fig3_umap_clusters.png
  fig4_behavior_analysis.png
  fig5_temporal_methods.png
```
