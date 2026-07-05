# SUBTLE Pipeline Reference

Behavior clustering pipeline based on SUBTLE (Kwon et al., IJCV 2024).

---

## 1. Algorithm Overview

```
keypoints (T, K, 3)
  → center + flatten (T, K*D)
  → Morlet CWT spectrogram (time-frequency)
  → np.concatenate across joints
  → PCA (100 components)
  → UMAP (n_neighbors=50, n_components=2)
  → Phenograph clustering → subclusters
  → DIB (Divisive Information Bottleneck) → superclusters
  → Transition matrix + retention metrics
```

### Stage Details

| Stage | Method | Key Params | Output |
|-------|--------|------------|--------|
| **Preprocess** | Global mean subtraction + flatten | — | `(T, K*D)` |
| **Spectrogram** | Morlet CWT (continuous wavelet) | `fs`, `omega`, `n_channels` | `(n_freq, T)` per channel |
| **Dim Reduction** | PCA → UMAP | `n_components=100` → `50 neighbors, 2D` | `(T, 2)` embedding |
| **Clustering** | Phenograph (Louvain on k-NN) | data-driven k | subclusters `(T,)` |
| **Hierarchy** | DIB (info bottleneck) | — | superclusters `(T,)` |
| **Metrics** | Transition probability, retention | — | `(C, C)` matrix, `(C,)` retention |

### Compatibility Patches (scipy >= 1.12)

`scipy.signal.cwt` (removed 1.12) and `scipy.signal.morlet2` (removed 1.15) are monkey-patched with numpy-based implementations in `subtle_wrapper.py:15-77`. Additionally, UMAP and Phenograph are forced to `n_jobs=1` to prevent macOS SIGSEGV (loky multiprocessing issue).

---

## 2. Implementation Map

### Core Modules

| Module | Path | Purpose |
|--------|------|---------|
| **SUBTLE wrapper** | `src/behavior_lab/models/discovery/subtle_wrapper.py` | CWT + UMAP + Phenograph |
| **B-SOiD** | `src/behavior_lab/models/discovery/bsoid.py` | Displacement + HDBSCAN + RF |
| **MoSeq** | `src/behavior_lab/models/discovery/moseq.py` | AR-HMM syllable discovery |
| **BehaveMAE** | `src/behavior_lab/models/discovery/behavemae.py` | Hierarchical MAE (pretrained) |
| **Baseline** | `src/behavior_lab/models/discovery/clustering.py` | PCA → UMAP → KMeans |

### Data Layer

| Module | Path | Purpose |
|--------|------|---------|
| **Types** | `src/behavior_lab/core/types.py` | `BehaviorSequence`, `ClusteringResult` |
| **Skeleton registry** | `src/behavior_lab/core/skeleton.py` | 10+ skeleton definitions |
| **SUBTLE loader** | `src/behavior_lab/data/loaders/subtle.py` | CSV/NPY/NPZ → `(T, 9, 3)` |
| **SkeletonBackend** | `src/behavior_lab/data/features/features.py` | 4D kinematic features |
| **CEBRABackend** | `src/behavior_lab/data/features/cebra_backend.py` | Temporal contrastive 32D |

### Evaluation

| Module | Path | Metrics |
|--------|------|---------|
| **ClusterMetrics** | `evaluation/evaluator.py` | Silhouette, CH, DB, NMI, ARI, V-measure, Hungarian acc |
| **BehaviorMetrics** | `evaluation/evaluator.py` | Bout duration, transition matrix, temporal consistency, entropy rate |
| **Linear probe** | `evaluation/evaluator.py` | Logistic regression on frozen features |

### Visualization

| Module | Path | Purpose |
|--------|------|---------|
| **HTML report** | `visualization/html_report.py` | Self-contained report with base64 images |
| **Skeleton anim** | `visualization/skeleton.py` | GIF generation with IQR outlier clipping |
| **Embedding plot** | `visualization/embedding.py` | UMAP/PCA scatter |

---

## 3. Model Comparison (6 models × 4 datasets)

### Models

| Model | Input | Temporal | Clustering | Transfer |
|-------|-------|----------|-----------|----------|
| **Clustering** | `(T,K,D)` any | None | KMeans | — |
| **B-SOiD** | `(T,K,2)` only | 10fps binning | HDBSCAN | RandomForest |
| **MoSeq** | `(T,K,D)` any | AR-HMM native | HMM states | — |
| **SUBTLE** | `[(T,K,3)]` only | Morlet CWT | Phenograph | Direct labels |
| **BehaveMAE** | `(T,3,12,2)` | Patch masking | KMeans | Frozen encoder |
| **CEBRA** | `(T,D)` any | Offset contrast | KMeans | Frozen encoder |

### Datasets

| Dataset | Shape | Labels | Species |
|---------|-------|--------|---------|
| CalMS21 | `(T, 7, 2)` | 4 classes | Mouse pair (2D) |
| SUBTLE fly | `(T, 9, 3)` | Unsupervised | Fly (3D) |
| Shank3KO | `(T, 16, 3)` | 2 genotypes | Mouse (3D SLEAP) |
| MABe22 | `(T, 3, 12, 2)` | Unsupervised | 3 mice (2D) |

### Phase 4 Results (key numbers)

| Model | CalMS21 Sil | CalMS21 ARI | SUBTLE fly clusters | MABe22 Sil |
|-------|------------|-------------|---------------------|-----------|
| Clustering | -0.017 | 0.003 | 8 | **0.311** |
| B-SOiD | **0.354** | — | — | 0.204 |
| MoSeq | 0.084 | — | 15 | 0.231 |
| SUBTLE | — | — | 46 sub / 30 super | — |
| BehaveMAE | — | — | — | 0.256 |

---

## 4. Reusable Components (Modularization)

### Layer 1: Core (zero external deps beyond numpy/scipy/sklearn)

Immediately portable to any project:

```python
# evaluation/evaluator.py — drop-in
from behavior_lab.evaluation.evaluator import (
    compute_cluster_metrics,    # Silhouette, ARI, NMI, ...
    compute_behavior_metrics,   # Bout duration, transitions, entropy
    ClusterMetrics,
    BehaviorMetrics,
)

# core/types.py — data contracts
from behavior_lab.core.types import ClusteringResult, BehaviorSequence
```

**Dependencies**: numpy, scipy, scikit-learn only.

### Layer 2: Visualization (+ matplotlib, imageio)

```python
# Skeleton animation with outlier clipping
from behavior_lab.visualization.skeleton import (
    animate_skeleton,       # keypoints → GIF
    clip_outliers_iqr,      # Per-joint Tukey fence
)

# Self-contained HTML report
from behavior_lab.visualization.html_report import generate_pipeline_report
```

### Layer 3: Discovery Models (heavy deps)

Each model is independently importable with lazy dependencies:

```python
from behavior_lab.models.discovery.subtle_wrapper import SUBTLE   # +subtle, phenograph, umap
from behavior_lab.models.discovery.bsoid import BSOiD              # +hdbscan, umap
from behavior_lab.models.discovery.moseq import KeypointMoSeq      # +hmmlearn
```

---

## 5. Improvements (Implemented)

### P1: Interface Consistency ✓

`SUBTLE.fit()` now returns `ClusteringResult` directly. `fit_raw()` for dict access:

```python
def fit(self, sequences) -> ClusteringResult:  # not dict
def fit_raw(self, sequences) -> dict:          # low-level access
```

### P2: MorletCWTBackend ✓

Standalone `FeatureBackend` at `data/features/morlet_backend.py`:

```python
from behavior_lab.data.features.morlet_backend import MorletCWTBackend
backend = MorletCWTBackend(fs=30.0, n_channels=25)
features = backend.extract(keypoints)  # (T, K, D) → (T, n_channels * K * D)
```

Enables SUBTLE-style spectral features with any clusterer (KMeans, HDBSCAN, etc.).

### P3: Subprocess Isolation ✓

`fit_predict(isolate=True)` runs SUBTLE in a subprocess (prevents macOS SIGSEGV):

```python
model = SUBTLE(config=SUBTLEConfig(isolate=True, timeout=300))
result = model.fit_predict([kp])  # safe subprocess execution
```

### P4: SUBTLEConfig ✓

```python
from behavior_lab.models.discovery.subtle_wrapper import SUBTLEConfig
cfg = SUBTLEConfig(fps=30, use_superclusters=True, isolate=True)
cfg = SUBTLEConfig.from_dict({"fps": 30, "custom_param": 42})  # extra → .extra
```

### P5: Behavior Manifold (deferred)

SUBTLE paper's behavior manifold (n semantic axes per animal) requires domain-specific annotation. Consider post-hoc UMAP axis interpretation or supervised probing per axis.

---

## 6. Quick Start (New Project)

```python
import numpy as np
from behavior_lab.models.discovery.subtle_wrapper import SUBTLE
from behavior_lab.evaluation.evaluator import Evaluator

# 1. Load keypoints (T, K, 3)
keypoints = np.load("keypoints.npy")

# 2. Cluster
model = SUBTLE(fps=30)
result = model.fit_predict([keypoints], use_superclusters=True)

# 3. Evaluate
evaluator = Evaluator()
cluster_m = evaluator.evaluate_clusters(result.embeddings, result.labels)
behavior_m = evaluator.evaluate_behavior(result.labels, fps=30.0)
evaluator.print_report(cluster_m)
evaluator.print_report(behavior_m)

# 4. Visualize
from behavior_lab.visualization.html_report import generate_pipeline_report
generate_pipeline_report(report_data, "output/report.html")
```

---

*behavior-lab SUBTLE Pipeline Reference | 2026-03-22*
