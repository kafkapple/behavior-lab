# Multi-Model Benchmark Guide

> How to evaluate and compare models across datasets in behavior-lab.

---

## Available Models by Paradigm

### Supervised (requires labels)

| Model | Type | Input Format | Best For |
|-------|------|-------------|----------|
| `infogcn` | GCN | (N,C,T,V,M) | Best accuracy, graph topology |
| `stgcn` | GCN | (N,C,T,V,M) | Standard baseline |
| `agcn` | GCN | (N,C,T,V,M) | Adaptive graph |
| `lstm` | Sequence | (T, K*D) | Temporal patterns |
| `transformer` | Sequence | (T, K*D) | Long-range dependencies |
| `mlp` | Sequence | (T, K*D) | Simplest baseline |

### Self-Supervised (no labels for pre-training)

| Model | Encoder | Method | Input |
|-------|---------|--------|-------|
| `gcn_mae` | GCN | MAE | (N,C,T,V,M) |
| `gcn_jepa` | GCN | JEPA | (N,C,T,V,M) |
| `gcn_dino` | GCN | DINO | (N,C,T,V,M) |
| `infogcn_mae` | InfoGCN | MAE | (N,C,T,V,M) |
| `infogcn_jepa` | InfoGCN | JEPA | (N,C,T,V,M) |
| `infogcn_dino` | InfoGCN | DINO | (N,C,T,V,M) |
| `interaction_mae` | Interaction | MAE | (N,C,T,V,M) |
| `interaction_jepa` | Interaction | JEPA | (N,C,T,V,M) |
| `interaction_dino` | Interaction | DINO | (N,C,T,V,M) |

### Unsupervised Discovery (no labels)

| Model | Method | Output |
|-------|--------|--------|
| `bsoid` | UMAP + HDBSCAN + RF | Cluster labels + embeddings |
| `moseq` | AR-HMM (keypoint-MoSeq) | Syllable labels |
| `subtle` | Temporal Link Embedding | Cluster labels |
| `behavemae` | Masked Autoencoder | Hierarchical embeddings |
| `clustering` | k-means / hierarchical | Cluster labels |

### External (PySKL wrappers)

| Model | Type | Source |
|-------|------|--------|
| `stgcn_pyskl` | ST-GCN | MMAction2/PySKL |
| `ctrgcn_pyskl` | CTR-GCN | PySKL |
| `aagcn_pyskl` | AA-GCN | PySKL |
| `msg3d_pyskl` | MS-G3D | PySKL |
| `poseconv3d_pyskl` | PoseConv3D | PySKL |

---

## How to Run

### Single Model (Python API)

```python
from behavior_lab.models import get_model, list_models

# List all models
print(list_models())

# Supervised graph model
model = get_model('infogcn', num_classes=60, skeleton='ntu')

# Behavior discovery
model = get_model('bsoid', fps=30)
result = model.fit_predict(keypoints)  # (T, K, D)

# Sequence classifier
model = get_model('lstm', num_classes=10, input_dim=60, epochs=50)
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

### Hydra CLI

```bash
# Single model training
python scripts/train.py model=infogcn dataset=ntu60_xsub skeleton=ntu25

# Fast debug (2 epochs, small batch)
python scripts/train.py model=stgcn training=fast_debug

# Multi-model sweep
python scripts/train.py -m model=infogcn,stgcn,agcn dataset=ntu60_xsub
```

### Discovery Benchmark

```bash
# B-SOiD on CalMS21
python scripts/benchmark.py dataset=calms21 model=bsoid

# All discovery models
python scripts/benchmark.py -m model=bsoid,moseq,subtle
```

---

## E2E Verification Models

The E2E test (`scripts/test_e2e.py`) runs quick verification:

| Dataset | Model | Purpose | Expected |
|---------|-------|---------|----------|
| CalMS21 | B-SOiD | Unsupervised discovery | NMI > 0, clusters found |
| NTU | Linear Probe | Spatial features | ~5% acc (60 classes, naive) |
| NW-UCLA | Linear Probe | Spatial features | ~40% acc (10 classes) |
| NW-UCLA | LSTM (2 epochs) | Temporal model | Pipeline works |
| NW-UCLA | Transformer (2 epochs) | Temporal model | Pipeline works |

**Note**: E2E uses minimal training (1-2 epochs) to verify pipeline, not for final accuracy.

---

## Evaluation Metrics

### Supervised

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| F1 (macro) | Per-class F1 averaged |
| Confusion Matrix | Per-class breakdown |

### Discovery / Unsupervised

| Metric | Description | Range |
|--------|-------------|-------|
| Silhouette | Cluster separation quality | [-1, 1] |
| Calinski-Harabasz | Variance ratio criterion | [0, ∞) |
| Davies-Bouldin | Cluster similarity | [0, ∞) (lower = better) |
| NMI | Agreement with ground truth | [0, 1] |
| ARI | Adjusted Rand Index | [-1, 1] |
| Hungarian Accuracy | Optimal label matching | [0, 1] |

### Behavior-specific

| Metric | Description |
|--------|-------------|
| Temporal Consistency | Label stability over time |
| Bout Duration | Mean duration per behavior |
| Entropy Rate | Behavioral complexity |
| Transition Matrix | State change probabilities |

---

## Recommended Benchmarks by Task

### Action Recognition (Human)
```
Dataset: NTU RGB+D 60 (Cross-Subject / Cross-View)
Models: infogcn > stgcn > lstm > linear_probe
Metric: Top-1 Accuracy
```

### Behavior Discovery (Animal)
```
Dataset: CalMS21
Models: bsoid, moseq, subtle, behavemae
Metric: NMI + ARI + Temporal Consistency
```

### Cross-Species Transfer
```
Pre-train: NTU (human), Fine-tune: CalMS21 (mouse)
Models: infogcn_dino (SSL pre-train → linear probe)
Metric: Transfer accuracy vs from-scratch
```

---

## Backlinks

- [Dataset Catalog](../datasets.md) — Dataset details and class names
- [Model Taxonomy](../model_taxonomy.md) — Model architecture details
- [E2E Verification](../e2e_verification.md) — Test results

---

*behavior-lab v0.1 | Created: 2026-02-08*
