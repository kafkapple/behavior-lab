# Evaluation Framework

> *Backlinks: [Overview](../overview.md) | [SSL Methods](ssl_methods.md) | [Cross-Species](cross_species.md)*

## Metrics by Paradigm

### Supervised Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | correct / total | Overall performance |
| **F1 (macro)** | mean(2*P*R/(P+R) per class) | Class-imbalanced data |
| **Confusion Matrix** | C[i,j] = count(true=i, pred=j) | Per-class analysis |

### SSL / Unsupervised Metrics

Feature extraction -> KMeans clustering -> Evaluate cluster-label alignment.

| Metric | Range | Meaning |
|--------|-------|---------|
| **NMI** | [0, 1] | Mutual information between clusters and labels (normalized) |
| **ARI** | [-1, 1] | Chance-adjusted agreement (0 = random) |
| **Silhouette** | [-1, 1] | Cluster cohesion vs separation (no labels needed) |
| **Purity** | [0, 1] | Fraction of dominant class per cluster |
| **Calinski-Harabasz** | [0, inf) | Ratio of between/within cluster variance |

### Pose Estimation Metrics

| Metric | Formula | Threshold |
|--------|---------|-----------|
| **PCK@k** | fraction of joints within k * body_size | k = 0.1, 0.2 |
| **OKS** | exp(-d^2 / (2 * s^2 * k^2)) | s = object scale, k = per-joint constant |

## Critical Issues in Unsupervised Evaluation

### Hungarian Matching Limitation

Standard cluster evaluation uses Hungarian algorithm for one-to-one cluster-label assignment:

```
Problem:
  Cluster A contains {80% attack, 20% mount}
  Cluster B contains {90% mount, 10% attack}
  -> Hungarian assigns: A->attack, B->mount
  -> Attack accuracy = 80%, Mount accuracy = 90%

But what if:
  Cluster A = "aggressive approach" (pre-attack + pre-mount)
  -> The model discovered a finer-grained behavior!
  -> Hungarian treats this as an error.
```

**Limitations**:
1. Assumes 1:1 mapping (cluster count = class count)
2. Penalizes finer-grained discovery
3. Biased toward majority classes

### Better Alternatives

| Method | Advantage |
|--------|-----------|
| **Many-to-One** | Multiple clusters can map to one class |
| **NMI/ARI** | Less sensitive to cluster count mismatch |
| **Silhouette** | No labels required (internal quality) |
| **Downstream probing** | Train linear classifier on frozen features |

### Recommended Evaluation Protocol

```
1. Primary (label-free):
   - Silhouette score (cluster quality)
   - Calinski-Harabasz index

2. Secondary (with labels):
   - NMI, ARI (cluster-label alignment)
   - Linear probe accuracy (frozen features -> logistic regression)

3. Optional:
   - Hungarian-matched accuracy (for comparability with prior work)
   - t-SNE/UMAP visualization (qualitative)
```

## Sparse Data Considerations

```
Dense data (images):  150K values -> Reconstruction error meaningful
Sparse data (skeleton): 1.8K values -> Reconstruction error noisy

Recommendation:
  - Prefer latent-space metrics for SSL (representation quality)
  - Use downstream task performance as primary measure
  - Report cluster metrics as secondary (NMI, ARI, Silhouette)
```

---

*See also: [SSL Methods](ssl_methods.md) | [Graph Models](graph_models.md) | [Architecture](../architecture.md)*
