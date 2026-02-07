# Graph Models for Skeleton Action Recognition

> *Backlinks: [Overview](../overview.md) | [Architecture](../architecture.md) | [SSL Methods](ssl_methods.md)*

## Graph Convolution on Skeletons

A skeleton with V joints and E edges forms a spatial graph G = (V, E). Graph convolution generalizes CNNs to non-Euclidean domains:

```
H^(l+1) = sigma( D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l) )

where:
  A = adjacency matrix + self-loops (V x V)
  D = degree matrix
  H^(l) = node features at layer l (V x C)
  W^(l) = learnable weights
```

## Spatial Partitioning Strategy (ST-GCN)

ST-GCN decomposes adjacency into 3 subsets relative to a center joint:

```
A = A_identity + A_inward + A_outward

A_identity:  self-connections (diagonal)
A_inward:    edges toward center (centripetal)
A_outward:   edges away from center (centrifugal)
```

Each subset gets independent weights, enabling direction-aware message passing:

```
Y = sum_k( A_k * X * W_k )   for k in {identity, inward, outward}
```

## Model Architectures

### ST-GCN (Yan et al., AAAI 2018)

```
Input (N,C,T,V,M)
  -> BatchNorm
  -> [SpatialGCN -> TemporalConv -> Residual] x 9
  -> GlobalAvgPool
  -> FC -> logits

SpatialGCN: Fixed 3-partition adjacency
TemporalConv: 1D conv (kernel=9) along time axis
```

- **Graph**: Fixed partition (no learning)
- **Temporal**: Single-scale 1D convolution
- **Parameters**: ~3M (NTU-25)

### 2s-AGCN (Shi et al., CVPR 2019)

```
AdaptiveGCN: A = A_fixed + B_adaptive + C_attention
  B: Learnable (V x V) parameter
  C: Data-dependent attention (softmax over dot product)
```

- **Graph**: Adaptive topology learning
- **Two-stream**: Joint positions + bone vectors (optional)
- **Parameters**: ~3.5M

### InfoGCN (Chi et al., CVPR 2022)

```
Input (N,C,T,V,M)
  -> [SA-GC + MS-TCN + Residual] x 9
  -> GlobalAvgPool
  -> FC -> (logits, z, mu, logvar)

SA-GC (Self-Attention Graph Convolution):
  Q, K, V = Linear(X)
  Attn = softmax(Q * K^T / sqrt(d))
  Y = sum_k( (A_k * Attn) * V * W_k )

MS-TCN (Multi-Scale Temporal Convolution):
  Y = Conv_1x1(X) + Conv_5(X) + Conv_7(X) + MaxPool_3(X)
  (multi-dilation captures short/mid/long temporal patterns)

Information Bottleneck:
  L = L_classification + alpha * L_mmd
  L_mmd = MMD(q(z|x), p(z))   (match posterior to prior)
```

- **Graph**: Dynamic attention per layer (learned topology)
- **Temporal**: Multi-scale (3 dilations + pooling)
- **Bottleneck**: MMD loss regularizes latent space
- **Parameters**: ~1.2M (CalMS21-7)

### InfoGCN-Interaction (for multi-subject)

Extends InfoGCN with inter-subject modeling:

```
Features per subject: f1, f2 = InfoGCN(subject_1), InfoGCN(subject_2)

InteractionPooling:
  features = [f1, f2, f1 * f2, |f1 - f2|]  (product + difference)

SubjectCrossAttention:
  Q = Linear(f1), K = Linear(f2), V = Linear(f2)
  f1' = MultiHeadAttn(Q, K, V)  (what subject 1 attends to in subject 2)

InterSubjectGraphConv:
  Build cross-subject edges (same_joint / full / learned)
  Apply GCN on concatenated graph [V1; V2]
```

## Comparison

| Feature | ST-GCN | 2s-AGCN | InfoGCN |
|---------|--------|---------|---------|
| Graph topology | Fixed 3-partition | Adaptive (B+C) | SA attention |
| Temporal conv | Single scale | Single scale | Multi-scale |
| Bottleneck | None | None | MMD (IB) |
| Multi-subject | Concatenation | Concatenation | Cross-attention |
| Parameters | ~3M | ~3.5M | ~1.2M |

## Implementation Notes

All graph models in behavior-lab share:
- Input format: `(N, C, T, V, M)` via `core.tensor_format.sequence_to_graph()`
- Adjacency from: `core.graph.Graph(skeleton)` -> `graph.A` shape `(3, V, V)`
- Output: `(N, num_classes)` logits (+ optional latent `z, mu, logvar` for InfoGCN)

---

*See also: [SSL Methods](ssl_methods.md) | [Evaluation](evaluation.md) | [Architecture](../architecture.md)*
