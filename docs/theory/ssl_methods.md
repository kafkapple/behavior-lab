# Self-Supervised Learning Methods for Skeleton Data

> *Backlinks: [Overview](../overview.md) | [Graph Models](graph_models.md) | [Evaluation](evaluation.md)*

## SSL Paradigm Classification

```
Self-Supervised Learning
├── Reconstruction (Non-Latent)
│   └── MAE: Reconstruct masked input in pixel space
│       Loss = MSE(x_hat, x_masked)
│
└── Embedding (Latent)
    ├── JEPA: Predict masked representation in latent space
    │   Loss = MSE(z_hat, z_target)   (no decoder needed)
    │
    └── DINO: Match distributions via self-distillation
        Loss = CE(softmax(s/tau_s), softmax(t/tau_t))
```

## Why Skeleton Data is Special

| Property | Image (~150K values) | Skeleton (~1.8K values) |
|----------|---------------------|------------------------|
| Density | Dense pixels | Sparse keypoints |
| Structure | Grid (CNN-friendly) | Graph (topology matters) |
| Redundancy | High (neighboring pixels similar) | Low (each joint unique) |
| Masking 40% | ~90K remain (sufficient) | ~1.1K remain (risky) |

**Implication**: Reconstruction targets (MAE) are harder for sparse data. Latent-space methods (JEPA, DINO) are more robust because they predict abstract representations rather than exact coordinates.

## Method Details

### MAE (Masked Autoencoder)

```
Input: x = (N, C, T, V, M)
  1. Mask 40% of temporal-joint blocks
  2. Encode visible tokens: z_vis = Encoder(x_visible)
  3. Decode with mask tokens: x_hat = Decoder(z_vis, mask_tokens)
  4. Loss = MSE(x_hat[masked], x[masked])

Encoder: GCN / InfoGCN / InfoGCN-Interaction
Decoder: 2-layer MLP (project back to input dim)
Mask strategy: Temporal blocks (consecutive frames per joint)
```

**Strengths**: Simple, well-understood, preserves spatial structure.
**Weaknesses**: Low-level target (coordinates), struggles with sparse data.

### JEPA (Joint-Embedding Predictive Architecture)

```
Input: x = (N, C, T, V, M)
  1. Split into context (60%) and target (40%) blocks
  2. Context encoder:  z_ctx = Encoder(x_context)       [trainable]
  3. Target encoder:   z_tgt = TargetEncoder(x_target)  [EMA updated]
  4. Predictor:        z_hat = Predictor(z_ctx)          [trainable]
  5. Loss = MSE(z_hat, z_tgt.detach())

EMA update: theta_target = m * theta_target + (1-m) * theta_context
  m = 0.996 (slow-moving average)

No decoder needed (prediction in latent space).
```

**Strengths**: Semantic-level prediction, efficient (no decoder), stable with EMA.
**Weaknesses**: Requires careful predictor capacity tuning, potential representation collapse.

### DINO (Self-Distillation with No Labels)

```
Input: x = (N, C, T, V, M)
  1. Augment: x_s = strong_aug(x), x_w = weak_aug(x)
  2. Student: p_s = softmax(Student(x_s) / tau_s)  [trainable]
  3. Teacher: p_t = softmax(Teacher(x_w) / tau_t)  [EMA updated]
  4. Loss = - sum( p_t * log(p_s) )  (cross-entropy)
  5. Centering: c = m*c + (1-m)*mean(teacher_output)
     p_t = softmax((Teacher(x_w) - c) / tau_t)

Temperature: tau_s = 0.1 (sharp student), tau_t = 0.04 (sharper teacher)
Centering prevents mode collapse (all outputs becoming uniform).

Augmentations:
  Strong: rotation (+-30), scaling (0.8-1.2), noise (sigma=0.02)
  Weak: slight noise only
```

**Strengths**: Best for skeleton data (distribution matching, augmentation-driven).
**Weaknesses**: Sensitive to augmentation design, centering is crucial.

## Encoder Compatibility

| Encoder | MAE | JEPA | DINO | Multi-Subject |
|---------|-----|------|------|---------------|
| GCN | Simple reconstruction | Latent prediction | Distribution matching | Mean pooling |
| InfoGCN | + attention + IB | + attention + IB | + attention + IB | Mean pooling |
| InfoGCN-Interaction | + cross-attention | + cross-attention | + cross-attention | Cross-attention |

All 9 combinations (3 encoders x 3 methods) are supported.

## Skeleton-Specific Adaptations

### Masking Strategy
- **Temporal block masking**: Mask consecutive frames for specific joints
- **Spatial masking**: Mask entire body parts (head, limbs)
- **Random joint masking**: Independent per-joint masking

### Augmentation Design (DINO)
- **Rotation**: Random SO(2) for 2D, SO(3) for 3D (preserves topology)
- **Scaling**: Uniform scaling (body-size invariant)
- **Noise**: Gaussian perturbation (simulates tracking error)
- **Temporal**: Random crop, speed variation

## Observed Results (CalMS21, debug mode)

| Model | Accuracy | NMI | Notes |
|-------|----------|-----|-------|
| gcn_dino | 0.37 | 0.05 | Baseline |
| infogcn_dino | 0.40 | 0.08 | +attention helps |
| interaction_dino | 0.43 | 0.12 | Best (cross-attn) |
| gcn_mae | ~0.30 | ~0.03 | Reconstruction struggles |
| gcn_jepa | ~0.33 | ~0.04 | Mid-range |

---

*See also: [Graph Models](graph_models.md) | [Cross-Species](cross_species.md) | [Evaluation](evaluation.md)*
