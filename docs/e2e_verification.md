# E2E Verification Report

> Full pipeline verification on real data: loader → preprocessing → discovery → evaluation → visualization.
>
> **Date**: 2026-02-08 | **Script**: `scripts/test_e2e.py` | **Outputs**: `outputs/e2e_test/`
> **Status**: ALL TESTS PASSED | **Detailed Report**: [`outputs/e2e_test/report.md`](../outputs/e2e_test/report.md) | **Interactive**: [`report.html`](../outputs/e2e_test/report.html)

---

## Summary

| Dataset | Train | Test | Shape | Key Result |
|---------|-------|------|-------|------------|
| CalMS21 | 19,144 | 4,787 | (64, 14, 2) | B-SOiD: 2 clusters, Silhouette 0.59 |
| NTU RGB+D | 500 | 100 | (300, 50, 3) | Linear probe: 5% (raw features) |
| NW-UCLA | 1,020 | 464 | (300, 20, 3) | Linear probe: 40.5% |

---

## CalMS21 — Mouse Social Behavior

### Data Loading

- NPZ format: `x_train (N, 2, T, 7, 2)` → reshaped to `(N, T, 14, 2)`
- One-hot `y_train (N, 4)` → `argmax` → int label (sequence-level)
- Per-frame labels: `np.full(T, label)` — each sequence has uniform action label

### Preprocessing Pipeline

```python
PreprocessingPipeline([
    Interpolator(max_gap=10),
    OutlierRemover(velocity_threshold=50.0),
    TemporalSmoother(window_size=5),
    Normalizer(center_joint=0),
])
```

- Before: range [85.76, 801.62], NaN=0
- After: range [-1.0, 0.578], NaN=0

### B-SOiD Discovery (1000 samples)

- Time: ~62s (64K frames concatenated)
- Clusters: 2 (+ noise)

| Metric | Value |
|--------|-------|
| Silhouette | 0.5937 |
| NMI | 0.0073 |
| ARI | 0.0238 |
| V-measure | 0.0073 |
| Hungarian Accuracy | 0.6255 |
| Temporal Consistency | 0.8977 |
| Num Bouts | 2,183 |
| Entropy Rate | 0.3189 |

**Note**: Low NMI/ARI is expected — B-SOiD discovers *behavioral motifs* (movement patterns) rather than replicating the 4-class CalMS21 annotation. The high silhouette (0.59) confirms well-separated clusters in feature space.

### Skeleton Visualization

- `calms21/sample_skeleton.png` — Body-part colored 2-mice skeleton (multi-person auto-detection)
- `calms21/sample_animation.gif` — 60-frame animated GIF with per-mouse color distinction
- `calms21/skeleton_comparison.png` — Raw vs preprocessed side-by-side comparison

Color system: head=red, body=blue, tail=gray. Mouse 0 = teal scheme, Mouse 1 = red scheme (darkened).

### Analysis Visualizations

- `calms21/preprocessing_comparison.png` — Before/after keypoint trajectory (joint 0, x/y)
- `calms21/bsoid_embedding.png` — UMAP 2D scatter (cluster colors)
- `calms21/bsoid_transition_matrix.png` — Behavior transition heatmap
- `calms21/bsoid_bout_duration.png` — Mean bout duration per cluster
- `calms21/bsoid_ethogram.png` — Temporal raster (first 5000 frames)

---

## NTU RGB+D — Human Action Recognition (Demo)

### Data

- Demo subset: 500 train, 100 test (full dataset: 57K)
- NPZ format: `x_train (500, 300, 150)` with `F=150 = 50 joints × 3D`
- One-hot `y_train (500, 60)` → `argmax` → 60 action classes

### Skeleton Visualization

- `ntu/sample_skeleton.png` — Body-part colored 3D skeleton (torso=blue, head=red, left_arm=green, right_arm=orange, left_leg=purple, right_leg=dark orange)
- Multi-person: NTU data has 50 joints (25×2 persons). When both are present, each person rendered with distinct color scheme.

### Linear Probe

- Features: mean-pooled keypoints `(N, 150)` — intentionally naive baseline
- Accuracy: **5.0%** (chance = 1.7%)
- F1 (macro): 0.006

The low accuracy confirms that raw mean-pooled features lack discriminative power for 60-class action recognition — temporal and graph-based models (ST-GCN, InfoGCN) are needed.

---

## NW-UCLA — Action Recognition

### Data

- 1,020 train / 464 test, 10 classes
- NPZ format: `(N, 300, 60)` with `F=60 = 20 joints × 3D`
- One-hot `(N, 10)` → `argmax`

### Linear Probe

- Accuracy: **40.5%** (chance = 10%)
- F1 (macro): 0.361

Better than NTU due to fewer classes and simpler actions. Still confirms need for temporal modeling.

### Skeleton Visualization

- `nwucla/sample_skeleton.png` — 20-joint colored skeleton (3D)
- `nwucla/sample_animation.gif` — 60-frame animated GIF with body-part coloring

---

## HTML Report

A self-contained HTML report is generated at `outputs/e2e_test/report.html`:

- **Tab navigation**: Overview + CalMS21 / NTU / NW-UCLA tabs
- **All images base64-embedded**: PNG and GIF inline, no external file dependencies
- **Metric cards**: Hover-interactive display of cluster and behavior metrics
- **Responsive CSS**: Works on desktop and mobile browsers

Open with: `open outputs/e2e_test/report.html` (macOS) or any browser.

---

## Data-Loader Fixes Applied

| Dataset | Before | After |
|---------|--------|-------|
| CalMS21 | Only `.npy` flat format | + NPZ `(N, 2, T, 7, 2)` + one-hot |
| NTU | `labels.flatten()` broke one-hot | One-hot detection + `argmax` |
| NW-UCLA | No loader | New `NWUCLALoader` class |

---

## Backlinks

- [Architecture](architecture.md) — Module structure, data format
- [Overview](overview.md) — Research questions and hypotheses
- [Model Taxonomy](model_taxonomy.md) — Model recommendations
- [Evaluation Theory](theory/evaluation.md) — Metric definitions

---

*behavior-lab v0.1 | E2E Verification | Updated: 2026-02-08*
