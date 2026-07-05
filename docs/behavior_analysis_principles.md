# Behavior Analysis Principles

This appendix explains the theoretical basis behind the behavior-analysis
workbench in `behavior-lab`. It is intentionally method-oriented: the point is
to make the assumptions, invariances, and failure modes explicit before
interpreting results.

## 1. Canonical Pose Representation

All downstream modules should agree on a canonical exchange format:

```text
(T, K, D)
```

- `T`: time index
- `K`: keypoints
- `D`: coordinates, typically `2` or `3`

The same shape can represent DLC, SLEAP, triangulated 3D, or canonical local
NPZ datasets. The value of the format is not the tensor shape itself; it is the
fact that every later step can be expressed as a deterministic transform of the
same object.

Why this matters:

- It separates pose estimation from behavior discovery.
- It makes feature extraction and clustering comparable across datasets.
- It allows missingness to stay explicit as `NaN` until a preprocessing stage
  decides how to interpolate, mask, or drop it.

## 2. From Pose to Features

The workbench supports both hand-designed and learned features.

### Raw keypoints

Flatten `(T, K, D)` into `(T, K*D)` and feed the coordinates directly into PCA,
UMAP, or HMM-style models.

Principle:

- Preserve geometry as much as possible.
- Let the downstream method learn what is useful.

Tradeoff:

- Sensitive to camera frame, scale, and identity swaps.

### Skeleton kinematics

Summary features such as speed, acceleration, spread, and spatial variance turn
pose into a compact motion descriptor.

Principle:

- Behaviors often differ more by motion dynamics than by exact limb position.
- Reducing the pose to kinematic summaries improves robustness and runtime.

Tradeoff:

- Loses joint-specific motif structure.

### Spectral features

Morlet/CWT-style features transform time into a time-frequency representation.

Principle:

- Repetitive actions are not just static poses; they have rhythm.
- Spectral features capture oscillatory structure and temporal periodicity.

Tradeoff:

- Feature dimension grows quickly and can be expensive to inspect manually.

### Self-supervised embeddings

Methods such as CEBRA or pretrained MAE encoders learn latent vectors from
time-windowed pose sequences.

Principle:

- The model learns a representation space where temporally related windows are
  close together.
- That latent space can then be clustered, summarized, or compared across
  datasets.

Tradeoff:

- Less interpretable than hand-crafted features.
- The learned space depends on training protocol and window design.

## 3. CEBRA For Behavior Data

CEBRA is introduced in a joint behavioral-neural setting, but its core
requirement is not neural spikes. It is a structured signal for which we can
define meaningful positive and negative pairs.

The paper describes CEBRA as a method that can be used with behavioral and/or
neural data, in hypothesis-driven or discovery-driven mode, and even label-free
for single- or multi-session settings. In practice, that means the encoder can
operate on behavior-only pose windows when the pair construction is defined by
time, context, or an auxiliary behavioral variable.

### What the anchor is

An anchor is one window or frame-level feature vector:

- raw keypoints flattened into a vector
- kinematic summary features
- spectral windows
- another latent projection of the same behavior sequence

### What counts as a positive

Positive pairs are samples that the task says should stay close:

- nearby windows from the same recording, when temporal continuity is the only
  signal available
- windows from the same behavioral state, if labels or weak annotations exist
- different augmentations of the same pose window
- windows matched by a behavioral context variable such as trial phase or
  session condition

### What counts as a negative

Negative pairs are samples that should not collapse together:

- temporally distant windows in a recording
- windows from different behavioral states
- windows from different sessions or conditions when that difference is not the
  object of the analysis

### Why behavior-only training is valid

Behavior sequences are not arbitrary. Adjacent frames are usually more similar
than distant frames, and recurring movements occupy consistent neighborhoods in
pose space. That gives contrastive learning a usable inductive bias even without
neural data.

The key point is that the representation is learning *behavioral geometry*, not
spike decoding. The neural-data paper is the origin story, not the limit of the
method.

### Practical inference path

1. Convert pose into a time-windowed feature vector.
2. Choose a pairing rule: temporal adjacency, weak behavioral context, or
   augmentation-based self-supervision.
3. Train the encoder so positives are close and negatives are separated.
4. Cluster or segment the latent space, then evaluate bouts and transitions.

What this gives us:

- a geometry that respects temporal continuity,
- a representation that often separates recurring motifs better than raw PCA,
- a bridge between feature learning and unsupervised clustering.

What it does not give us:

- a direct biological label,
- a guarantee that a high silhouette score corresponds to the right syllable
  vocabulary,
- a free pass to mix incompatible temporal scales.

### Failure mode

If the anchor window is too short, the model learns jitter. If it is too long,
it averages away motifs. If positives are defined too loosely, different
behaviors collapse. If negatives are sampled from a different geometry
(2D/3D, different skeleton, different fps), the embedding becomes dataset
separation rather than behavior separation.

## 4. Discovery Families

### Geometry-first clustering

Examples:

- `kmeans_pca_umap`
- simple density clustering on extracted features

Principle:

- Reduce dimensionality, then cluster the resulting feature cloud.

Good for:

- Fast baselines
- Controlled ablations

Weakness:

- Usually ignores sequence order.

### Behavior segmentation methods

Examples:

- B-SOiD
- keypoint-MoSeq
- the PCA-HMM fallback

Principle:

- Cluster on motion features, then segment time into discrete states.
- Some methods use explicit temporal priors, others use a learned classifier or
  HMM/SLDS-style state model.

Good for:

- Motif discovery
- Transition statistics
- Bout-level interpretation

Weakness:

- More sensitive to parameterization and dataset geometry than simple clustering.

### Spectral hierarchy methods

Examples:

- SUBTLE

Principle:

- Convert motion into spectral features, embed with UMAP, cluster with
  Phenograph, then derive hierarchical motif structure.

Good for:

- Spontaneous 3D movement
- Hierarchical motif relationships

Weakness:

- Heavy dependency stack and platform instability can dominate practical use.

### Pretrained representation methods

Examples:

- hBehaveMAE

Principle:

- Use a pretrained encoder to map windows into latent vectors.
- Cluster the latent vectors rather than raw poses.

Good for:

- Cross-dataset comparison
- Low-label or no-label regimes

Weakness:

- Interpretation depends on the checkpoint and the dataset it was trained on.

## 5. Metrics And Interpretation

### Silhouette

Silhouette measures how much a sample is closer to its own cluster than to the
nearest other cluster.

Interpretation:

- Higher is better for geometric separation.
- It is useful for comparing clustering runs on the same embedding.

Limitations:

- It does not validate biological correctness.
- It can be misleading if the cluster count is tiny or the embedding is not
  meant to be cluster-shaped.

### ARI / NMI

ARI and NMI compare discovered labels to ground-truth annotations when they are
available.

Interpretation:

- Useful only when the label vocabulary is comparable.
- Better suited for benchmark datasets than for fully unsupervised discovery.

### Bout duration

Bout duration summarizes how long a syllable/state persists.

Interpretation:

- Short bouts suggest fragmentation.
- Extremely long bouts may indicate under-segmentation.

### Transition matrix

Transition matrices record how often one motif follows another.

Interpretation:

- They summarize sequential grammar.
- They are often more informative than labels alone because they expose motif
  ordering, not just motif identity.

## 6. Cross-Dataset Caution

Do not compare a 2D and a 3D method as if they were solving the same problem.
The following layers are separate:

1. Pose quality and missingness
2. Feature choice
3. Discovery model
4. Evaluation metric

If any one layer changes, the score can change even when the underlying animal
behavior is unchanged.

## 7. Practical Rule

For every experiment, record:

- pose source
- feature block
- discovery method
- temporal resolution
- metric
- known failure mode

That is the minimum needed to make a behavior-analysis result reproducible.

## Backlinks

- [Behavior Analysis PRD](behavior_analysis_prd.md)
- [Behavior Analysis Workbench](behavior_analysis_workbench.md)
- [Notebook MoC](../notebooks/behavior_analysis_workbench/README.md)
