# System Architecture

## Module Map

```
behavior-lab/
├── core/           numpy only, zero torch dependency
│   ├── skeleton    SkeletonDefinition registry (7+ species)
│   ├── graph       Adjacency matrices from skeleton
│   ├── tensor_format   (T,K,D) <-> (N,C,T,V,M) bridge
│   └── types       Protocols: ActionClassifier, PoseEstimator
│
├── data/           PyTorch datasets + raw parsers
│   ├── feeders/    SkeletonFeeder (unified, config-driven)
│   ├── loaders/    Raw data parsers (NPZ, JSON, H5)
│   ├── preprocessing/  Augmentation (rotation, scaling, noise)
│   └── features/   Kinematic + morphometric extraction
│
├── models/         All model implementations
│   ├── graph/      InfoGCN, ST-GCN, AGCN (input: N,C,T,V,M)
│   ├── sequence/   LSTM, MLP, Transformer (input: T,K,D)
│   ├── ssl/        MAE, JEPA, DINO + encoders
│   ├── unsupervised/  PCA + UMAP + KMeans pipeline
│   └── losses/     Label smoothing, MMD, contrastive
│
├── training/       Trainer + SSL trainer + callbacks
├── evaluation/     Metrics + comparison framework
├── pose/           [dlc] DLC SuperAnimal + YOLO wrappers
├── visualization/  Skeleton, trajectory, attention plots
└── app/            [web] FastAPI + React (optional)
```

## Data Format Specification

### Canonical Format: `(T, K, D)`

All data flows through this format as the universal representation.

| Dim | Meaning | Example |
|-----|---------|---------|
| T | Time frames | 64, 100, 300 |
| K | Keypoints (joints) | 7 (MARS), 25 (NTU), 27 (DLC) |
| D | Dimensions per joint | 2 (x,y) or 3 (x,y,z) |

### Graph Format: `(N, C, T, V, M)`

Required only for GCN-family models. Converted via `tensor_format.py`.

| Dim | Meaning | Example |
|-----|---------|---------|
| N | Batch size | 16, 32, 64 |
| C | Channels (=D) | 2 or 3 |
| T | Time frames | 64 |
| V | Vertices (=K) | 25 |
| M | Subjects (persons) | 1 or 2 |

### Conversion Rules

```
Sequence -> Graph:
  (T, K, D) ──sequence_to_graph(skeleton)──> (C, T, V, M)
  (N, T, K, D) ──────────────────────────> (N, C, T, V, M)

Graph -> Sequence:
  (C, T, V, M=1) ──graph_to_sequence()──> (T, V, C)
  (C, T, V, M=2) ─────────────────────> (T, M*V, C)

Special Cases:
  - Multi-person flattened: (T, M*K, D) -> auto-detected -> (C, T, V, M)
  - Channel padding: 2D input + 3D skeleton -> zero-pad 3rd channel
  - Temporal pad/crop: max_frames parameter
```

## Data Flow

```
[Raw Sources]
  Video (mp4) ──> PoseEstimator ──> (T, K, 3)
  NPZ file ────> Loader ──────────> (T, K, D) or (N, C, T, V, M)
  JSON/CSV ────> Loader ──────────> (T, K, D)

[Preprocessing]
  (T, K, D) ──> Augmentation (rotate, scale, noise)
            ──> Feature extraction (velocity, spread)
            ──> Normalization (body-size, z-score)

[Model Input]
  SequenceModel:  (T, K, D) directly
  GraphModel:     tensor_format.sequence_to_graph() -> (N, C, T, V, M)
  Unsupervised:   features -> (T, n_features)

[Training]
  Hydra config ──> Trainer ──> Model + DataLoader + Optimizer
                           ──> Checkpoint + Metrics + Logs

[Evaluation]
  Supervised:    accuracy, F1, confusion matrix
  SSL:           NMI, ARI, silhouette (via KMeans clustering)
  Unsupervised:  silhouette, calinski-harabasz, UMAP viz
```

## Configuration System (Hydra)

```yaml
# configs/config.yaml
defaults:
  - skeleton: mars_mouse7
  - dataset: mars
  - model: infogcn
  - training: default

# Override via CLI:
# python scripts/train.py model=stgcn training=fast_debug
# python scripts/train.py +experiment=mars_infogcn
```

### Config Groups

| Group | Purpose | Examples |
|-------|---------|---------|
| `skeleton/` | Joint topology definitions | ntu25, mars_mouse7, dlc_topviewmouse27 |
| `dataset/` | Data paths + split strategy | ntu60_xsub, mars, calms21 |
| `model/` | Model architecture + hyperparams | infogcn, stgcn, lstm, transformer |
| `ssl/` | SSL method config | mae, jepa, dino |
| `training/` | Optimizer + schedule | default (110ep), fast_debug (5ep) |
| `experiment/` | Composed presets | mars_infogcn, ntu_ssl_dino |

## Skeleton Registry

### Built-in Skeletons

| Name | Joints | Dims | Persons | Source |
|------|--------|------|---------|--------|
| `ntu` | 25 | 3D | 1-2 | NTU RGB+D (Kinect) |
| `ucla` | 20 | 3D | 1-10 | N-UCLA (Kinect) |
| `coco` | 17 | 2D | 1 | MS COCO (2D pose) |
| `mars` | 7 | 2D | 2 | CalMS21 (top-view mouse) |
| `calms21` | 7 | 2D | 2 | Alias for mars |
| `dlc_topviewmouse` | 27 | 2D | 1 | DLC SuperAnimal |
| `dlc_quadruped` | 39 | 2D | 1 | DLC SuperAnimal |

### Keypoint Presets (DLC)

```
TopViewMouse 27 (full)
  └── standard 11 (nose, ears, body, tail, hips)
        └── mars 7 (CalMS21-compatible)
              └── locomotion 5 (centroid + extremities)
                    └── minimal 3 (nose, center, tail)
```

### Extension

```python
# Option 1: YAML config
# configs/skeleton/my_skeleton.yaml

# Option 2: Runtime registration
from behavior_lab.core import register_skeleton, SkeletonDefinition
register_skeleton("my_skeleton", SkeletonDefinition(...))
```

## Dependencies

```
Core (numpy only):  core/
  └── numpy, scipy

ML Layer:           data/, models/, training/, evaluation/
  └── + torch, scikit-learn, hydra-core, einops

Optional:
  [dlc]  deeplabcut>=3.0     pose/
  [web]  fastapi, react      app/
  [viz]  matplotlib, seaborn  visualization/
```

---

*See also: [Overview](overview.md) | [Integration Plan](INTEGRATION_PLAN.md) | [Theory: Graph Models](theory/graph_models.md)*
