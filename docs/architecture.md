# System Architecture

> [← MoC](README.md) | [Overview →](overview.md) | [Model Taxonomy →](model_taxonomy.md)

## Module Map

```
behavior-lab/
├── core/              numpy only, zero torch dependency
│   ├── skeleton       SkeletonDefinition registry (7+ species)
│   ├── graph          Adjacency matrices from skeleton
│   ├── tensor_format  (T,K,D) <-> (N,C,T,V,M) bridge
│   └── types          Protocols: ActionClassifier, PoseEstimator
│
├── data/              PyTorch datasets + raw parsers
│   ├── feeders/       SkeletonFeeder (unified, config-driven)
│   ├── loaders/       Raw data parsers (NPZ, JSON, H5)
│   ├── preprocessing/ Augmentation (rotation, scaling, noise)
│   └── features/      Kinematic + morphometric extraction
│
├── models/            30+ models via get_model() factory
│   ├── graph/         InfoGCN, STGCN, AGCN + PySKL  (N,C,T,V,M)
│   ├── sequence/      LSTM, MLP, Transformer         (T,K*D)
│   ├── ssl/           3 encoders × 3 methods = 9     (N,C,T,V,M)
│   ├── discovery/     B-SOiD, MoSeq, SUBTLE,         (T,K,D)
│   │                  BehaveMAE, clustering
│   └── losses/        Label smoothing, MMD
│
├── training/          Trainer + SSL trainer
├── evaluation/        Classification + Cluster + LinearProbe metrics
├── pose/              [dlc] DLC SuperAnimal + YOLO wrappers
├── visualization/     [viz] Skeleton, trajectory, attention plots
└── app/               [web] FastAPI + React (optional)
```

> **분류 체계 상세**: [Model Taxonomy](model_taxonomy.md) — 30+ 모델 카탈로그, 분류 기준 비판 및 대안

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

### External Model Format Bridge

외부 모델 wrapper들은 (T,K,D) 입력을 내부적으로 각 라이브러리 포맷으로 변환:

| Model | Internal Format | Conversion |
|-------|----------------|------------|
| PySKL | (M, T, V, C) → (N, C, T, V, M) | `pose_to_pyskl_format()` |
| BehaveMAE | (B, 1, T, 1, K*D) | `pose_to_behavemae_input()` |
| B-SOiD | (T', n_features) @ 10fps | `_compute_bsoid_features()` |
| MoSeq | {name: (T, K, D)} dict | `_to_kpms_format()` |
| SUBTLE | List[(T, K*D)] | `_preprocess()` |

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
  GraphModel:      tensor_format → (N, C, T, V, M)
  SequenceModel:   (T, K*D) directly
  SSL:             tensor_format → (N, C, T, V, M)
  Discovery:       (T, K, D) → internal conversion per wrapper
  Graph(PySKL):    (T, K, D) → pose_to_pyskl_format() → (M, T, V, C)

[Training]
  Hydra config ──> Trainer ──> Model + DataLoader + Optimizer
                           ──> Checkpoint + Metrics + Logs

[Evaluation]
  Supervised:    accuracy, F1, confusion matrix
  SSL:           NMI, ARI, silhouette (via KMeans on features)
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
# python scripts/train.py model=bsoid   (unsupervised)
# python scripts/train.py model=stgcn_pyskl   (external)
```

### Config Groups

| Group | Purpose | Files |
|-------|---------|-------|
| `skeleton/` | Joint topology definitions | ntu25, ucla20, mars_mouse7, coco17, dlc_* |
| `dataset/` | Data paths + split strategy | ntu60_xsub, mars |
| `model/` | Model architecture + hyperparams | infogcn, stgcn, agcn, bsoid, moseq, subtle, behavemae, *_pyskl |
| `ssl/` | SSL method config | mae, jepa, dino |
| `training/` | Optimizer + schedule | default, fast_debug, ssl_pretrain |

## Skeleton Registry

> [← MoC § Skeleton](README.md#skeleton-registry-7-species)

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
# Option 1: YAML config (configs/skeleton/my_skeleton.yaml)
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

Optional extras:
  [clustering]  umap-learn              unsupervised/clustering
  [bsoid]       umap-learn, hdbscan     discovery/bsoid
  [moseq]       keypoint-moseq          discovery/moseq
  [subtle]      subtle                  discovery/subtle
  [pyskl]       pyskl, mmcv, mmaction2  graph/pyskl
  [dlc]         deeplabcut>=3.0         pose/
  [web]         fastapi, react          app/
  [viz]         matplotlib, seaborn     visualization/
```

---

> [← MoC](README.md) | [Overview](overview.md) | [Model Taxonomy](model_taxonomy.md) | [Theory →](theory/)

*behavior-lab v0.1 | Updated: 2026-02-07*
