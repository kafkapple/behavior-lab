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
├── visualization/     [viz] Skeleton (colored, multi-person), HTML report
│   ├── colors         Body-part palette + multi-person distinction
│   ├── skeleton       plot/animate/compare (auto body-part coloring)
│   ├── embedding      UMAP/t-SNE scatter plots
│   ├── analysis       Transition matrix, bout duration, ethogram
│   └── html_report    Self-contained HTML report generator
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

### Pose Output Contract (stage boundary, 2026-10-01)

Decision (vault `261002_Behavior_Lab_two_stage_unification_vision`, provisional 2026-10-02, confirmed against source 2026-10-01): the hand-off between keypoint production and behavior analysis follows the `movement` axis layout; internal compute keeps `(T, K, D)`; archival export targets NWB `ndx-pose`. Source check results below are the binding details.

| Field | Contract | Source (verbatim) |
|-------|----------|-------------------|
| axes | `time, space, keypoint, individual` (singular names) | `movement` v0.17.0 `docs/source/user_guide/movement_dataset.md`: "position: ... shape (`time`, `space`, `keypoint`, `individual`)". Plural names were used through v0.16.0 (PR #973, 2026-05-15) |
| confidence | separate array `(time, keypoint, individual)`, model-reported | same doc: "confidence: ... with shape (`time`, `keypoint`, `individual`)". `ndx-pose` spec: "Confidence or likelihood of the estimated positions, scaled to be between 0 and 1." |
| missing | `NaN` in `position` plus our own `valid (time, keypoint, individual)` bool | `movement` has no mask variable; `valid` is a behavior-lab addition |
| spatial unit | required `attrs["space_unit"]` (`px` / `mm` / `gslrm`) + `attrs["reference_frame"]` | `movement` attrs are `fps`, `time_unit`, `source_software` only (no spatial unit). `ndx-pose` carries `unit` (default `pixels`) and `reference_frame` per series |
| provenance | `attrs`: `source_software`, model, calibration id, post-processing, git commit | `movement`: `source_software`; `ndx-pose`: `scorer`, `source_software`, `source_software__version` |
| 3D | allowed; `space` = `x, y, z` | `movement` loaders `from_anipose_file`, `from_dlc_file` (3D), `from_numpy`. The user guide sentence "Currently, we support only 2D poses" is stale relative to those loaders |
| multi-camera archival | `ndx-pose` >= 0.4.0 `MultiCameraPoseEstimation` + `CalibratedCamera` | `ndx-pose` CHANGELOG 0.4.0 (2026-09-30); needs `pynwb >= 4.0.0` |

`BehaviorSequence` today has `keypoints`, `labels`, `skeleton_name`, `fps`, `metadata` only (`core/types.py`). Adding `confidence` and `valid` fields is open item 2 of the vision note's migration order; until then they travel in `metadata`.

#### AVATAR `avatar_11` to `subtle_mouse` (9) index map

AVATAR files: `~/data/avatar_gslrm/keypoints/*.npz` with `keypoints (T, 11, 3)`, `valid (T, 11)`, `frames`, `names`. Skeleton canon = BS `temporal_deform/skeletons_external.py` (`avatar_11`, SLEAP `mice_of` naming). SUBTLE order = `core/skeleton.py` `SUBTLE_MOUSE_SKELETON`.

| subtle idx | subtle name | avatar idx | avatar name |
|-----------:|-------------|-----------:|-------------|
| 0 | nose | 0 | nose1 |
| 1 | neck | 1 | neck1 |
| 2 | tail_base | 6 | tailstart1 |
| 3 | mid_back | — | none (no trunk point between neck and tail base) |
| 4 | right_hindpaw | 8 | hindlegR1 |
| 5 | left_hindpaw | 7 | hindlegL1 |
| 6 | right_forepaw | 5 | forelegR1 |
| 7 | left_forepaw | 4 | forelegL1 |
| 8 | tail_tip | 10 | tailend1 |

Unused AVATAR points: 2 `earL1`, 3 `earR1`, 9 `tail1`. Caveats: (1) `mid_back` has no source, so a 9-point array cannot be built by re-indexing alone; any synthesized point (e.g. neck–tail_base midpoint) must be marked derived in `metadata`. (2) L/R in AVATAR names were not checked against the labelling protocol (BS comment). (3) `_gslrm.npz` coordinates are in GS-LRM normalized space (range about -0.7 to 0.9, neck–tail_base median 0.56), not mm; the unit field above exists for exactly this. (4) `SUBTLELoader.load_preprocessed` rejects files whose joint count differs from its skeleton; other layouts go through `behavior_lab.data.ingest()`, which keeps `names` and `valid` in `metadata`.

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
# python scripts/train.py model=bsoid   (discovery)
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
  [clustering]  umap-learn              discovery/clustering
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

## Visualization System

### Color Pipeline

```
skeleton.body_parts → get_joint_colors(skeleton) → per-joint hex colors
                    → get_limb_colors(skeleton)  → per-edge hex colors

skeleton.num_persons > 1 → get_person_colors(n) → distinct person palette
```

**Body-part palette** (`BODY_PART_COLORS`): head=red, torso=blue, left_arm=green, right_arm=orange, left_leg=purple, right_leg=dark orange, tail=gray.

**Multi-person**: Automatic detection when `K >= num_joints * num_persons`. Each person rendered with a distinct base color (teal, red, blue, orange).

### HTML Report

```python
generate_pipeline_report(report_data, "report.html")
```

Structure: Header → Tab navigation (Overview + per-dataset) → Metric cards + embedded images (base64 PNG/GIF). Single HTML file, no external dependencies.

---

> [← MoC](README.md) | [Overview](overview.md) | [Model Taxonomy](model_taxonomy.md) | [Theory →](theory/)

*behavior-lab v0.1 | Updated: 2026-02-08*
