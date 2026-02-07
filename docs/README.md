# behavior-lab Documentation (MoC)

> Map of Content — 모든 문서의 진입점. 개요를 여기서 파악하고, 상세는 백링크로.

---

## Project Identity

Skeleton-based behavior recognition across species (human + animal).
Supervised, Self-supervised, Unsupervised 3 paradigm을 단일 `get_model()` factory로 통합.

**Canonical format**: `(T, K, D)` — 모든 데이터의 기본 표현
**Graph format**: `(N, C, T, V, M)` — GCN 모델 전용 (자동 변환)

---

## Model Taxonomy (5 Categories, 30+ Models)

```
get_model('name')
    │
    ├── graph/         GCN family (자체 + 외부 통합)
    │   ├── Self ───── InfoGCN, STGCN, AGCN
    │   └── PySKL ──── ST-GCN++, CTR-GCN, MS-G3D, AAGCN, PoseConv3D
    ├── sequence/      Sequence classifiers
    │   └── MLP, LSTM, Transformer, RuleBased
    ├── ssl/           Self-supervised (3 encoder × 3 method = 9)
    │   └── {GCN, InfoGCN, Interaction} × {MAE, JEPA, DINO}
    └── discovery/     Behavior discovery (라벨 불필요)
        └── B-SOiD, MoSeq, SUBTLE, BehaveMAE, clustering
```

| 상황 | 추천 모델 |
|------|-----------|
| 라벨 있음 + graph topology 활용 | `infogcn`, `stgcn`, `ctrgcn_pyskl` |
| 라벨 있음 + 빠른 실험 | `lstm`, `transformer` |
| 라벨 없음 + feature 학습 | `infogcn_dino` (SSL) |
| 라벨 없음 + behavior 발견 | `bsoid`, `moseq`, `subtle` |
| 라벨 없음 + 계층적 표현 | `behavemae` |
| 다중 개체 상호작용 | `infogcn_interaction` → `interaction_dino` |

> **상세**: [Model Taxonomy](model_taxonomy.md) — 분류 기준 분석, 대안 비교, 리팩토링 로드맵

---

## Document Map

### Core

| Document | 핵심 내용 | 백링크 |
|----------|----------|--------|
| **[Overview](overview.md)** | 연구 질문, 가설 (H1-H4), 3-paradigm 파이프라인 | ← taxonomy, theory/* |
| **[Architecture](architecture.md)** | 모듈 맵, 데이터 포맷, Hydra config, skeleton registry | ← taxonomy, overview |
| **[Model Taxonomy](model_taxonomy.md)** | 30+ 모델 카탈로그, 분류 체계 비판, 대안 비교 | ← architecture |

### Theory

| Document | 핵심 내용 | 관련 모델 |
|----------|----------|-----------|
| **[Graph Models](theory/graph_models.md)** | GCN → ST-GCN → InfoGCN 이론 체계 | graph/ (self + PySKL) |
| **[SSL Methods](theory/ssl_methods.md)** | MAE vs JEPA vs DINO, sparse data 문제 | ssl/ |
| **[Cross-Species](theory/cross_species.md)** | Canonical 5-part skeleton, 종간 전이 | core/skeleton.py |
| **[Evaluation](theory/evaluation.md)** | NMI/ARI/Silhouette, Hungarian matching | evaluation/ |

### Reference

| Document | 내용 |
|----------|------|
| **[Integration Plan](INTEGRATION_PLAN.md)** | 3-repo 통합 히스토리, Phase 1-7 |

---

## Skeleton Registry (7 Species)

| Name | Joints | Dims | Species | Use Case |
|------|--------|------|---------|----------|
| `ntu` | 25 | 3D | Human | NTU RGB+D, action recognition |
| `ucla` | 20 | 3D | Human | N-UCLA |
| `coco` | 17 | 2D | Human | 2D pose estimation |
| `mars` / `calms21` | 7 | 2D | Mouse | Social behavior (top-view) |
| `dlc_topviewmouse` | 27 | 2D | Mouse | DLC SuperAnimal (full) |
| `dlc_quadruped` | 39 | 2D | Quadruped | DLC SuperAnimal |

> **상세**: [Architecture § Skeleton Registry](architecture.md#skeleton-registry)

---

## Quick Start

```python
from behavior_lab.models import get_model, list_models
from behavior_lab.core import get_skeleton, sequence_to_graph

# All available models
print(list_models())

# Supervised
model = get_model('infogcn', num_classes=60, skeleton='ntu')

# SSL
model = get_model('infogcn_dino', num_classes=60, skeleton='ntu')

# Behavior Discovery
model = get_model('bsoid', fps=30)
results = model.fit(data)  # (T, K, 2) -> labels, embeddings

# External (PySKL)
model = get_model('stgcn_pyskl', graph='ntu', num_classes=60)
```

```bash
# Hydra CLI
python scripts/train.py model=infogcn dataset=ntu60_xsub skeleton=ntu25
python scripts/train.py model=stgcn training=fast_debug
```

---

## Datasets

| Dataset | Species | Joints | Classes | Sequences | Config |
|---------|---------|--------|---------|-----------|--------|
| CalMS21 (MARS) | Mouse | 7 | 4 | 19K | `dataset/mars` |
| NTU RGB+D 60 | Human | 25 | 60 | 57K | `dataset/ntu60_xsub` |
| NW-UCLA | Human | 20 | 10 | 1.5K | — |

---

## Source Repositories (Historical)

| Repository | Status | Key Assets Migrated |
|------------|--------|---------------------|
| infogcn-project | **Archived** | InfoGCN, SSL, skeleton registry, feeders |
| superanimal-behavior-poc | **Archived** | DLC wrapper, sequence classifiers, evaluation |
| animal-behavior-analysis | **Archived** | Web app, clustering pipeline, feature extraction |

---

*behavior-lab v0.1 | Created: 2026-02-07*
