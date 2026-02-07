# behavior-lab: Modular Integration Plan

> **Note (v0.2)**: 이 문서는 초기 통합 계획의 히스토리 기록입니다.
> 실제 구현에서 디렉토리명이 변경되었습니다:
> `graph_models/` → `graph/`, `sequence_models/` → `sequence/`,
> `unsupervised/` → `discovery/`, `external/` → `graph/`에 통합.
> 현재 구조는 [Architecture](architecture.md) 참조.

## Context

**문제**: 3개의 분산된 레포 (infogcn-project, superanimal-behavior-poc, animal-behavior-analysis)가 스켈레톤 정의, 데이터 형식, 모델 인터페이스가 각각 달라 연구 워크플로우가 단편화됨.

**목적**: 단일 `behavior-lab` 레포로 통합하여 모듈식 연구 실험 플랫폼 구축. 스켈레톤 기반 행동 인식(인간+동물)을 GCN/SSL/시퀀스 모델로 통합 실험.

**핵심 결정**:
- 연구 실험 플랫폼 (CLI/스크립트 중심)
- Web UI는 optional sub-package `[web]`
- 기존 레포는 당분간 유지

---

## Architecture Overview

```
core/ (numpy only)     →  data/ (feeders, loaders, features)
  skeleton, graph,         →  models/ (graph, sequence, ssl, unsupervised)
  tensor_format, types         →  training/ + evaluation/
                                    →  pose/ [dlc] + viz/ + app/ [web]
```

**핵심 설계 원칙**:
- `core/`는 torch 의존성 없음 (numpy only) → 경량 환경에서도 skeleton registry 사용 가능
- Canonical data format은 `(T, K, D)` → Graph model 입력 시에만 `(N, C, T, V, M)` 변환
- Hydra `_target_` 패턴으로 config 기반 실험 재현
- 새 skeleton = YAML 1개 + (선택) Python 파일 1개

---

## Directory Structure

```
behavior-lab/
├── pyproject.toml                    # pip install -e ".[all]"
├── configs/                          # Hydra config root
│   ├── config.yaml                   # Main defaults
│   ├── skeleton/                     # ntu25, ucla20, mars_mouse7, dlc_topviewmouse27, ...
│   ├── dataset/                      # ntu60_xsub, mars, calms21, ...
│   ├── model/                        # infogcn, stgcn, agcn, mlp, lstm, transformer
│   ├── ssl/                          # mae, jepa, dino
│   ├── training/                     # default, fast_debug, ssl_pretrain
│   ├── pose/                         # dlc_topviewmouse, dlc_quadruped, yolo
│   └── experiment/                   # Composed experiment presets
│
├── src/behavior_lab/
│   ├── core/                         # Zero heavy deps (numpy only)
│   │   ├── skeleton.py               # SkeletonDefinition + SkeletonRegistry
│   │   ├── graph.py                  # Adjacency matrix from skeleton
│   │   ├── tensor_format.py          # (T,K,D) <-> (N,C,T,V,M) bridge
│   │   └── types.py                  # BehaviorSequence, ClassificationResult, Protocols
│   │
│   ├── data/
│   │   ├── feeders/                  # PyTorch Datasets
│   │   │   ├── base.py              # BaseSkeletonFeeder Protocol
│   │   │   ├── ntu_feeder.py        # NTU RGB+D (from infogcn)
│   │   │   ├── mars_feeder.py       # MARS/CalMS21 (from infogcn)
│   │   │   └── sequence_feeder.py   # Generic (T,K,3) → auto-converts to graph format
│   │   ├── loaders/                  # Raw data parsers (non-PyTorch)
│   │   │   ├── mars_loader.py       # (from superanimal-poc)
│   │   │   ├── calms21_loader.py
│   │   │   └── custom_loader.py
│   │   ├── preprocessing/            # (from infogcn)
│   │   │   └── augmentation.py      # Unified: random_rot + SkeletonAugmentor
│   │   └── features/                 # Biomechanical features
│   │       ├── kinematic.py          # speed, velocity, acceleration
│   │       ├── morphometric.py       # body_spread, spatial_variance
│   │       └── pipeline.py           # FeatureExtractor orchestrator
│   │
│   ├── models/
│   │   ├── _registry.py              # get_model(name, **cfg) factory
│   │   ├── graph_models/             # (N,C,T,V,M) input
│   │   │   ├── infogcn.py           # InfoGCN + Interaction (from infogcn)
│   │   │   ├── stgcn.py             # ST-GCN (from infogcn baselines)
│   │   │   ├── agcn.py              # 2s-AGCN (from infogcn baselines)
│   │   │   └── modules.py           # EncodingBlock, SA_GC, TCN
│   │   ├── sequence_models/          # fit/predict/evaluate interface
│   │   │   ├── base.py              # BaseActionClassifier ABC
│   │   │   ├── rule_based.py        # (from superanimal-poc)
│   │   │   ├── mlp.py, lstm.py, transformer.py
│   │   │   └── (from superanimal-poc action_models.py)
│   │   ├── ssl/                      # Self-supervised learning
│   │   │   ├── encoders.py          # GCN/InfoGCN/Interaction encoders
│   │   │   ├── methods.py           # MAE, JEPA, DINO
│   │   │   └── models.py            # SSLModel wrapper + build_model()
│   │   ├── unsupervised/
│   │   │   └── clustering.py        # PCA→UMAP→KMeans (from animal-behavior)
│   │   └── losses/
│   │       ├── label_smoothing.py
│   │       └── mmd.py
│   │
│   ├── pose/                         # [dlc] optional
│   │   ├── base.py                  # PoseEstimator Protocol
│   │   ├── superanimal.py           # DLC wrapper (merge superanimal + animal-behavior)
│   │   ├── yolo_pose.py             # YOLO pose (from superanimal-poc)
│   │   └── presets.py               # DLC keypoint preset definitions
│   │
│   ├── evaluation/                   # (merge superanimal + infogcn)
│   │   ├── metrics.py, comparison.py, evaluator.py
│   │
│   ├── visualization/                # (merge all three)
│   │   ├── skeleton_vis.py, attention_vis.py, trajectory_vis.py
│   │   ├── report_vis.py, dashboard.py
│   │
│   ├── training/
│   │   ├── trainer.py               # Unified Trainer (Hydra config)
│   │   ├── ssl_trainer.py           # SSL pretraining
│   │   └── callbacks.py             # Checkpoint, early stopping
│   │
│   └── app/                          # [web] optional
│       ├── backend/                  # FastAPI (from animal-behavior)
│       └── frontend/                 # React (from animal-behavior)
│
├── scripts/                          # CLI entry points
│   ├── train.py, train_ssl.py, evaluate.py, infer.py
│   └── run_experiment.py
│
├── tests/
│   ├── test_core/, test_data/, test_models/, test_integration/
│
└── docs/
```

---

## Key Migration Mapping

| Source | Target | Notes |
|--------|--------|-------|
| **infogcn** `src/data/skeleton_registry.py` | `core/skeleton.py` | Base. DLC presets 추가 |
| **infogcn** `src/data/graph.py` | `core/graph.py` | SkeletonDefinition과 통합 |
| **New** | `core/tensor_format.py` | `(T,K,D)` <-> `(N,C,T,V,M)` 변환 |
| **infogcn** `src/data/feeder.py` | `data/feeders/ntu,mars_feeder.py` | skeleton config 사용으로 수정 |
| **superanimal** `src/data/datasets.py` | `data/loaders/mars,calms21_loader.py` | |
| **animal-behavior** `feature_service.py` | `data/features/kinematic,morphometric.py` | |
| **infogcn** `src/models/infogcn.py` | `models/graph_models/infogcn.py` | |
| **infogcn** `src/models/baselines.py` | `models/graph_models/stgcn.py,agcn.py` | 분리 |
| **superanimal** `src/models/action_models.py` | `models/sequence_models/*.py` | 분리 |
| **infogcn** `ssl_framework/*` | `models/ssl/*` | 그대로 이동 |
| **animal-behavior** `clustering_service.py` | `models/unsupervised/clustering.py` | |
| **superanimal** `src/models/predictor.py` | `pose/superanimal.py` | animal-behavior 서비스와 병합 |
| **superanimal** `src/evaluation/*` | `evaluation/*` | |
| **animal-behavior** `backend/+frontend/` | `app/` | import 경로만 변경 |

---

## Data Format Unification

**Canonical form**: `(T, K, D)` — 모든 데이터의 기본 표현

```
Pose Estimator → (T, K, 3)
                    │
       ┌────────────┼─────────────────┐
       ↓            ↓                 ↓
  SequenceModel  FeatureExtract   tensor_format.py
  (T, K, D)      → (T, n_feat)   → (N, C, T, V, M)
  MLP/LSTM/      clustering       GraphModel
  Transformer                     InfoGCN/STGCN/AGCN
```

`sequence_feeder.py`가 핵심 접착제: `(T,K,3)` 입력을 받아 `__getitem__`에서 자동으로 `(C,T,V,M)` 반환.

---

## Installation Extras

```toml
dependencies = ["numpy", "scipy", "scikit-learn", "hydra-core", "omegaconf", "tqdm", "einops"]

[project.optional-dependencies]
torch = ["torch>=2.0", "torchvision"]
dlc = ["deeplabcut[modelzoo]>=3.0.0rc1"]
web = ["fastapi", "uvicorn", "python-multipart"]
viz = ["matplotlib", "seaborn", "imageio", "imageio-ffmpeg"]
clustering = ["umap-learn"]
all = ["behavior-lab[torch,viz,clustering]"]
dev = ["behavior-lab[all]", "pytest", "pytest-cov", "ruff", "mypy"]
```

---

## Migration Phases (7 Phases, ~10-15일)

### Phase 1: Scaffolding + Core (1-2일)
- 프로젝트 구조 생성, `pyproject.toml`, git init
- `core/skeleton.py`: infogcn SkeletonDefinition 확장 + DLC preset skeletons
- `core/graph.py`: Graph 클래스 → SkeletonDefinition 통합
- `core/tensor_format.py`: format bridge 구현
- `configs/skeleton/*.yaml`: 모든 skeleton YAML
- **검증**: `from behavior_lab.core import get_skeleton; s = get_skeleton('mars')`

### Phase 2: Data Layer (2-3일)
- feeders: NTU/MARS feeder 이동 + `sequence_feeder.py` (cross-format bridge)
- loaders: MARS/CalMS21/Custom loader 이동
- preprocessing + augmentation 통합
- features: kinematic + morphometric 추출
- **검증**: MARS JSON → BehaviorSequence → SequenceFeeder → DataLoader → `(N,C,T,V,M)`

### Phase 3: Models (2-3일)
- graph_models: InfoGCN, STGCN, AGCN 이동
- sequence_models: BaseActionClassifier + MLP/LSTM/Transformer 이동
- ssl: encoders, methods, models 이동
- unsupervised: clustering 이동
- `_registry.py`: 통합 model factory
- **검증**: 각 모델 forward pass shape 검증

### Phase 4: Training + Evaluation (2일)
- Unified Trainer (Hydra config 기반)
- SSL trainer
- ComprehensiveEvaluator 통합
- scripts/ CLI entry points
- **검증**: `python scripts/train.py dataset=mars model=infogcn training=fast_debug`

### Phase 5: Pose + Visualization (1-2일)
- DLC SuperAnimal + YOLO wrapper 통합
- 시각화 도구 통합 (skeleton, trajectory, dashboard)
- **검증**: mock predictor → features → classification pipeline

### Phase 6: Web App (1-2일)
- FastAPI backend + React frontend를 `app/` sub-package로 이동
- `behavior_lab` 패키지 import로 변경
- **검증**: `pip install -e ".[web]"` → FastAPI 서버 기동

### Phase 7: Polish (1-2일)
- README, docs, tests 보강
- CI/CD setup

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| DLC dependency conflict | `[dlc]` extra 완전 분리, mock fallback |
| Tensor format 변환 정밀도 | Round-trip test 필수 (seq→graph→seq == original) |
| Graph 클래스 vs SkeletonDefinition 이중화 | Phase 1에서 통합 |
| SSL/Supervised 인터페이스 불일치 | Config 기반 data format 선택 (isinstance 아님) |
| Multi-person (M dim) 처리 차이 | skeleton + dataset config에서 num_persons 명시, auto-padding |

---

## Verification (End-to-End)

```bash
# Phase 1 검증
python -c "from behavior_lab.core import get_skeleton; print(get_skeleton('mars'))"

# Phase 2-3 검증
python -c "
from behavior_lab.core import sequence_to_graph, get_skeleton
from behavior_lab.models import get_model
import numpy as np
s = get_skeleton('mars')
seq = np.random.randn(64, 7, 2)
tensor = sequence_to_graph(seq, s)
model = get_model('infogcn', skeleton='mars', num_classes=4)
# forward pass test
"

# Phase 4 검증
python scripts/train.py dataset=mars model=infogcn training=fast_debug

# Full pipeline 검증
python scripts/run_experiment.py experiment=mars_infogcn
```

---

*Created: 2026-02-07*
