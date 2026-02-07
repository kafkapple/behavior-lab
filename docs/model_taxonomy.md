# Model Taxonomy: Classification System Analysis

> [← MoC](README.md) | [Architecture](architecture.md) | [Overview](overview.md)

## Current Taxonomy (v0.2 — Architecture-first)

> v0.1에서 Option C(Architecture-first) 적용 완료. 변경 이력은 아래 [v0.1 분석](#v01-taxonomy-analysis-historical) 참조.

### Directory Structure

```
models/
├── graph/               ← Architecture: GCN family (자체 + 외부 통합)
│   ├── infogcn.py          InfoGCN, InfoGCN_Interaction
│   ├── baselines.py        STGCN, AGCN
│   ├── modules.py          SA-GC, EncodingBlock, MS-TCN
│   └── pyskl.py            ST-GCN++, CTR-GCN, MS-G3D, AAGCN, ... (PySKL)
├── sequence/            ← Architecture: Sequence classifiers
│   └── classifiers.py     MLP, LSTM, Transformer, RuleBased
├── ssl/                 ← Paradigm: Self-supervised learning
│   ├── encoders.py         GCN, InfoGCN, Interaction encoder
│   ├── methods.py          MAE, JEPA, DINO
│   └── models.py           3 encoders × 3 methods = 9 combos
├── discovery/           ← Task: Behavior discovery (라벨 불필요)
│   ├── clustering.py       PCA → UMAP → KMeans
│   ├── bsoid.py            B-SOiD (UMAP+HDBSCAN+RF)
│   ├── moseq.py            keypoint-MoSeq (AR-HMM/SLDS)
│   ├── subtle_wrapper.py   SUBTLE (Wavelet+UMAP+Phenograph)
│   └── behavemae.py        BehaveMAE (Hierarchical MAE)
└── losses/              ← Role: Loss functions
    └── __init__.py         LabelSmoothing, MMD
```

### v0.1 → v0.2 변경 요약

| Before (v0.1) | After (v0.2) | 이유 |
|----------------|--------------|------|
| `graph_models/` | `graph/` | 간결화 |
| `sequence_models/` | `sequence/` | 간결화 |
| `unsupervised/` | `discovery/` | 의도 명확화 (행동 발견) |
| `external/pyskl.py` | `graph/pyskl.py` | GCN 계열 → graph/에 통합 |
| `external/` 디렉토리 | 삭제 | singleton 카테고리 해소 |

### Full Model Catalog (30+ models)

| # | Model | Category | Dir | Paradigm | Input | Source | Reference |
|---|-------|----------|-----|----------|-------|--------|-----------|
| 1 | InfoGCN | Graph | graph_models | Supervised | (N,C,T,V,M) | Self | Chi+ CVPR'22 |
| 2 | InfoGCN-Interaction | Graph | graph_models | Supervised | (N,C,T,V,M) | Self | Chi+ CVPR'22 |
| 3 | STGCN | Graph | graph_models | Supervised | (N,C,T,V,M) | Self | Yan+ AAAI'18 |
| 4 | AGCN | Graph | graph_models | Supervised | (N,C,T,V,M) | Self | Shi+ CVPR'19 |
| 5 | MLP | Sequence | sequence_models | Supervised | (T,K*D) | Self | — |
| 6 | LSTM | Sequence | sequence_models | Supervised | (T,K*D) | Self | — |
| 7 | Transformer | Sequence | sequence_models | Supervised | (T,K*D) | Self | — |
| 8 | RuleBased | Sequence | sequence_models | Supervised | (T,K*D) | Self | — |
| 9 | GCN-MAE | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | He+ CVPR'22 |
| 10 | GCN-JEPA | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | Assran+ CVPR'23 |
| 11 | GCN-DINO | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | Caron+ ICCV'21 |
| 12 | InfoGCN-MAE | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | — |
| 13 | InfoGCN-JEPA | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | — |
| 14 | InfoGCN-DINO | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | — |
| 15 | Interaction-MAE | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | — |
| 16 | Interaction-JEPA | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | — |
| 17 | Interaction-DINO | SSL | ssl | Self-Supervised | (N,C,T,V,M) | Self | — |
| 18 | PCA+UMAP+KMeans | Clustering | unsupervised | Unsupervised | (N,D) features | Self | — |
| 19 | B-SOiD | Discovery | unsupervised | Unsupervised | (T,K,2) | Ext | Hsu+ NatComm'21 |
| 20 | keypoint-MoSeq | Discovery | unsupervised | Unsupervised | (T,K,D) | Ext | Weinreb+ NatMeth'24 |
| 21 | SUBTLE | Discovery | unsupervised | Unsupervised | (T,K,D) | Ext | Kwon+ IJCV'24 |
| 22 | BehaveMAE | Discovery | unsupervised | Self-Supervised | (T,K,D) | Ext | Stoffl+ ECCV'24 |
| 23 | ST-GCN++ (PySKL) | Graph | external | Supervised | (T,K,D)→auto | Ext | Duan+ ACMMM'22 |
| 24 | CTR-GCN (PySKL) | Graph | external | Supervised | (T,K,D)→auto | Ext | Chen+ ICCV'21 |
| 25 | MS-G3D (PySKL) | Graph | external | Supervised | (T,K,D)→auto | Ext | Liu+ CVPR'20 |
| 26 | AAGCN (PySKL) | Graph | external | Supervised | (T,K,D)→auto | Ext | Shi+ TIP'20 |
| 27 | DG-STGCN (PySKL) | Graph | external | Supervised | (T,K,D)→auto | Ext | — |
| 28 | PoseConv3D (PySKL) | Conv3D | external | Supervised | (T,K,D)→auto | Ext | Duan+ CVPR'22 |

---

## v0.1 Taxonomy Analysis (Historical)

### Problem: Mixed Classification Axes

현재 분류는 **3가지 다른 기준을 혼용**합니다:

```
                     분류 기준
                    ┌─────────────────────────────────────┐
                    │                                     │
              Architecture          Learning Paradigm        Source
              (구조)                (학습 방식)             (출처)
                    │                     │                   │
          ┌────────┼────────┐    ┌───────┼───────┐          │
          │        │        │    │       │       │          │
      graph_   sequence_  (Conv) ssl  unsuper-  (sup.)   external
      models   models            vised
```

이로 인한 **구조적 문제**:

| 문제 | 구체 사례 |
|------|----------|
| **BehaveMAE 정체성 모호** | MAE(SSL)인데 unsupervised/에 배치. 외부 behavior discovery 도구라서. |
| **STGCN 중복** | graph_models/baselines.py에 자체 구현 + external/pyskl.py에 PySKL 버전 |
| **ssl/이 graph_models/에 의존** | ssl/encoders.py가 graph_models/의 GCN 모듈 사용 → 순환 참조 가능성 |
| **external/이 singleton** | pyskl.py 1개뿐. 카테고리 존재 의의 약함. |
| **B-SOiD는 semi-supervised** | UMAP+HDBSCAN(unsup) + RandomForest(sup) 이중 구조인데 unsupervised/에 배치 |

---

## Alternative Taxonomies

### Option A: By Task (연구 관점)

```
models/
├── action_recognition/     # "행동을 분류하라" — 라벨 필요
│   ├── graph/                 InfoGCN, STGCN, AGCN, PySKL models
│   └── sequence/              MLP, LSTM, Transformer, RuleBased
├── representation/         # "표현을 학습하라" — 라벨 불필요
│   ├── ssl.py                 MAE, JEPA, DINO + encoders
│   └── behavemae.py           BehaveMAE (hierarchical)
├── behavior_discovery/     # "행동을 발견하라" — 라벨 불필요
│   ├── clustering.py          PCA→UMAP→KMeans
│   ├── bsoid.py               B-SOiD
│   ├── moseq.py               MoSeq
│   └── subtle.py              SUBTLE
├── modules/                # 공유 빌딩블록
│   ├── gcn.py                 SA-GC, TCN, EncodingBlock
│   └── encoders.py            GCN/InfoGCN/Interaction backbone
└── losses/
```

| 장점 | 단점 |
|------|------|
| 연구자 사고방식과 일치 ("뭘 하고 싶은가?") | 같은 GCN 아키텍처가 여러 폴더에 분산 |
| 논문 3-paradigm 구조와 직접 매핑 | 새 모델 추가 시 "이건 recognition인가 discovery인가?" 애매 |
| BehaveMAE 배치 문제 자연스럽게 해결 | modules/ 의존 관계 복잡해질 수 있음 |

### Option B: By Learning Paradigm (엄격한 단일 축)

```
models/
├── supervised/
│   ├── infogcn.py, stgcn.py, agcn.py    (graph)
│   ├── mlp.py, lstm.py, transformer.py   (sequence)
│   └── pyskl.py                          (external graph)
├── self_supervised/
│   ├── mae.py, jepa.py, dino.py          (methods)
│   ├── encoders.py                       (backbones)
│   └── behavemae.py                      (external)
├── unsupervised/
│   ├── clustering.py, bsoid.py, moseq.py, subtle.py
├── modules/                              (shared)
└── losses/
```

| 장점 | 단점 |
|------|------|
| 단일 축 → 배치 기준 명확 | InfoGCN이 supervised/에도, ssl encoder로도 사용 → 중복 |
| ML 교과서 분류와 일치 | sequence vs graph 구분이 사라져 탐색 불편 |

### Option C: By Architecture First (현재와 유사, 정리)

```
models/
├── graph/                  # 모든 GCN 기반 (자체 + 외부)
│   ├── infogcn.py, stgcn.py, agcn.py
│   ├── pyskl.py               (외부 wrapper)
│   └── modules.py             (공유 블록)
├── sequence/               # 모든 시퀀스 기반
│   └── classifiers.py
├── ssl/                    # SSL 방법론 (encoder + method)
│   └── ...
├── discovery/              # Behavior discovery (unsupervised)
│   ├── clustering.py, bsoid.py, moseq.py
│   ├── subtle.py, behavemae.py
└── losses/
```

| 장점 | 단점 |
|------|------|
| 현재 구조와 가장 가까워 마이그레이션 최소 | 여전히 2축 혼용 (architecture + paradigm) |
| PySKL → graph/에 통합 → external/ 제거 | BehaveMAE(SSL)가 discovery/에 있는 건 여전히 모호 |

---

## Comparison Matrix

| 기준 | **현재 (v0.1)** | **A: Task** | **B: Paradigm** | **C: Architecture** |
|------|----------------|-------------|-----------------|---------------------|
| 분류 축 | 3개 혼용 | Task(연구 목적) | Learning 방식 | Architecture + 2nd |
| BehaveMAE 배치 | ❌ 모호 | ✅ representation/ | ✅ self_supervised/ | ⚠️ discovery/ |
| STGCN 중복 | ❌ 2곳 | ⚠️ 1곳 (내부/외부 구분 없음) | ⚠️ supervised/ 내 구분 필요 | ✅ graph/에 통합 |
| 새 모델 추가 용이 | ⚠️ 어디에 넣을지 불명확 | ✅ task 기준 명확 | ✅ 패러다임 기준 명확 | ✅ 아키텍처 기준 명확 |
| 마이그레이션 비용 | — | 높음 (전면 재구조화) | 중간 | 낮음 (rename 수준) |
| 연구자 직관 | ⚠️ | ✅ 최고 | ✅ | ⚠️ |
| 개발자 직관 | ✅ | ⚠️ | ⚠️ | ✅ 최고 |

---

## Recommendation

### 현재 (v0.2): Option C 적용 완료

핵심 문제(external/ singleton, 이름 불일치) 해결됨:

1. ~~`external/pyskl.py`~~ → `graph/pyskl.py` (같은 GCN 계열 통합)
2. ~~`unsupervised/`~~ → `discovery/` (의도 명확화)
3. ~~`graph_models/`~~ → `graph/`, ~~`sequence_models/`~~ → `sequence/` (간결화)

### 향후 (v1.0): Option A (Task-based) 전환 고려

연구 플랫폼으로 성숙하면, 연구자 관점 분류(`action_recognition/`, `representation/`, `behavior_discovery/`)가 자연스러움.
`get_model()` factory가 있으므로 내부 구조 변경이 **API에 영향 없음**.

---

## Factory Interface (현재 구현)

```python
from behavior_lab.models import get_model, list_models

# Supervised graph
model = get_model('infogcn', num_classes=60, skeleton='ntu')
model = get_model('stgcn_pyskl', graph='ntu', num_classes=60)

# Supervised sequence
model = get_model('lstm', input_dim=42, hidden_dim=128, num_classes=4)

# Self-supervised
model = get_model('infogcn_dino', num_classes=60, skeleton='ntu')

# Unsupervised / Discovery
model = get_model('bsoid', fps=30)
model = get_model('moseq', project_dir='./out', num_iters=50)
model = get_model('behavemae', checkpoint_path='ckpt.pth', dataset='mabe22')

# List all
print(list_models())
# {'graph': [...], 'sequence': [...], 'ssl': [...],
#  'unsupervised': [...], 'external': [...]}
```

---

*Created: 2026-02-07 | behavior-lab v0.1*
