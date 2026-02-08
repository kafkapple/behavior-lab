# behavior-lab

Modular skeleton-based behavior recognition platform for humans and animals.

Supervised, Self-supervised, Unsupervised 3가지 학습 패러다임을 단일 `get_model()` factory로 통합.

## Features

- **30+ models** — Graph (InfoGCN, ST-GCN, PySKL), Sequence (LSTM, Transformer), SSL (MAE/JEPA/DINO), Discovery (B-SOiD, MoSeq, SUBTLE, BehaveMAE)
- **7 skeleton species** — Human (NTU 25j, UCLA 20j, COCO 17j), Mouse (CalMS21 7j, DLC 27j), Quadruped (39j)
- **Canonical format** — `(T, K, D)` 통일 표현, GCN 전용 `(N, C, T, V, M)` 자동 변환
- **Multi-person/animal** — CalMS21 2-mice, NTU 2-person 색 구분 시각화
- **Self-contained HTML report** — base64 임베딩 이미지, 탭 네비게이션, 단일 파일 결과 리포트
- **Hydra config** — 재현 가능한 실험 관리

## Installation

```bash
git clone https://github.com/kafkapple/behavior-lab.git
cd behavior-lab

# Core only
pip install -e .

# With visualization + clustering
pip install -e ".[viz,clustering]"

# Full (torch + viz + clustering + B-SOiD + loaders)
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

**Python >= 3.10** required. Conda environment recommended.

## Quick Start

### Python API

```python
from behavior_lab.models import get_model, list_models
from behavior_lab.data.loaders import get_loader
from behavior_lab.core.skeleton import get_skeleton

# List all available models
print(list_models())

# --- Data Loading ---
loader = get_loader("calms21", data_dir="data/calms21")
sequences = loader.load_split("train")  # List[BehaviorSequence]
# Each: keypoints (T, K, D), labels (T,), skeleton_name, fps

# --- Behavior Discovery (B-SOiD) ---
import numpy as np
all_kp = np.concatenate([s.keypoints for s in sequences], axis=0)

bsoid = get_model("bsoid", fps=30)
result = bsoid.fit_predict(all_kp)
print(f"Clusters: {result.n_clusters}, Labels: {result.labels.shape}")

# --- Visualization ---
from behavior_lab.visualization import plot_skeleton, animate_skeleton

skeleton = get_skeleton("calms21")
fig, ax = plot_skeleton(
    sequences[0].keypoints, skeleton=skeleton,
    frame=0, show_labels=True,
)

# --- Supervised (InfoGCN) ---
model = get_model("infogcn", num_classes=60, skeleton="ntu")
```

### CLI (Hydra)

```bash
# Supervised training
python scripts/train.py model=infogcn dataset=ntu60_xsub skeleton=ntu25

# SSL pretraining
python scripts/train_ssl.py model=infogcn_dino dataset=calms21

# Behavior discovery
python scripts/discover.py model=bsoid dataset=calms21

# E2E verification (CalMS21 + NTU + NW-UCLA)
python scripts/test_e2e.py
```

## Datasets

| Dataset | Species | Joints | Persons | Classes | Train | Test |
|---------|---------|--------|---------|---------|-------|------|
| **CalMS21** | Mouse | 7 | 2 | 4 (attack, investigation, mount, other) | 19,144 | 4,787 |
| **NTU RGB+D 60** | Human | 25 | 1-2 | 60 (drink, eat, clap, ...) | ~40K | ~16K |
| **NW-UCLA** | Human | 20 | 1-10 | 10 (pick up, throw, ...) | 1,020 | 464 |

Data files go in `data/{dataset_name}/` (not tracked by git).

## Model Taxonomy

```
get_model('name')
    |
    +-- graph/         GCN family
    |   +-- InfoGCN, STGCN, AGCN (self)
    |   +-- ST-GCN++, CTR-GCN, MS-G3D, AAGCN (PySKL)
    +-- sequence/      MLP, LSTM, Transformer
    +-- ssl/           3 encoders x 3 methods = 9 models
    |   +-- {GCN, InfoGCN, Interaction} x {MAE, JEPA, DINO}
    +-- discovery/     B-SOiD, MoSeq, SUBTLE, BehaveMAE, clustering
```

| Scenario | Recommended |
|----------|-------------|
| Labeled + graph topology | `infogcn`, `stgcn`, `ctrgcn_pyskl` |
| Labeled + fast experiment | `lstm`, `transformer` |
| Unlabeled + feature learning | `infogcn_dino` (SSL) |
| Unlabeled + behavior discovery | `bsoid`, `moseq`, `subtle` |
| Multi-subject interaction | `infogcn_interaction` |

## Project Structure

```
behavior-lab/
+-- src/behavior_lab/
|   +-- core/              Skeleton registry, graph, tensor format (numpy only)
|   +-- data/
|   |   +-- loaders/       CalMS21, NTU, NW-UCLA, Rat7M parsers
|   |   +-- preprocessing/ Pipeline (interpolate, smooth, normalize) + augmentation
|   |   +-- feeders/       PyTorch Dataset (SkeletonFeeder)
|   +-- models/            30+ models via get_model() factory
|   +-- training/          Hydra-config Trainer
|   +-- evaluation/        Classification, Cluster, Behavior metrics
|   +-- visualization/     Colored skeleton, embedding, HTML report
+-- scripts/               train, evaluate, discover, benchmark, test_e2e
+-- configs/               Hydra config groups (skeleton, dataset, model, training)
+-- docs/                  Theory, architecture, guides
+-- tests/                 Unit + integration tests
```

## Documentation

| Document | Description |
|----------|-------------|
| [Overview](docs/overview.md) | Research questions, hypotheses, methodology |
| [Architecture](docs/architecture.md) | Module map, data format, config system |
| [Model Taxonomy](docs/model_taxonomy.md) | 30+ model catalog with classification analysis |
| [Quick Start Guide](docs/guides/quickstart.md) | Step-by-step experiment guide |
| [E2E Verification](docs/e2e_verification.md) | Real-data pipeline verification results |
| [Graph Models](docs/theory/graph_models.md) | GCN -> ST-GCN -> InfoGCN theory |
| [SSL Methods](docs/theory/ssl_methods.md) | MAE vs JEPA vs DINO for skeleton |
| [Cross-Species](docs/theory/cross_species.md) | Canonical 5-part skeleton mapping |
| [Evaluation](docs/theory/evaluation.md) | Metrics theory (NMI, ARI, Hungarian) |

## E2E Verification Results

```bash
python scripts/test_e2e.py
```

Outputs in `outputs/e2e_test/`:

| Output | Description |
|--------|-------------|
| `report.html` | Self-contained HTML report (open in browser) |
| `report.md` | Markdown summary |
| `report.json` | Machine-readable metrics |
| `calms21/sample_skeleton.png` | Colored 2-mice skeleton |
| `calms21/sample_animation.gif` | Multi-person animated GIF |
| `ntu/sample_skeleton.png` | Body-part colored human skeleton |
| `nwucla/sample_skeleton.png` | 20-joint colored skeleton |

## License

MIT
