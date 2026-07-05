# INTEGRATION PLAN — Survey Methods (2026-07-05)

> Scope decision (bori 승인): **Option A — VAME wrapper 1개 신설 + 중복 비교 하네스 통합.** 새 하네스 금지, 기존 통합계약(`BehaviorClusterer` Protocol + `get_model` registry + `compare_discovery_methods` + `html_report`) 재사용. `/karpathy /ponytail --devil` 렌즈.
>
> 연계 서베이(정본): Obsidian `40_Areas/2_Research/_Surveys/BehaviorAnalysis/4_Papers/` (`_MoC_Behavior_Papers` + Paper_* 9 + Review_* 2).
> Status: **PLAN — 코드 미구현. 승인 후 착수.**

---

## 0. Devil's verdict — 서베이 방법 중 무엇을 넣나

| 방법 | 결정 | seam | 근거 |
|---|---|---|---|
| **VAME** | ✅ 신규 | `models/discovery/vame.py` (BehaviorClusterer) | 비교세트 유일 gap. RNN-VAE 임베딩→클러스터, keypoint 입력 = 기존 계약에 완벽 정합 |
| keypoint-MoSeq · MoSeq · B-SOiD · BehaveMAE · SUBTLE · CEBRA | ✅ 이미 통합 | 기존 | 재작업 불필요 |
| CLOSER · BehaVERT | ⏸ 보류 (Option B) | FeatureBackend | repo 미검증·데이터 특이(AVATAR/자체 curation). 이번 스코프 제외 |
| BEAST (2D) | ⏸ 보류 (Option C) | FeatureBackend | video-frame 입력 경로 신규 필요(skeleton 중심과 긴장) |
| **BEAST3D** | ❌ 제외 | — | 3DGS 32h×8 A100 학습 = **BehaviorSplatter 레포 소관**. behavior-lab(skeleton 클러스터링)과 부적합. 외부 baseline 문서로만 |
| **DANNCE** | ❌ 제외 | — | 상류 3D pose estimator. behavior-lab은 이미 `.mat` 출력을 `loaders/rat7m.py`·`li2023.py`로 소비. 모듈 중복 |

**핵심 통찰**: 실익은 "방법 추가"보다 **중복 비교 하네스 통합**. 이게 "일관된 방식·비교 시각화"의 실제 병목.

---

## 1. VAME 통합 — 5개 변경 (surgical)

`compare_discovery_methods`가 미등록 메서드를 `get_model(key).fit_predict(kp)`로 라우팅(`experiments/discovery.py:111-113`) → **레지스트리 등록만으로 비교·시각화에 자동 편입.**

### (1) 신규 `src/behavior_lab/models/discovery/vame.py`
`moseq.py` 패턴 복제 — lazy import, project-dir 기반, `fit_predict(keypoints)->ClusteringResult`. 필수는 `fit_predict` 하나(Protocol 준수 위해 fit/predict/get_embeddings/save/load도 얇게).

```python
"""VAME wrapper: RNN-VAE motion embedding + latent clustering.
Reference: Luxem et al. (2022), Communications Biology. Install: pip install vame-py
"""
import numpy as np
from ...core.types import ClusteringResult

class VAME:
    def __init__(self, project_dir='./vame_output', latent_dim=30,
                 time_window=30, n_clusters=15, num_epochs=100, ...):
        ...  # store config; self._model=None

    def fit(self, keypoints: np.ndarray) -> 'VAME':
        try:
            import vame
        except ImportError:
            raise ImportError("Install VAME: pip install vame-py "
                              "(or git+https://github.com/LINCellularNeuroscience/VAME.git)")
        # (T,K,D) -> VAME egocentric-aligned CSV/np → vame.init_new_project/train/segment
        ...
        return self

    def predict(self, keypoints) -> np.ndarray:   # (T,) motif labels
        ...
    def get_embeddings(self, keypoints) -> np.ndarray:   # (T, latent_dim)
        ...
    def fit_predict(self, keypoints: np.ndarray) -> ClusteringResult:
        self.fit(keypoints); labels = self.predict(keypoints)
        emb = self.get_embeddings(keypoints)
        return ClusteringResult(labels=labels, embeddings=emb[:, :2],
            n_clusters=len(set(labels)), features=emb,
            metadata={"algorithm": "vame", "latent_dim": self.latent_dim})
    def save(self, path): ...
    def load(self, path): ...
```
> ⚠ VAME는 자체 multiprocessing 사용 → macOS SIGSEGV 시 `subtle_wrapper.py::SUBTLEConfig(isolate=True)` subprocess 격리 패턴 재사용(신규 격리 메커니즘 금지).

### (2) 레지스트리 1줄 — `src/behavior_lab/models/__init__.py:34` `_DISCOVERY_MODELS`
```python
    'vame': ('behavior_lab.models.discovery.vame', 'VAME'),
```

### (3) 신규 `configs/model/vame.yaml` (moseq.yaml 복제)
```yaml
name: vame
_target_: behavior_lab.models.discovery.vame.VAME
project_dir: ./vame_output
latent_dim: 30
time_window: 30
n_clusters: 15
num_epochs: 100
```

### (4) catalog ModuleSpec — `src/behavior_lab/data/features/catalog.py` `DISCOVERY_METHODS`
input `(T,K,D)` / output `ClusteringResult` / module_path / deps `["vame-py"]` / strengths `자기지도 표현·계층 motif` / caveats `클러스터 수 비자동·latent blackbox·seed 민감`.

### (5) pyproject extra — `pyproject.toml [project.optional-dependencies]`
```toml
vame = ["vame-py"]   # + "all"/"external" 집계에 추가
```

### 흐름 (변경 없이 재사용)
`python scripts/discover.py model=vame dataset=calms21` · `compare_discovery_methods(kp, methods=(..., "vame"))` → `html_report.py`.

---

## 2. 중복 비교 하네스 통합 (ponytail cleanup — 실익 최대)

**현황: 비교 하네스 4개 병존**, 각자 result dataclass·per-method `run_*`·inline loader 재선언.

| 파일 | 상태 | 조치 |
|---|---|---|
| `src/behavior_lab/experiments/discovery.py::compare_discovery_methods` | ✅ 정본(clean, get_model·evaluation 재사용) | **유지·확장 (SSOT)** |
| `scripts/run_behavior_workbench_batch.py` (`METHODS` dict + `run_*`) | 현행 canonical batch | **`compare_discovery_methods` 호출로 재작성** (자체 METHODS 제거), `BatchResult`는 `DiscoveryRun`에서 파생 |
| `scripts/compare_clustering.py` (~800L, inline `load_calms21/subtle/...`) | 중복 최대 | **deprecate → thin shim**(`compare_discovery_methods` + `get_loader` 위임) 또는 삭제. inline loader는 `LOADER_REGISTRY`로 대체 |
| `scripts/benchmark.py` + `benchmark_compare.py` + notebooks `compare_methods*.py` | Hydra/노트북 | `compare_discovery_methods` 경유로 정렬(중복 `run_*` 제거) |

**원칙**: result dataclass는 `ClusteringResult`(per-run) + `DiscoveryRun`(비교행)만. `ModelResult`/`BatchResult`류 중복 struct는 파생/제거. 새 loader inline 금지 → `get_loader`.

> ⚠ 삭제/재작성은 test 리스크 → **각 파일 조치 전 diff 제시**, `pytest` + `scripts/test_e2e.py` green 확인 후 진행. `compare_clustering.py` 삭제는 별도 승인.

---

## 3. 명시적 OUT (이번 스코프 제외)
- **BEAST3D / Pose Splatter**: BehaviorSplatter 레포. 여기선 `docs/` 외부 baseline 표로만 참조.
- **DANNCE-the-network**: 이미 데이터로 소비. 신규 모듈 X.
- **CLOSER · BehaVERT · BEAST(2D)**: Option B/C에서 FeatureBackend로. 이번 제외.

---

## 4. 검증 계획
1. `pip install -e ".[dev]"` + VAME extra.
2. `pytest` (기존 green 유지) + VAME 단위테스트(작은 (T,K,D) fixture로 `fit_predict` smoke).
3. `compare_discovery_methods(kp, methods=("kmeans_pca_umap","bsoid","keypoint_moseq","subtle","behavemae","vame"))` → 9방법 일관 비교.
4. `html_report.py` → self-contained HTML 대시보드(embedding/ethogram/transition/metrics) 생성 확인.
5. `ruff check .` clean.

## 5. 리스크
- VAME 설치·API drift(pip `vame-py` vs github). 설치 실패 시 lazy ImportError로 격리(기존 패턴) — 코어 무영향.
- 하네스 통합 시 기존 스크립트 호출 시그니처 변경 → 노트북 참조 확인 필요.
- VAME multiprocessing macOS 크래시 → subprocess 격리 재사용.

---

## 6. 착수 순서 (승인 후)
1. `vame.py` wrapper + 5-edit → `pytest` green.
2. `compare_discovery_methods`에 VAME 포함 e2e + html_report 확인.
3. `run_behavior_workbench_batch.py` → `compare_discovery_methods` 재작성 (diff 제시).
4. `compare_clustering.py` deprecate/삭제 (별도 승인) → 최종 `pytest`+e2e+ruff.

*source: agent | 2026-07-05 | 연계: Obsidian _MoC_Behavior_Papers*
