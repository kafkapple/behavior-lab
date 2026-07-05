# INTEGRATION PLAN — Heavy Pretrained Model Harness (Module 2, 2026-07-05)

> Module 1 (VAME + 하네스 수렴) 다음 모듈. bori 요청: CLOSER·BehaVERT·BEAST(2D)·hBehaveMAE를 **"무거운 모델 하네스" 카테고리로 별도 묶어**, 우선순위순 근거기반 구현/테스트. **동일 멀티뷰 데이터로 비교·시각화.** `/karpathy /ponytail /research proposal-review`.
> Status: **PLAN (structural) — 코드 미구현. per-model priority/dataset은 repo·data 검증 후 확정(§3, §4).**

---

## 1. 왜 별도 카테고리인가 (devil / ponytail)

기존 `discovery/`(경량, 자체 클러스터링 소유)와 성격이 다름:
- 이들은 **대규모 사전학습 표현 모델** — 무거운 학습/추론(BEAST 25h×8 A40, BEAST3D 32h×8 A100), 체크포인트 의존, GPU 필요.
- 출력이 **discrete label이 아니라 embedding** → 클러스터링은 하류(`cluster_features`)에 위임.
- ⇒ `BehaviorClusterer`(discovery seam)가 아니라 **`FeatureBackend`(embedding seam)** 에 정합.

**ponytail: 새 base class 만들지 않는다.** behavior-lab에 이미 `FeatureBackend.extract(windows)->(N,D)` 존재(CEBRA·DINOv2·Morlet). 무거운 모델도 전부 이 계약으로 → `cluster_features` → `DiscoveryRun`/`html_report` = **동일 비교·시각화 하네스 재사용.**

## 2. 카테고리 설계 — `pretrained_embedder`

- **명칭**: catalog category `"pretrained_embedder"` (heavy pretrained representation models). 신규 subpackage 불필요 — 위치는 `data/features/`(기존 backend와 동거) 또는 얇은 `models/pretrained/` 네임스페이스. **ponytail 기본안: `data/features/{beast,closer,behavert}_backend.py` + hBehaveMAE는 기존 유지 + catalog에 `pretrained_embedder` 재분류.**
- **공통 계약** (기존 `FeatureBackend`):
  ```python
  class XBackend:  # FeatureBackend
      def __init__(self, checkpoint_path=None, device="cpu", ...): ...
      def extract(self, data) -> np.ndarray:   # (N, D) per-frame or per-window embedding
          # lazy import heavy dep; load pretrained ckpt; forward
  ```
- **비교 흐름 (변경 없이 재사용)**:
  `extract()` → `cluster_features(emb)` → `compare_discovery_methods`/batch runner → `html_report`. 경량(VAME·B-SOiD·MoSeq)과 **한 표에서 동일 데이터로 비교.**
- **입력 modality 차이(핵심)**:

| 모델 | 입력 | behavior-lab 정합 |
|---|---|---|
| hBehaveMAE | windowed **keypoints** | ✅ 기존 (`behavemae.py::encode`) |
| CLOSER | **3D pose** (ST-GCN) | keypoint backend 정합 (AVATAR 9kp) |
| BehaVERT | **keypoint** (BERT token) | keypoint backend 정합 |
| BEAST (2D) | **raw video frames** | ⚠ video-frame 로더 신규 필요 (skeleton 중심과 유일한 마찰점) |

→ keypoint 3종은 기존 `(T,K,D)` 파이프 재사용, BEAST만 video 경로 추가.

## 3. 우선순위 (근거기반 — repo 실측 완료 2026-07-05)

> 원칙: **코드·체크포인트 공개도(재현성) × 통합 공수 × behavior-lab 정합도.**

| 순위 | 모델 | 실측 근거 | 판정 |
|---|---|---|---|
| **1** | **hBehaveMAE** | ✅ 이미 통합(`behavemae.py`, test 31 pass) + **공개 가중치**(Zenodo 13790191) · pose 입력 | **즉시** — 가중치 다운로드만 |
| **2** | **BEAST** | ✅ code 공개(paninski-lab/beast 27★, `pip install beast-backbones`) · `predict_video(save_latents=True)` = embedding API · **체크포인트 비공개**(rig마다 자가 pretrain) · **video 입력** | 중 — video 경로 신규 + pretrain 필요 |
| **3** | **CLOSER** | ⚠️ code 있으나 취약(0★, 가중치 미확인·9kp AVATAR 하드코딩) · Zenodo data 공개(1.1+4.7GB) · pose 입력 | stretch — kp 스켈레톤 편집+재학습 필요 |
| — | **BehaVERT** | 🔴 **repo 빈 껍데기(vaporware)** — `bert_models.py` 등 미존재, import 실패, 체크포인트·데이터 0 | **제외** — 코드 릴리스 시 재검토 |

**핵심 정정(devil)**: "4모델 동일 데이터 비교"는 **오늘 실행 불가** — 실제 runnable = **BEAST(video) + hBehaveMAE(pose)** 2종. CLOSER는 stretch, BehaVERT는 불가. 계획을 이 현실에 맞춤.

## 4. 동일 멀티뷰 데이터 실험 (proposal-review 확정)

**데이터셋 top-3** (멀티뷰 + 3D pose GT + 공개, 실측):
| 순위 | 데이터셋 | 종 | cam | video | 3D pose | 다운로드 | 크기 |
|---|---|---|---|---|---|---|---|
| 1 | **s-DANNCE** | rat+mouse | 6 | ✅ | 23kp + **행동라벨(HLAC)** | Harvard Dataverse `socialDANNCE_data` | ~197GB |
| 2 | **SBeA** | mouse(Shank3B) | 4 | ✅ | 16kp | figshare 30338953/30338929 | ~28GB (gpu03 보유) |
| 3 | **DANNCE Mouse** | mouse | 6 | ✅(100fps) | 22kp(310 GT frame) | Duke 10.7924/r4hq43h4c | 소(GT)/대(raw) |

> ⚠️ 정정: sDannce 노트의 "Zenodo 14629232=데이터"는 **코드**. 실제 데이터=Harvard Dataverse. (Obsidian 노트 정정 대상)

**실험 프로토콜** (단계별, 근거기반):
1. **P0 (즉시, GPU 무관)** — hBehaveMAE 가중치(Zenodo 13790191) 다운로드 → 기존 `run_behavemae` 경로로 SBeA(gpu03 보유)에서 embedding → `cluster_features` → `compare_discovery_methods`에 경량(VAME·MoSeq·B-SOiD·KP-MoSeq)과 **동일 표** 비교.
2. **P1 (GPU)** — BEAST: `beast extract`로 SBeA/s-DANNCE video → `beast train`(rig별 pretrain, GPU) → `predict_video(save_latents=True)` → latent를 FeatureBackend로 래핑(`data/features/beast_backend.py`, VAME 패턴) → 동일 비교표 편입.
3. **P2 (stretch)** — CLOSER: Zenodo data + ST-GCN `inward` edge를 대상 kp 수로 편집 → 재학습 → `extract_embedding` → 비교.
4. **평가** — silhouette + ARI/NMI(라벨 있으면; s-DANNCE HLAC 활용) + bout/transition + `html_report` 단일 대시보드. **외부 앵커**(s-DANNCE 행동라벨) 우선 = 비지도 ground-truth 문제 회피(landscape 리뷰 3원칙).

**구현 상태**: 통합 seam은 `FeatureBackend.extract()->(N,D)` 재사용(신규 base class 금지). BEAST/CLOSER wrapper는 **GPU+data 확보 후** 착수(VAME처럼 wiring test 먼저 + 실행은 GPU-gated). 지금 speculative video-wrapper 코드는 karpathy 원칙상 미작성.

## 5. 검증 (Bash 실행 필요 — 현재 세션 제약)
`pip install -e ".[dev]"` → `pytest` → `compare_discovery_methods`(경량+heavy) e2e → `html_report` → `ruff check .`.
⚠ 현재 세션은 Bash 차단 → 사용자 실행 또는 권한 허용 필요.

---
*source: agent | 2026-07-05 | 연계: INTEGRATION_PLAN_survey_methods.md · Obsidian _MoC_Behavior_Papers*
