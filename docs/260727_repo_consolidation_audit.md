# 260727 — behavior-* 저장소 통합 감사 + sdannce-poc 역할 규명

> 조사 범위: gpu03 `~/dev/` 의 behavior-lab 계열 4개 + mac `~/dev/behavior-tools` + `~/dev/sdannce-poc`.
> 방법: git ancestry 검증(`merge-base --is-ancestor`) · 전체 트리 `diff -rq` · 산출물 npz 헤더 확인. 실사일 2026-07-27.

---

## TL;DR

- behavior-lab 계열 "중복 4개"는 실제로 **정본 1 + 완전 후행 1 + 유물 1 + 의도적 분리 1** — 무조건 병합은 오답.
- 삭제 가능 2개(`behavior-lab-kp`, `behavior-lab_stale_260705`), 단 **선(先) 산출물 이관 필요** — kp 벤치마크 결과 11M + GT 데이터 2.2M 은 git 에 없음.
- sdannce-poc 는 behavior-lab 의 중복 아님 — 겹치는 건 클러스터링 1개 스크립트뿐, 나머지 8,000 LOC(6뷰 뷰어·SAM2)는 고유.

---

## 1. 현황 인벤토리

| 디렉토리 | 머신 | git | HEAD | 크기 | 판정 |
|---|---|---|---|---:|---|
| `behavior-lab` | gpu03 + mac | `kafkapple/behavior-lab` main | `713f7f0` (07-16) | 8.5G | **정본** |
| `behavior-lab-kp` | gpu03 only | 같은 repo, main | `667ccd6` (06-03) | 15M | 코드 후행(44 커밋), 산출물만 고유 |
| `behavior-lab_stale_260705` | gpu03 only | **없음** (plain dir) | — | 3.8M | 2월 스냅샷 유물 |
| `behavior-tools` | **mac only** | `kafkapple/behavior-tools` | `871f98f` (02-08) | 1.8M | 별개 repo, 경계 문서화됨 |
| `BehaviorSplatter` | gpu03 + mac | 별개 | — | — | 3DGS, 범위 밖 |

mac `behavior-lab` = gpu03 와 동일 HEAD, 워킹트리 clean → 이미 동기화됨.

### 1.1 behavior-lab-kp — 코드는 완전 후행

- `git merge-base --is-ancestor 667ccd6 713f7f0` → **참**. kp 의 HEAD 가 정본 히스토리에 포함, 그 사이 44 커밋.
- 워킹트리 clean(수정 0, untracked 2). untracked 2개(`scripts/03_infer_dlc.sh`, `04_zeroshot_superanimal.sh`)는 정본에 **더 최신 버전 존재**(06-04, 각각 5574B/6616B vs kp 5367B/6179B) → 폐기 대상.
- kp 계열 코드(`benchmark_kp_dlc.py`·`prepare_kp_splits.py`·`render_kp_*.py`)는 정본에 모두 있음. 문서도 `docs/kp_benchmark_v0.1.md` 로 존재.

**고유 자산 (git 미추적, 소실 시 복구 불가)**

| 경로 | 크기 | 내용 |
|---|---:|---|
| `outputs/kp_benchmark/dlc_resnet50_imagenet_full_kp.npz` | — | `keypoints_3d(18000,22,3)` · `valid_mask(18000,22)` · `frame_ids` · `keypoint_names(22)` |
| `outputs/kp_benchmark/dlc_superanimal_zeroshot_hrnet_w32_full_kp.npz` | — | 동일 스키마 (zero-shot 대조군) |
| `outputs/kp_benchmark/{results.csv, per_kp_error.csv}` | 11M 계 | 벤치마크 집계 결과 |
| `data/mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz` | 968K | `keypoints(3600,22,3)` |
| `data/markerless_mouse_1/labels/li_m1_gt.npz` | 1.2M | `keypoints_3d(81,22,3)` GT + `label3d_dannce.mat` |
| `data/splits/{li_m1_external,mammal_m1_train,mammal_m1_test}.csv` | 36K | 분할 정의 |

> 이 산출물의 학습 런 원본 9.3G = `/node_data/joon/behavior-lab-kp-benchmark/` (repo 밖, 영향 없음).

### 1.2 behavior-lab_stale_260705 — 유물

- 정본 대비 전체 트리 diff: **stale-only 5건 / 내용 상이 17건 / 나머지 동일**. 상이 17건은 전부 2월판 구버전.
- stale-only 5건 처리:

| 항목 | 처리 | 근거 |
|---|---|---|
| `docs/260705_gpu03_experiment_setup.md` | **살려야 함** | 정본 git 히스토리에 존재한 적 없음. gpu03 정찰 + git 동기화 블로커 분석 의사결정 기록 |
| `configs/pose/` | 폐기 | 빈 디렉토리 (파일 0개) |
| `scripts/compare_clustering.py` | 폐기 | `4d4119f "refactor(viz): unified comparison report on shared modules, drop dupes"` 에서 **의도적 삭제** — git 복구 가능 |
| `scripts/generate_cluster_report.py` | 폐기 | 동상 |
| `.DS_Store` | 폐기 | 잡파일 |

### 1.3 behavior-tools — 병합 금지

`docs/behavior_lab_boundary.md` 에 소유권이 명시적으로 문서화되어 있음:

| 담당 | 저장소 |
|---|---|
| 멀티뷰 영상 분할·프레임 추출, 이미지 큐레이션(CLIP/DINO), SAM 어노테이션, super-resolution | behavior-tools |
| pose 로더·`(T,K,D)` 정규화, feature 추출, 비지도 행동 발견, motif 전이 분석 | behavior-lab |

→ **중복이 아니라 설계된 분리**. 통합하면 이 경계 문서가 무효화됨.

**단, SSOT 충돌 1건 — 260727 해소**: behavior-tools 가 "SAM 어노테이션" 소유권을 주장했으나 실체는 SAM1 기반이었음.
⚠️ **정정**: 최초 감사에서 이를 "199 LOC 스텁"으로 적었으나 오독이었다 — 본문은 `sam_model_registry`/`SamPredictor` 를 쓰는
완성 코드였고, 스텁처럼 보인 건 docstring 에 남은 `"TODO: Port from gpu03:~/dev/mouse-super-resolution/sam_annotator"`
한 줄 때문이었다(해당 경로는 실제로 없음, 확인 완료).
실질 근거는 다른 데 있었다: (a) 레포 안팎 어디서도 import 되지 않음, (b) 의존 패키지 `segment-anything` 이 gpu03 어느
env 에도 미설치 → 실행 불가, (c) sdannce-poc 에 SAM2 실동작 구현 존재.
→ behavior-tools `annotator/` 삭제 + 경계 문서의 소유권 행을 sdannce-poc 로 이관 (behavior-tools `e98c208`).

---

## 2. 통합 실행 계획 (승인 필요)

> 삭제 대상 파일 3개 이상 → `~/.claude/CLAUDE.md` §3.1 하드게이트. 아래는 **제안**이며 미실행.

**Step 1 — kp 산출물 이관 (비파괴)**
```
mkdir -p ~/dev/behavior-lab/outputs/kp_benchmark ~/dev/behavior-lab/data
mv ~/dev/behavior-lab-kp/outputs/kp_benchmark/* ~/dev/behavior-lab/outputs/kp_benchmark/
mv ~/dev/behavior-lab-kp/data/{mammal_mouse,markerless_mouse_1,splits} ~/dev/behavior-lab/data/
```
정본 `.gitignore` 가 `/data/`·`/outputs/` 를 제외하므로 커밋 대상 아님 — 파일시스템 이동만.

**Step 2 — stale 문서 살리기 (비파괴 + 커밋)**
```
cp ~/dev/behavior-lab_stale_260705/docs/260705_gpu03_experiment_setup.md ~/dev/behavior-lab/docs/
```
이 파일 + 본 감사 문서를 함께 커밋 → GitHub 백업으로 영속화.

**Step 3 — 디렉토리 제거 (파괴적, 명시 승인 필요)**
```
rm -rf ~/dev/behavior-lab-kp ~/dev/behavior-lab_stale_260705
```
Step 1·2 완료 확인 후에만. 두 디렉토리 모두 gpu03 전용이라 mac 영향 없음.

**Step 4 — behavior-tools 경계 정정 — ✅ 완료 (`e98c208`)**
`annotator/` 삭제 + pyproject extra 제거 + `docs/behavior_lab_boundary.md` 소유권 행을 sdannce-poc 로 이관.
`superres/` 는 유지 — 대체재가 없고 코드도 완성 상태(같은 stale TODO 만 제거).

---

## 3. sdannce-poc — 역할과 중복 여부

### 3.1 역할

Social DANNCE 3D 키포인트 PoC. 상위 프로젝트 = BehaviorSplatter (FaceLift NeurIPS 2026 제출).
근거 논문 = Klibaite et al., *Mapping the landscape of social behavior*, Cell 188(8), 2025.

세 층으로 구성:

| 층 | 규모 | 내용 |
|---|---:|---|
| `src/sdannce_utils/` | 5 모듈 | `constants`(23관절·스켈레톤 엣지) · `io`(키포인트/캘리브/어노테이션 로드) · `projection`(DANNCE MATLAB 규약 — K 전치 안 함) · `config` |
| `viewers/` | 5,170 LOC | 6뷰 웹 뷰어(1699) · 마스크 어노테이터(851) · 오프라인 렌더러(846) · 멀티뷰 그리드(560) · L1/L2 접촉 비교(582) · 스레드안전 CameraPool(233) |
| `segmentation/` | 2,852 LOC | SAM2 — 키포인트 유도 자동 분할 · 전체/범위/희소 전파 · 청크 전파 · Gradio 어노테이터 · FastAPI 서비스 |

데이터: `2022_09_22_M3_M4` (SCN2A 사회성 랫 쌍), 1920×1200 · 50fps · 90K 프레임 · 6 동기 카메라 · 개체당 23관절.

### 3.2 중복 판정

| 축 | 판정 | 근거 |
|---|---|---|
| 6뷰 뷰어 / 마스크 어노테이션 UI | **고유** | 다른 어느 repo에도 대응물 없음 |
| DANNCE 투영 규약 (`projection.py` + `docs/theory/projection_convention.md`) | **고유** | behavior-lab 은 3D 재구성 규약을 다루지 않음 |
| SAM2 분할 | **해소됨** | behavior-tools 의 SAM1 모듈 삭제, 소유권 sdannce-poc 로 이관 (§1.3) |
| 클러스터링 (`scripts/run_clustering_poc.py`) | **실제 중복 (경미)** | KMeans/HDBSCAN + TPI·엔트로피·bout 지표. behavior-lab 은 VAME·MoSeq·B-SOiD·SUBTLE·BehaveMAE 보유 + **이미 `outputs/sdannce/{vame,kpms}/labels.npy` 산출** + `experiments/pipeline.py:44` 에 `"sdannce"` DatasetSpec 등록. 경계 문서상 "비지도 행동 발견 = behavior-lab" |

**단, 통합의 전제조건이 아직 미충족**: behavior-lab `data/loaders/` 에 s-DANNCE 로더가 **없음** (calms21·li2023·mabe22·mammal_mouse·ntu_rgbd·nwucla·rat7m·shank3ko·sleap·subtle 만 존재). `pipeline.py` 는 sdannce 를 rat7m 스켈레톤에 매핑하는 스펙 한 줄뿐.
→ 260705 문서의 "s-DANNCE loader 부재" 지적이 **260727 현재도 유효**. 로더 없이 클러스터링만 옮기면 파이프라인이 끊김.

**권고**: sdannce-poc 는 독립 유지. 클러스터링 일원화를 원하면 순서는 ① behavior-lab 에 sdannce 로더 추가 → ② `run_clustering_poc.py` 를 behavior-lab 호출로 대체 → ③ sdannce-poc 에서 제거. 로더 없이 ③ 부터 하면 안 됨.

### 3.3 상태 리스크

- 마지막 커밋 **2026-03-23**, 이후 워킹트리에 7개 미커밋 (수정 2 + untracked 5).
- untracked = `kp_sam2_rat7m.py`(274 LOC) · `kp_sam2_rat7m_v4.py`(275) · `kp_sam2_segment_v2.py`(258) · `run_rat7m_sam2_poc.sh` · `configs/data/session_rat7m_s4_d1.yaml` — 04-15 ~ 05-29 작업분.
- 위협 모델: origin 이 GitHub 이므로 커밋분은 안전하나, 이 807 LOC + 설정은 gpu03 `/home/joon`(NFS) 단일 사본. 디렉토리 손실 시 재현 불가.
- 동일 유형 선례 = PS `convert_m5_for_ps.py` 미커밋 소실(34일 정체). `~/dev/CLAUDE.md` §4.1 "작업 단위마다 커밋" 위반 상태.
- 🟡 **커밋 권고** — 4개월 방치된 미커밋 실험 코드.

---

## 4. Red Team 검토 (자기 계획 반박)

| 반론 | 검증 결과 |
|---|---|
| "ancestor 검사는 추적 파일만 커버 — kp 워킹트리에 미커밋 변경이 있으면 삭제 시 소실" | 타당한 지적. `git status --porcelain` = 2줄, 둘 다 untracked이며 정본에 더 최신판 존재. **반박 해소** |
| "trees diff 에서 `outputs` 를 제외했으니 stale 산출물을 놓쳤을 수 있음" | stale 최상위에 `outputs/` 자체가 없음(총 파일 256개, ls 확인). **해소** |
| "behavior-* 4개니까 다 합치면 깔끔" | **기각** — behavior-tools 경계 문서가 존재. 병합은 설계 결정을 되돌리는 것 |
| "kp 는 15M 밖에 안 되니 그냥 둬도 무해" | 부분 타당. 다만 같은 repo·같은 브랜치명으로 44커밋 후행 사본이 남아 있으면 다음 세션이 잘못된 사본에서 작업할 위험. 이관+삭제가 낫되 **긴급도는 낮음** |
| "sdannce-poc 클러스터링을 behavior-lab 으로 즉시 이관" | **기각** — 로더 부재로 파이프라인 단절. §3.2 순서 준수 필요 |

**미검증 항목 (정직 표기)**: `HOT:preprocessed/FaceLift_mouse`(9.1G) 와 `COLD:preprocessed/FaceLift_mouse` 의 동일성 미확인 — 중복 회수 가능성은 있으나 이번 감사에서 체크섬 비교 미수행.
