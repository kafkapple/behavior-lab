# `~/dev` — 단일 진입점 (MoC)

> 여기서 "어디를 봐야 하는지"만 찾고, 내용은 링크된 정본에서 읽는다. **이 파일은 포인터만 담는다** —
> 사실을 여기 복사하면 그 순간부터 드리프트한다.
> 작성 2026-07-27 · 모든 경로는 작성 시점에 실재 확인함.
> **경로 표기**: 모두 `~/dev/` 기준 상대경로다 — 이 파일이 어디에 있든 동일하게 읽힌다.
>
> **실체는 `behavior-lab/docs/dev/` 안에 있고 `~/dev/` 의 셋(`README.md`·`DATASETS.md`·`STORAGE_GUIDE.md`)은
> 심볼릭 링크다.** `~/dev` 가 git 저장소가 아니라 추적이 안 되던 것을 260727 에 behavior-lab 으로 옮겨
> GitHub 백업에 태웠다. 편집은 어느 쪽을 열어도 같은 파일이다.

---

## 1. 저장소 지도

| 레포 | 역할 | 문서 진입점 | 위치 |
|---|---|---|---|
| **BehaviorSplatter** | GS-LRM 3D 재구성 · 4D deform (논문 본체) | `BehaviorSplatter/docs/references.md` → `BehaviorSplatter/docs/HANDOFF.md` | mac + gpu03 |
| **behavior-lab** | 포즈 로더 · `(T,K,D)` 정규화 · feature · 비지도 행동발견 · KP 벤치마크 | `behavior-lab/docs/README.md` | mac + gpu03 |
| **sdannce-poc** | 멀티뷰 어노테이션 · 6뷰 뷰어 · kp-guided SAM2 마스크 | `sdannce-poc/docs/README.md` | mac + gpu03 |
| **behavior-tools** | 영상 분할 · 프레임 추출 · 이미지 큐레이션 | `behavior-tools/docs/README.md` | mac only |
| **FaceLift** | 업스트림 레퍼런스 (KP22 정의 보유) | — | mac + gpu03 |
| **pose-splatter** | 별개 논문 구현 (arXiv 2505.18342) | `README.md` | gpu03 only |
| ~~mouse-kp-benchmark~~ | **deprecated** → behavior-lab 로 대체 | — | mac only |

**누가 무엇을 소유하는가** = 경계 문서 3부작 (서로 참조, 내용 중복 없음):
`sdannce-poc/docs/repo_boundary.md` ·
`behavior-tools/docs/behavior_lab_boundary.md` ·
`behavior-lab/docs/260727_repo_consolidation_audit.md`

---

## 2. 좌표 · 규약 SSOT

> **좌표 규약 단일 진입점 = 이 절.** 인용 주소는 `behavior-lab/docs/dev/README.md §2`.
> 별도 규약 문서를 새로 만들지 않는다 — 정의는 코드에 있어야 드리프트하지 않고, 여기는 "어느 코드냐"만 가리킨다.
> 좌표 관련 질문은 무조건 이 표에서 출발할 것.

가장 자주 사고가 나는 축. **코드 정본이 있으면 코드가 우선** — 문서만 있으면 드리프트한다.

| 규약 | 코드 정본 | 스펙 문서 |
|---|---|---|
| **포즈 텐서 `(T,K,D)`** — T=프레임, K=관절, D=좌표차원(2\|3) | `behavior-lab/src/behavior_lab/core/tensor_format.py` (+회귀 테스트 `tests/test_core/test_tensor_format.py`) | `behavior-lab/docs/architecture.md` §Data Format Specification |
| **GCN 텐서 `(N,C,T,V,M)`** | 위 파일 `sequence_to_graph` / `graph_to_sequence` | 위 문서 §Conversion Rules |
| **GS-LRM 카메라·좌표계** (OpenCV, World Z-up, fx=548.99) | — | `BehaviorSplatter/docs/camera_conventions.md` |
| **M5 좌표 변환 상수** (`M5_SCENE_CENTER`, `M5_DISTANCE_SCALE`) | `BehaviorSplatter/src/behaviorsplatter/notebooks/kp_utils.py` L47-48 — **하드코딩 금지, import 할 것** | `BehaviorSplatter/CLAUDE.md` §Camera Conventions |
| **DANNCE 투영 규약** (MATLAB K 비전치, cx·cy 3행) | `sdannce-poc/src/sdannce_utils/projection.py` | `sdannce-poc/docs/theory/projection_convention.md` |
| **SBeA P 행렬 + camera↔P 순서** | `sdannce-poc/segmentation/kp_sam2_sources.py` `_cal_sbea_p_matrix` | `sdannce-poc/configs/segmentation/sbea.yaml` 헤더 |
| **KP22 스켈레톤** (마우스 22관절) | `BehaviorSplatter/src/behaviorsplatter/temporal_deform/keypoints_22.py` `KP22_NAMES`·`SKELETON_BONES` | 업스트림 정의 `FaceLift/configs/keypoints/mouse_22.yaml` · 요약 `BehaviorSplatter/docs/keypoint_skeleton_conventions.md` |
| **rat23 스켈레톤** (s-DANNCE) | `sdannce-poc/src/sdannce_utils/constants.py` | — |
| **SBeA 16관절** | `behavior-lab/scripts/sbea_dlc_triangulate.py` `BODYPARTS` (저자 CSV 헤더 순서) | — |

**알아둘 소프트스팟**
- `BehaviorSplatter/scripts/preprocess_m5_subject_centered.py` L29-30 은 M5 상수를 import 하지 않고 재정의한다. 현재 값·수식은 동일해 문제 없으나, 한쪽만 고치면 갈라진다.
- s-DANNCE `.mat` 캘리브 파싱이 sdannce-poc(MATLAB 유지)와 BehaviorSplatter(OpenCV 전치) 두 곳에 있다. **수학적 등가이며 의도적 분리** — 근거는 `repo_boundary.md §의도적 중복 1건`.
- SBeA `camera-{i}.mp4` 는 `cam_mat_all[:,:,i]` 와 대응하지 않는다(미문서화). 세션마다 `sbea_dlc_triangulate.py` 가 24순열을 재투영 오차로 자동 판정하고, 그 결과를 config `camera_order` 에 기록해 생산자·소비자가 같은 값을 쓰게 한다.

---

## 3. 데이터 · 스토리지

| 주제 | 정본 |
|---|---|
| 데이터셋 온디스크 실사 (32개, 도메인별 경로·용량·포맷·shape) | `DATASETS.md` |
| 스토리지 3-tier 정책 (NFS vs NVMe vs HDD, 쓰기 경로 규칙) | `STORAGE_GUIDE.md` |
| 결과 폴더 SSOT (`~/results/{project}/`) | `CLAUDE.md` §5 |
| 코딩 원칙 | `CODING_PRINCIPLES.md` |

빠른 좌표: `~/data` → `/node_data/joon/data` (심링크). `~/dev/datasets` 는 심링크 1개짜리 스텁이지 데이터셋 허브가 아니다.

---

## 4. 리포트 (HTML, 자체 완결형)

| 리포트 | 내용 | 경로 |
|---|---|---|
| SBeA 파이프라인 | 저자 DLC → DLT 삼각측량 → SAM2. 2D 오버레이·잔차 분포·3D 스켈레톤·마스크 | `behavior-lab/outputs/sbea_report.html` |
| SAM2 변종 대조 (Rat7M) | v1 vs v4 파라미터 판정 — IoU 0.368, 연결성분 51→631 | `sdannce-poc/outputs/sam2_variant_compare.html` |
| SAM2 변종 대조 (s-DANNCE) | segment v1 vs v2 — IoU 0.970, 사실상 동일 | `sdannce-poc/outputs/sam2_segment_variant_compare.html` |

재생성: `behavior-lab/scripts/render_sbea_report.py` · `sdannce-poc/scripts/compare_sam2_variants.py`
(`outputs/` 는 gitignore 대상 — 리포트는 재생성 가능한 산출물이지 소스가 아니다.)

---

## 5. 파이프라인 한눈에

```
영상 + 캘리브
   └─ behavior-lab  scripts/sbea_dlc_triangulate.py     DLC 2D → DLT → (T,K,3) npz
        └─ sdannce-poc  segmentation/kp_sam2.py         키포인트 프롬프트 → mask npz
             └─ BehaviorSplatter  scripts/preprocess_run.py   (mask.source=npz_dir)
                  └─ opencv_cameras.json + RGBA 512² → GS-LRM 학습
```

레포 간 인계는 **파일 포맷**이지 코드 import 가 아니다. mask npz 키 규약(`animal_N` 쓰기 / `mask`·`ratN` 도 읽기)은
`sdannce-poc/docs/repo_boundary.md` §mask npz 키 규약 참조.

데이터셋 추가 = `sdannce-poc/configs/segmentation/*.yaml` 1개 (캘리브·키포인트·영상레이아웃 3축이 기존 조합이면 코드 0줄).
