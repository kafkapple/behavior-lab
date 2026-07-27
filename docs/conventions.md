# 좌표 · 규약 SSOT

> **여기가 단일 진입점이다.** 좌표계·텐서 형태·스켈레톤 정의에 관한 질문은 이 문서에서 출발한다.
>
> **원칙**: 값의 정본은 항상 **코드**다. 이 문서는 "어느 코드가 정본인지"와 "그 코드가 지금 무엇을
> 주장하는지"를 적는다. 아래 `checked` 블록의 값은 `tests/test_conventions_doc.py` 가 실제 소스와
> 대조하므로, 코드를 바꾸고 문서를 안 고치면 **테스트가 깨진다**. 손으로 맞추지 말 것.
>
> 작성 2026-07-27 · 사본 위치 무관하게 읽히도록 모든 경로는 `~/dev/` 기준.

---

## 1. 포즈 텐서

| 규약 | 형태 | 의미 |
|---|---|---|
| **Canonical** | `(T, K, D)` | T=프레임 · K=관절 · D=좌표차원(2=픽셀, 3=월드) |
| **Graph (GCN 전용)** | `(N, C, T, V, M)` | N=배치 · C=채널(=D) · V=정점(=K) · M=개체 |

- 코드 정본: `behavior-lab/src/behavior_lab/core/tensor_format.py`
  — `sequence_to_graph()` / `graph_to_sequence()`
- 상세 스펙: `behavior-lab/docs/architecture.md` §Data Format Specification
- 회귀 테스트: `behavior-lab/tests/test_core/test_tensor_format.py`

모든 로더는 원본 포맷(HDF5·pkl·mat·npz)을 `(T, K, D)` 로 정규화해 내보낸다. 이 계약이 레포 경계선이다 —
정규화 **이전**(픽셀·카메라)은 sdannce-poc, **이후**(feature·행동발견)는 behavior-lab.

## 2. 스켈레톤

| 이름 | 관절 | 간선 | 코드 정본 | 쓰는 곳 |
|---|---:|---:|---|---|
| **KP22** (마우스) | 22 | 21 | `BehaviorSplatter/src/behaviorsplatter/temporal_deform/keypoints_22.py` — `KP22_NAMES`·`SKELETON_BONES` | BehaviorSplatter, MAMMAL, Li2023 |
| **rat23** (s-DANNCE) | 23 | 22 | `sdannce-poc/src/sdannce_utils/constants.py` — `KP_NAMES`·`SKELETON_EDGES` | sdannce-poc, s-DANNCE 세션 |
| **SBeA16** | 16 | 15 | `behavior-lab/scripts/sbea_dlc_triangulate.py` — `BODYPARTS`·`render_sbea_report.py::EDGES` | SBeA |

- KP22 업스트림 정의: `FaceLift/configs/keypoints/mouse_22.yaml` (가장 풍부) · 요약 `BehaviorSplatter/docs/keypoint_skeleton_conventions.md`
- **해부학 배선 규칙**(3종 공통): 전지는 흉추/목에, 후지는 천추/꼬리뿌리에 붙는다. 네 다리를 몸통 중앙 한 점에
  모으지 않는다. 260702 에 KP22 에서 정정했고 SBeA16 도 같은 기준으로 배선했다.
- SBeA16 관절 순서는 저자 배포 CSV 헤더 순서를 그대로 따른다(바꾸면 DLC 출력과 어긋난다).
- **SBeA16 좌우 대칭쌍 5개**(`sbea_lr_resolve.py` — `SYM_PAIRS`): 귀·앞다리·뒷다리·앞발톱·뒷발톱.
  저자 DLC 는 카메라마다 이 쌍의 좌/우를 다르게 붙이며, 그게 재투영 잔차의 지배 성분이다
  (260727, `scripts/diag_sbea_residual.py`). 완화는 카메라 간 *일관성*만 보장하고
  **어느 쪽이 해부학적 왼쪽인지는 검증되지 않았다** — 검출기 다수결을 승계한다.

## 3. 카메라 · 투영

| 대상 | 규약 | 정본 |
|---|---|---|
| **GS-LRM / BehaviorSplatter** | OpenCV (X-right, Y-down, Z-forward), World Z-up. `u = fx·X/Z + cx` | `BehaviorSplatter/docs/camera_conventions.md` |
| **DANNCE / s-DANNCE** | MATLAB — K 를 **전치하지 않고** cx·cy 가 3행. `M = [R;t] @ K`, `pts_h @ M` | `sdannce-poc/src/sdannce_utils/projection.py` · 스펙 `sdannce-poc/docs/theory/projection_convention.md` |
| **SBeA** | 3×4 P 행렬 직접 사용(분해 불필요). 왜곡 계수 **미배포** | `sdannce-poc/segmentation/kp_sam2_sources.py::_cal_sbea_p_matrix` |

- **M5 좌표 변환**(MAMMAL mm → GS-LRM 정규화): `kp_gslrm = (kp_mm − M5_SCENE_CENTER) × M5_DISTANCE_SCALE`.
  정본 `BehaviorSplatter/src/behaviorsplatter/notebooks/kp_utils.py` — **소수값 하드코딩 금지, import 할 것**
  (`0.008781` 오타가 렌더러 3곳에 퍼진 전례가 있다).
- DANNCE(MATLAB)와 BehaviorSplatter(OpenCV 전치) 두 파싱은 **수학적으로 등가**다. 각자 자기 툴체인과
  픽셀 단위로 맞아야 해서 통합하지 않는다 — 근거 `sdannce-poc/docs/repo_boundary.md §의도적 중복 1건`.
  한쪽 컨벤션을 바꾸면 다른 쪽 재투영이 깨지므로 **양쪽 동시 확인**.
- **SBeA camera↔P 대응은 파일명 순서가 아니다**(미문서화). `camera-{i}.mp4` ↔ `cam_mat_all[:,:,order[i]]`.
  `behavior-lab/scripts/sbea_dlc_triangulate.py` 가 세션마다 24순열을 재투영 오차로 자동 판정하고,
  결과를 `sdannce-poc/configs/segmentation/sbea.yaml` 의 `calibration.camera_order` 에 적어 생산자·소비자가
  같은 값을 쓰게 한다. rec10-M2/rec10-M7 실측: identity 130~163 px vs `[3,0,1,2]` 24~26 px.

## 4. 마스크 npz 키

- **쓰기**: `animal_0`, `animal_1`, … (`sdannce_utils.io.save_mask(..., version=2)`)
- **읽기**: 구형 `mask` / `ratN` 도 받아야 한다 — `sdannce_utils.io.load_mask` 와
  `BehaviorSplatter/scripts/preprocess_run.py::_load_mask_npz` 둘 다 세 형식을 처리한다.

## 5. 자동 검증되는 값

아래 블록은 `behavior-lab/tests/test_conventions_doc.py` 가 소스에서 직접 읽어 대조한다.
값이 바뀌면 테스트가 실패하므로 이 문서는 코드보다 뒤처질 수 없다.

```yaml checked
kp22_names: 22
kp22_bones: 21
rat23_names: 23
rat23_edges: 22
sbea16_bodyparts: 16
sbea16_edges: 15
sbea16_sym_pairs: 5
m5_scene_center: [59.672, 51.517, 107.099]
m5_distance_scale: 0.008772357
tensor_format_api: ["sequence_to_graph", "graph_to_sequence"]
sbea_camera_order: [3, 0, 1, 2]
```

> 실행: `cd ~/dev/behavior-lab && pytest tests/test_conventions_doc.py -v`
> 형제 레포(BehaviorSplatter·sdannce-poc)가 없는 환경에서는 해당 항목만 skip 된다.

## 6. 알아둘 소프트스팟

| 항목 | 상태 |
|---|---|
| `BehaviorSplatter/scripts/preprocess_m5_subject_centered.py` L29-30 이 M5 상수를 import 하지 않고 재정의 | 현재 값·수식 동일해 무해. 한쪽만 고치면 갈라짐 |
| SBeA 재투영 잔차 ~25 px | SAM2 프롬프트엔 충분, 정량 3D 분석엔 부족. 유력 원인은 왜곡 미모델링 |
| SBeA 저자 DLC 모델은 단일 개체용(`Mouse2Dproject`) | social 세션은 개체 분리 선행 필요 |
