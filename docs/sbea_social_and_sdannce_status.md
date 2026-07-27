# SBeA social 세션 · s-DANNCE 자체 예측 — 설계와 선행조건

> 260727. 두 건 모두 **미착수 상태로 종결**하고 착수 조건만 확정한 문서다.
> 실행 결정이 필요할 때 여기서 시작한다. 진행 상황 정본은 git log.

## 1. SBeA social 세션 (구 핸드오프 R2)

### 왜 지금 안 되는가

저자 배포 모델은 `Mouse2Dproject` — **프레임당 1포즈**를 내는 단일 개체 DLC 다.
social 세션(30개, `social/rec*-M?M?-*`)은 한 화면에 마우스 2마리이므로 모델이
둘 중 하나에만 반응하거나 둘을 섞는다. 배포본에 identity 분리 모델은 없다
(전수 census: mp4 200 + caliParas.mat 50 + csv 8 + TF ckpt 3, 그 외 없음).

### 착수 시 설계

이미 있는 부품만으로 구성된다 — 신규 학습 없음.

| 단계 | 방법 | 재사용할 것 |
|---|---|---|
| 1. 카메라별 개체 분리 | SAM2 를 2-instance 로 프롬프트해 마우스별 마스크 트랙 생성 | `sdannce-poc/segmentation/kp_sam2.py` |
| 2. 카메라 간 identity 매칭 | 트랙 A·B 를 4대에서 어떻게 짝지을지 = 16가지 배정. 각 배정으로 삼각측량 후 **재투영 잔차 최소** 배정 선택 | `sbea_lr_resolve.py` 의 Viterbi 가 구조적으로 동일 |
| 3. 개체별 2D 포즈 | 마스크 bbox 로 crop → 저자 DLC 를 crop 마다 실행 → 좌표를 원본 프레임으로 역변환 | `sbea_dlc_triangulate.py::infer_camera` |
| 4. 삼각측량 | 개체별로 기존 경로 그대로 | `solve_all` |

**2번이 이 설계의 핵심이자 검증된 부분**이다. 좌/우 관절 문제와 수학적으로 같은
문제(카메라별 이산 배정을 다중뷰 일관성으로 푸는 것)이고, 거기서 이미 확인된
교훈이 그대로 적용된다:

- 프레임 독립으로 풀면 **flicker 한다**. 시간 평활(Viterbi switch penalty) 필수.
- 재투영 잔차는 목적함수라 자기 결과를 검증 못 한다. **독립 지표를 따로 둘 것** —
  개체 분리에서는 마스크 IoU 연속성과 개체별 몸길이 안정성이 후보.

### 착수 조건

- 2마리가 **접촉·교차하는 구간**에서 SAM2 트랙이 유지되는지 먼저 확인. 여기서
  깨지면 1단계가 무너져 2~4단계가 무의미하다. 이게 실질적 gate.
- 잔차 기대치는 individual 과 같은 ~25px 수준. 즉 **정량 3D 사회행동 분석은
  이 경로로 안 된다.** SAM2 프롬프트·개체 추적 용도까지가 현실적 범위(§3).

## 2. s-DANNCE 자체 예측 파이프라인 (구 핸드오프 R6)

`sdannce-poc/shell/run_poc.sh` 는 작성돼 있고 260304 에 동작 확인됐으나
**현재 로컬에 선행 자산이 없어 실행 불가**하다. 스크립트가 자체적으로 검사하고
안내 메시지와 함께 종료한다.

| 필요 자산 | 경로 | 출처 |
|---|---|---|
| 사전학습 가중치 | `sdannce/demo/markerless_mouse_1/DANNCE/weights/dannce-c-r7m.pth` | `https://duke.box.com/shared/static/71wxs2jqmacqy66zvfbjywh0lj6fnjxj.pth` |
| 데모 영상 | `sdannce/demo/markerless_mouse_1/videos/Camera1/` | `https://tinyurl.com/DANNCEmm1vids` |

현재 쓰는 s-DANNCE 키포인트는 **저자 배포 `.mat`** 이고, 다운스트림은 그걸로
충분하다. 자체 예측이 필요해지는 시점은 배포 `.mat` 이 없는 세션을 다뤄야 할
때뿐이다. 그 전까지 다운로드는 비용만 든다.

## 3. 두 건에 공통으로 걸리는 상한

SBeA 3D 는 **SAM2 프롬프트용**으로 검증됐고 그 용도로 쓴다. 정량 3D kinematics
(사지 각도·보행 주기 등)는 이 데이터로 지원되지 않는다 — 재투영 잔차 ~25px 의
지배적 원인이 저자 모델의 좌/우 라벨 불일치이고, 부분 완화 후에도 중앙선 관절의
~19-20px 바닥까지 내려가지 않는다. 근거 = `scripts/diag_sbea_residual.py`,
한계 서술 = `scripts/sbea_lr_resolve.py` docstring.

정량 분석이 필요해지면 저자 모델 재사용이 아니라 **다개체 대응 모델을 새로 학습**
하는 쪽이 정공법이다. 그건 별개 프로젝트 규모다.
