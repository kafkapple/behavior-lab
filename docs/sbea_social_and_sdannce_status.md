# SBeA social · s-DANNCE — 상태와 착수 순서

> **260729 전면 개정.** 초판(260727)은 social 개체 분리를 "우리가 설계할 것"으로 적었다.
> 틀렸다. 저자가 이미 발표·배포한 기능이다. 아래는 그 정정과 그에 따른 착수 순서다.

## tldr

1. **SBeA 저자 파이프라인을 먼저 돌려라.** 우리 데이터(`*-caliParas.mat` + `*-camera-#`)가
   저자 코드가 기대하는 입력 형식 그대로다. 그런데 **코드를 클론한 적이 없다.**
2. 우리가 짠 것(DLT 삼각측량 + 카메라 순서 24순열 탐색 + 좌/우 재라벨)은 **저자 파이프라인의
   일부를 손으로 다시 만든 것**이다. 저자는 epipolar 제약으로 뷰 간 조합을 재투영 오차
   최소화로 이미 푼다.
3. social 개체 분리 = 저자 tracker 의 5단계 중 5단계. **설계 대상이 아니라 실행 대상.**
4. "정량 3D 는 별개 프로젝트 규모" 도 틀렸다. SBeA 는 **few-shot 설계**(윤곽 라벨 ~400-800
   프레임)이고 identity 는 label-free 다.

## 1. 사실 정정 (260729 검증)

출처: Han et al., *Nature Machine Intelligence* 6:48-61 (2024),
`10.1038/s42256-023-00776-5` · 코드 `github.com/YNCris/SBeA_release` ·
`README_SBeA_tracker.md` (직접 열람). 3-verifier 교차검증 3/3 일치.

| 초판(260727) 서술 | 실제 |
|---|---|
| social 분리를 SAM2 2-instance 로 우리가 설계 | 저자가 **VisTR 기반 video instance segmentation** 으로 구현·배포 |
| 카메라 간 identity 매칭을 재투영 일관성으로 우리가 설계 | 저자가 **epipolar 제약 + 재투영 오차 최소화**로 이미 수행 |
| identity 는 미해결 | **label-free / zero-shot, bidirectional transfer learning, >90%** (EfficientNet) |
| 정량 3D 하려면 새 모델 학습 = 별개 프로젝트 | **few-shot 프레임워크** — 윤곽 라벨 ~400-800 프레임 |
| 왜곡 계수 미배포 = 왜곡 미모델링 | 저자는 **체커보드 Zhang 방식**(MouseVenue3D)으로 캘리브. 왜곡은 캘리브 단계에서 처리됐고 배포된 `cam_mat_all` 이 그 결과물 |

**우리가 가진 것과 없는 것**: 배포 자산(영상 200 + `caliParas.mat` 50 + csv 8 + DLC ckpt 3)은
받았으나 **파이프라인 코드는 클론한 적이 없다**. 그래서 손으로 다시 만들었다.
동봉된 DLC 스냅샷은 저자 3모델(분할·포즈·ID) 중 **포즈 1개**일 뿐이다.

## 2. 저자 tracker 파이프라인 (5단계)

`README_SBeA_tracker.md` 인용 요지:

1. `configfile.yaml` 생성/로드
2. 경로 설정 — `*-caliParas.mat`, `*-camera-#.avi`
   ← 우리 데이터는 `rec10-M2-20221108-caliParas.mat` · `...-camera-0.mp4`.
   **stem 은 일치, 컨테이너만 avi↔mp4 차이.** 코드가 확장자를 하드코딩했는지 S1 에서 확인할 것
   (트랜스코딩이면 비용 무시 가능, 리더가 다르면 손볼 지점)
3. 로드·라벨·학습 — 3모델 동시: VisTR 분할 · DeepLabCut 포즈 · EfficientNet ID
   (학습데이터 생성은 YOLACT++ 기반)
4. 평가 (선택)
5. **"Predict 3D poses with identities of new videos"** → `.mat` 로 "3D skeletons rotated to
   ground with identities" + JSON 분할결과 + CSV identity

우리 파이프라인에 없는 것: 분할·identity·지면 정렬(ground rotation). 있는 것: DLT 삼각측량뿐.

## 3. 우리 25 px 잔차와의 관계 — 미해결

전 20세션 재투영 잔차 중앙값 24.7 px 이 **저자 기준으로 정상인지 비정상인지 모른다.**
논문이 보고한 절대 수치(px 또는 mm)를 확보하지 못했다 — Nature 페이월, bioRxiv/ResearchSquare
403. Fig. 6j 에 종별 재투영 오차 비교가 있는 것으로 보인다.

다만 저자의 정성적 서술 하나는 우리 관측과 정합한다: **"재투영 오차는 카메라 커버리지가
불완전할수록 커진다"**(개 데이터에서 유의하게 높았음). 우리가 260728 에 본 "이미지 주변부일수록
잔차 증가"(내→외 5분위 23.4→28.8 px)와 같은 방향이다 — 주변부 = 아레나 가장자리 =
커버리지 나쁜 영역.

**따라서 R1 의 잔여 질문은 "왜곡인가"가 아니라 "우리 24.7 px 이 저자 파이프라인에서도
나오는가"로 바뀐다.** 이건 저자 코드를 돌려야만 답할 수 있다.

## 4. 착수 순서 (비용 오름차순, 각 단계가 다음 단계 필요성을 판정)

### S1. 저자 코드 클론 + Fig. 6j 수치 확보 — 수시간

```
git clone https://github.com/YNCris/SBeA_release
```
동시에 논문 PDF 확보(기관 접근 또는 저자 요청)해 **보고된 재투영 오차 절대값**을 읽는다.

판정: 저자 보고치가 우리 24.7 px 와 같은 자릿수 → 우리 파이프라인 정상, 잔차는 데이터 상한.
크게 낮음 → 우리 재구성이 뭔가 놓치고 있다는 뜻이고 S2 가 그걸 찾는다.

**이 단계 전에는 좌/우 완화든 왜곡이든 더 파지 말 것.** 기준선을 모르는 채 최적화 중이다.

### S2. 저자 pretrained 로 individual 세션 1개 재현 — 1일

저자 3모델 중 배포된 건 DLC 스냅샷뿐이므로, 분할·ID 모델은 학습이 필요할 수 있다.
그 전에 **저자 3D 재구성 코드만이라도 우리 2D 에 물려** 우리 DLT 대비 잔차를 비교한다.

판정 대상: 카메라 순서(우리는 24순열 탐색으로 (3,0,1,2) 를 찍었다 — 저자 코드는 규약을
알고 있을 것), 지면 정렬, epipolar 최적화 유무.

### S3. social 세션 — 저자 경로로 실행 — 2-3일

S1-S2 가 통과하면 5단계 predict 를 social 30세션에 돌린다. 필요 라벨은 few-shot 규모
(윤곽 ~400-800 프레임). identity 는 label-free 라 라벨 불요.

**우리가 설계했던 SAM2 경로는 폐기한다** — 저자 VisTR 분할 + EfficientNet ID 가 이미 있고
그쪽이 논문으로 검증돼 있다.

### S4. 그래도 부족하면 — finetune, 그다음에야 신규 학습

- **같은 단일개체 아키텍처 finetune 은 좌/우 문제를 못 고친다.** 원인이 데이터량이 아니라
  구조다 — part heatmap 이 독립이라 스켈레톤 제약이 없고, 단일 시점에서 자기폐색 시
  좌/우는 정보이론적으로 미결정이다. 데이터를 더 줘도 이미지에 없는 정보는 안 생긴다.
- **구조를 바꾸는 게 답**: multi-animal DLC 의 **PAF**(part affinity field)는 사지 연결
  방향을 예측해 조립 단계에서 좌/우·개체 혼동을 줄인다. 또는 SBeA 처럼 분할 윤곽으로
  몸통 방향을 먼저 잡는다.
- 신규 대규모 학습은 **최후 수단**이고, S1-S3 를 건너뛰고 여기로 가면 이미 있는 걸 또 만든다.

## 5. s-DANNCE 자체 예측 (구 R6) — 변경 없음

`sdannce-poc/shell/run_poc.sh` 는 동작하나 선행 자산 미다운로드로 실행 불가.
현재 쓰는 KP 는 저자 배포 `.mat` 이고 다운스트림엔 충분하다. 배포 `.mat` 이 없는 세션을
다뤄야 할 때까지 다운로드는 비용만 든다.

| 필요 자산 | 경로 | 출처 |
|---|---|---|
| 사전학습 가중치 | `sdannce/demo/markerless_mouse_1/DANNCE/weights/dannce-c-r7m.pth` | `duke.box.com/shared/static/71wxs2jqmacqy66zvfbjywh0lj6fnjxj.pth` |
| 데모 영상 | `sdannce/demo/markerless_mouse_1/videos/Camera1/` | `tinyurl.com/DANNCEmm1vids` |

## 6. 방법론 경고 (S1-S4 전 구간 적용)

260728 에 실제로 당한 것이라 적어둔다. **재투영 오차는 최적화 대상이라 자기 결과를 검증할 수
없다.** 200프레임 2세션에서 좌/우 완화가 본 길이 −31%/−16% 로 좋아 보였으나, lam 을 고른 바로
그 세션에서 잰 값이었고 20세션 전량에서는 −2%·6세션 악화였다.

S2 의 카메라 순서·S3 의 개체 매칭 모두 **재투영 최소화로 이산 배정을 푸는 같은 구조**다.
같은 함정이 있다. 반드시 **독립 지표**(본 길이 강성·마스크 IoU 연속성 등)를 따로 두고,
자유 파라미터는 **선택에 쓰지 않은 세션**에서 검증할 것.
