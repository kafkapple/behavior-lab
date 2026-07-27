# gpu03 Dataset Inventory

> 온디스크 데이터셋 실사 목록. 스토리지 정책 = `STORAGE_GUIDE.md`, 스켈레톤 스키마(관절명·클래스명) = `behavior-lab/docs/datasets.md`.
> 실사일: **2026-07-27** (ssh gpu03, `du`/`ls`/numpy 헤더 직접 확인). 모든 수치는 그날 관측값.

---

## TL;DR

- `~/dev/datasets` 는 **데이터셋 폴더가 아님** — 심볼릭 링크 1개짜리 스텁. 실제 데이터는 `/node_data/joon/data`(~33G) + `/node_data_2/joon/data`(~297G).
- 총 5개 도메인 32개 데이터셋 — 행동/스켈레톤 11 · 3D 재구성 12 · VLM/AV 벤치마크 4 · 로보틱스 5 · 신경/오디오 1.
- 즉시 조치 필요 3건: 빈 벤치마크 2개(omnibench·omnivideobench) · 미해제 zip 1개(cnn_subspace_auditory) · `/node_data` 94% 포화.

---

## 1. 위치 맵

| 경로 | 실체 | 용량 | 티어 |
|---|---|---:|---|
| `~/dev/datasets/` | **스텁** — `cnn_subspace_auditory` 심링크 1개뿐 | 4K | — |
| `~/data` | `/node_data/joon/data` 영구 심링크 | — | — |
| `/node_data/joon/data` | 활성 학습 데이터 (NVMe) | ~33G | HOT |
| `/node_data_2/joon/data` | 원본·아카이브 (SATA HDD) | ~297G | COLD |

디스크 여유: `/` 409G(51%) · `/node_data` 455G(**94%**) · `/node_data_2` 657G(91%).

---

## 2. 행동/스켈레톤 데이터셋 (behavior-lab)

| 데이터셋 | 경로 (root 기준) | 용량 | 포맷·구조 |
|---|---|---:|---|
| CalMS21 원본 | `HOT:calms21_dl/task1_classic_classification/` | 2.7G | JSON ×4 — train 1.2G / test 632M / taskprog feature train 685M·test 354M |
| CalMS21 BehaveMAE용 | `HOT:calms21_bmae/calms21_train.npy` | 48M | object-npy `{sequences: 246개 시퀀스}` |
| MABe22 | `HOT:mabe22_dl/user_train.npy` | 814M | object-npy `{vocabulary, sequences}` |
| NTU RGB+D 60 | `HOT:ntu_dl/ntu60_3danno.pkl` | 1.2G | pkl `{split: 6분할(xsub/xset/xview × train/val), annotations: 56,578개}`, 항목키 `frame_dir·label·keypoint·total_frames` |
| NTU BehaveMAE 서브셋 | `HOT:ntu_bmae/ntu_train.npy` | 15M | object-npy `{sequences: 500개}` |
| SHOT7M2 | `HOT:shot7m2/test/` | 3.4G | npy ×3 — poses `{sequences:{keypoints}}` · benchmark_labels·actions `{frame_number_map: 2,720 에피소드, label_array, vocabulary, task_type}` |
| s-DANNCE WK1 (SCN2A M1) | `HOT:sdannce/wk1/SCN2A_WK1_2022_09_16_M1/` | 1.1G | `videos/ COM/ SDANNCE/ calibration/` + `meta.json` 624K + 다운로드 스크립트 |
| SBeA | `COLD:external/SBeA/` | 29G | `individual/ social/ sbea_release_assets/` |
| PAIR-R24M | `COLD:external/PAIR-R24M/` | 65G | `PAIR-R24M-Dataset/` + 카메라 파싱 스크립트·`cam_pose_summary.json` |
| markerless_mouse_1 | `COLD:external/markerless_mouse_1/` | 1.2M | `labels/` 만 (영상 없음) |
| markerless_mouse_2 | `COLD:external/markerless_mouse_2/` | 11G | `videos_raw/ videos_concat/ labels/ samples/` |

> `HOT:` = `/node_data/joon/data/`, `COLD:` = `/node_data_2/joon/data/`.

**정정 사항**: `behavior-lab_stale_260705/docs/260705_gpu03_experiment_setup.md` 는 "SBeA 미보유"로 기록 — 260727 현재 29G 존재. 해당 문서는 stale.

---

## 3. 3D 재구성 / 멀티뷰 (FaceLift · BehaviorSplatter · PoseSplatter · MonoFusion)

| 데이터셋 | 경로 | 용량 | 구조 |
|---|---|---:|---|
| FaceLift_mouse_6view | `COLD:shared/FaceLift_mouse_6view/` | **117G** | `checkpoint/ configs/ gaussians/ m5_data/ metadata/` + README |
| d7_experiments | `COLD:shared/d7_experiments/` | 37G | `ve_d8v2_clean_20260502_1654/` + README |
| preprocessed (COLD) | `COLD:preprocessed/` | 31G | `Mouse_v6_fx549 · Mouse_v7_fx{549,1000,1500} · Mouse_m5_v2_sc · M5_M2_union · FaceLift_mouse · markerless_mouse_2` |
| FaceLift_mouse (HOT) | `HOT:preprocessed/FaceLift_mouse/` | 9.1G | 학습 활성 사본 |
| deform | `COLD:deform/rat7m_v10a_w50_3060/` | 2.9G | deform 모듈 실험 데이터 |
| _final_paper_260521 | `HOT:shared/_final_paper_260521/` | 2.6G | `FaceLift_code`·`PoseSplatter_mouse_6view_v2` 가 여기로 심링크 |
| bs_temporal_approved_260521 | `COLD:bs_temporal_approved_260521/` | 1.2G | center_fixed 계열 23개 런 + 로그 |
| raw | `HOT:raw/` | 952M | `dannce` 311M · `markerless_mouse_1_nerf` 300M · `samples` 227M · `tets` 115M · `vlm-poc` 648K |
| ps_m5_baseline_260724_full | `HOT:ps_m5_baseline_260724_full/fj1/` | 356M | `images.zarr` + `camera_params.h5` + `center_rotation.npz` + `vertical_lines.npz` |
| synthetic | `HOT:synthetic/` | 230M | `textured_obj` · `mouse_mesh` · `_rectangle_position_test` + render 로그 |
| WK1/SOC1r 전처리 | `HOT:preprocessed/{WK1_dense,WK1_fx549,WK1_fx1000,WK1_fx1500_sample200fr,SOC1r1_dense,SOC1r2_dense}` | 18M~84M | 프레임별 `{6자리ID}/images/ + opencv_cameras.json` |
| Rat7M_dense | `HOT:preprocessed/Rat7M_dense/` | 91M | 동일 프레임 디렉토리 구조 |
| ps_m5_baseline_260724 | `HOT:ps_m5_baseline_260724/fj1/` | 1.3M | full 버전의 축소 샘플 |

---

## 4. VLM / 오디오-비디오 벤치마크 (avllm-temporal · qwen-multimodal)

| 벤치마크 | 경로 | 용량 | 상태 |
|---|---|---:|---|
| AVHBench | `HOT:benchmarks/avhbench/` | 4.4G | ✅ 사용 가능 — `qa.json` 6,408 항목(`video_id·task·text·label`) + `videos/` |
| Video-MME | `HOT:benchmarks/video-mme/` | 1.6M | ⚠️ 메타데이터만 — `metadata.json` 2,700 항목(`video_id·duration·domain·sub_category·url`), 영상은 `hf_cache` 미다운로드 |
| OmniBench | `HOT:benchmarks/omnibench/` | 8K | ❌ 빈 `hf_cache` 스텁 |
| OmniVideoBench | `HOT:benchmarks/omnivideobench/` | 4K | ❌ 빈 디렉토리 (파일 0개) |

---

## 5. 로보틱스 (LIBERO-PRO)

| 데이터셋 | 경로 | 용량 |
|---|---|---:|
| LIBERO 원본 | `HOT:libero_pro/libero_datasets/` | 6.0G |
| lpwm_frames_expert_goal_mini | `HOT:libero_pro/` | 165M |
| lpwm_frames_action_mini | `HOT:libero_pro/` | 24M |
| lpwm_frames_mini | `HOT:libero_pro/` | 24M |
| lpwm_frames_smoke | `HOT:libero_pro/` | 96K |
| lpwm markerless_mouse_1 파생 | `HOT:mouse/{_mini,_smoke12}` | 69M |

---

## 6. 신경/오디오

| 데이터셋 | 경로 | 용량 | 상태 |
|---|---|---:|---|
| cnn_subspace_auditory | `COLD:cnn_subspace_auditory/recordings.zip` | 4.3G | ⚠️ **미해제 zip 1개** — `~/dev/datasets/` 의 유일한 심링크 대상 |

---

## 7. 갭 · 리스크

| 항목 | 관찰 | 위협 모델 | 판단 |
|---|---|---|---|
| `/node_data` 94% (455G 여유) | HOT 티어 포화 임박 | 학습 중 checkpoint/outputs 쓰기 실패 → 런 중단. FaceLift 6view 급 산출물 1~2회분 여유 | 🟡 모니터 — COLD 이관 후보 = `HOT:preprocessed/FaceLift_mouse` 9.1G(COLD에 동일 이름 존재, 중복 여부 확인 필요) |
| omnibench·omnivideobench | 파일 0~8K | 벤치마크 스크립트가 빈 디렉토리를 "다운로드됨"으로 오인 가능 | 🟡 다운로드 미완 — 사용 전 재확인 필요 |
| video-mme | 메타데이터만 | 위와 동일 | 🟡 |
| cnn_subspace_auditory | zip 미해제 | 사용 시점에 4.3G 해제 필요, HOT 여유와 무관(COLD 소재) | 🟢 |
| `HOT:derived/`, `HOT:preprocessed/markerless_mouse_2`, `HOT:preprocessed/WK1_v2_sc_fx1000` | 0 바이트 빈 디렉토리 | 없음 | 🟢 정리 대상 |
| `~/dev/datasets` 스텁 | 이름이 데이터셋 허브처럼 보이나 실제로는 링크 1개 | 신규 세션·협업자가 "데이터셋 여기 있음"으로 오독 | 🟡 이 문서로 포인터 정정 |

**미확인 (실사 범위 밖)**: `HOT:preprocessed/FaceLift_mouse`(9.1G) 와 `COLD:preprocessed/FaceLift_mouse` 의 내용 동일성 — 중복 여부는 체크섬 비교 필요, 이번 실사에서 미수행.
