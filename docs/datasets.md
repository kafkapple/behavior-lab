# Dataset Catalog

> Comprehensive catalog of all datasets supported by behavior-lab.

---

## Currently Implemented (E2E Verified)

### CalMS21 — Mouse Social Behavior (2D)

| Field | Value |
|-------|-------|
| Species | Mouse (Mus musculus) |
| Subjects | 2 mice per scene |
| Joints | 7 per mouse (14 total) |
| Dimensions | 2D (top-view) |
| Classes | 4: other, attack, investigation, mount |
| Format | HDF5 (.h5) |
| Sequences | ~19K train / ~4.8K test |
| FPS | 30 Hz |
| Skeleton Name | `calms21`, `calms21_mouse` |
| Reference | Sun et al. (2021), CalMS21 NeurIPS |
| Download | [CalMS21 Challenge](https://data.caltech.edu/records/s0vdx-0k302) |

#### Class Names

| ID | Name | Description |
|----|------|-------------|
| 0 | Other | No specific behavior |
| 1 | Attack | Aggressive behavior |
| 2 | Investigation | Social investigation |
| 3 | Mount | Mounting behavior |

#### Joint Names

| # | Name | Abbreviation | Body Part |
|---|------|-------------|-----------|
| 0 | nose | Nos | head |
| 1 | left_ear | LEa | head |
| 2 | right_ear | REa | head |
| 3 | neck | Nec | head |
| 4 | left_hip | LHi | body |
| 5 | right_hip | RHi | body |
| 6 | tail_base | TBa | tail |

#### Analysis Model
- **B-SOiD** (Unsupervised Discovery): UMAP + HDBSCAN + Random Forest

---

### NTU RGB+D 60 — Human Action (3D)

| Field | Value |
|-------|-------|
| Species | Human |
| Subjects | 1-2 persons per scene |
| Joints | 25 per person (50 max) |
| Dimensions | 3D (Y-up, Kinect v2 depth sensor) |
| Classes | 60 action categories |
| Format | NPZ (pre-aligned) |
| Sequences | 500 demo subset |
| FPS | 30 Hz |
| Skeleton Name | `ntu`, `ntu_rgbd`, `ntu60` |
| Reference | Shahroudy et al. (2016), CVPR |
| Download | [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) |

#### Class Names (60 categories)

| ID | Name | ID | Name | ID | Name |
|----|------|----|------|----|------|
| 0 | drink water | 20 | put on a hat/cap | 40 | kicking something |
| 1 | eat meal | 21 | take off a hat/cap | 41 | reach into pocket |
| 2 | brush teeth | 22 | cheer up | 42 | hopping |
| 3 | brush hair | 23 | hand waving | 43 | jump up |
| 4 | drop | 24 | kicking something | 44 | phone call |
| 5 | pick up | 25 | put/take sth. | 45 | play with phone |
| 6 | throw | 26 | point to sth. | 46 | type on keyboard |
| 7 | sit down | 27 | taking a selfie | 47 | point to sth. |
| 8 | stand up | 28 | check time | 48 | taking a selfie |
| 9 | clapping | 29 | rub two hands | 49 | check time |

*(Full list in `behavior_lab/data/loaders/ntu_rgbd.py:NTU60_CLASSES`)*

#### Joint Names (25 joints)

| # | Name | Body Part |
|---|------|-----------|
| 0 | base_spine | torso |
| 1 | mid_spine | torso |
| 2 | neck | torso/head |
| 3 | head | head |
| 4-7 | left_shoulder/elbow/wrist/hand | left_arm |
| 8-11 | right_shoulder/elbow/wrist/hand | right_arm |
| 12-15 | left_hip/knee/ankle/foot | left_leg |
| 16-19 | right_hip/knee/ankle/foot | right_leg |
| 20 | spine (shoulder center) | torso |
| 21 | left_hand_tip (HandTipLeft) | left_arm |
| 22 | left_thumb (ThumbLeft) | left_arm |
| 23 | right_hand_tip (HandTipRight) | right_arm |
| 24 | right_thumb (ThumbRight) | right_arm |

#### Analysis Model
- **Linear Probe** (Supervised Baseline): Mean-pooled keypoints → LogisticRegression

---

### NW-UCLA — Human Action (3D)

| Field | Value |
|-------|-------|
| Species | Human |
| Subjects | 1 person |
| Joints | 20 (Kinect v1) |
| Dimensions | 3D (Y-up, Kinect v1) |
| Classes | 10 action categories |
| Format | NPY |
| Sequences | ~1.0K train / ~464 test |
| FPS | 30 Hz |
| Skeleton Name | `nwucla`, `nw_ucla`, `ucla` |
| Reference | Wang et al. (2014), CVPR |

#### Class Names

| ID | Name |
|----|------|
| 0 | pick up with one hand |
| 1 | pick up with two hands |
| 2 | drop trash |
| 3 | walk around |
| 4 | sit down |
| 5 | stand up |
| 6 | donning |
| 7 | doffing |
| 8 | throw |
| 9 | carry |

#### Analysis Model
- **Linear Probe** + **LSTM/Transformer** (2-epoch quick test)

---

## Newly Added (Skeleton Defined, Loader Implemented)

### SUBTLE — Mouse Spontaneous Behavior (3D)

| Field | Value |
|-------|-------|
| Species | Mouse |
| Joints | 9 |
| Dimensions | 3D (Z-up) |
| Classes | 4: walking, grooming, rearing, standing |
| Format | CSV (27 cols = 9 joints × 3D) |
| FPS | 20 Hz |
| Skeleton Name | `subtle`, `subtle_mouse` |
| Reference | Kwon et al. (2022), bioRxiv |
| Source | [GitHub](https://github.com/jeakwon/subtle) |

#### Joint Names

Column order from `kinematics.py` `avatar_configs['nodes']`:

| # | Name (code) | Original (kinematics.py) | Body Part |
|---|-------------|--------------------------|-----------|
| 0 | nose | nose | head |
| 1 | neck | neck | head |
| 2 | tail_base | anus | body |
| 3 | mid_back | chest | body |
| 4 | right_hindpaw | rfoot | right_back_leg |
| 5 | left_hindpaw | lfoot | left_back_leg |
| 6 | right_forepaw | rhand | right_front_leg |
| 7 | left_forepaw | lhand | left_front_leg |
| 8 | tail_tip | tip | tail |

#### Edge Topology (from `kinematics.py` `avatar_configs['edges']`)

```
nose→neck→mid_back(chest)→tail_base(anus)→tail_tip
                   ├─left_forepaw
                   └─right_forepaw
          tail_base├─left_hindpaw
                   └─right_hindpaw
```

---

### MABe22 — Mouse Triplet Social Behavior (2D)

| Field | Value |
|-------|-------|
| Species | Mouse |
| Subjects | 3 mice per scene (triplet) |
| Joints | 12 per mouse (36 total) |
| Dimensions | 2D (top-view) |
| Data Shape | (N, T, 36, 2) where 36 = 3 mice × 12 joints |
| Format | NPY |
| FPS | 30 Hz |
| Skeleton Name | `mabe22`, `mabe22_mouse`, `mabe` |
| Reference | Sun et al. (2023), ICML Vol. 202, pp. 32936-32990 |
| Download | [Caltech Data](https://data.caltech.edu/records/s0vdx-0k302) |

#### Joint Names (12 per mouse)

| # | Name | Body Part |
|---|------|-----------|
| 0 | nose | head |
| 1 | left_ear | head |
| 2 | right_ear | head |
| 3 | neck | head |
| 4 | left_forepaw | front_paws |
| 5 | right_forepaw | front_paws |
| 6 | center_back | body |
| 7 | left_hindpaw | hind_paws |
| 8 | right_hindpaw | hind_paws |
| 9 | tail_base | tail |
| 10 | tail_middle | tail |
| 11 | tail_tip | tail |

#### Analysis Model
- **BehaveMAE** (Hierarchical MAE): 3-level moveme/action/activity discovery

---

### Rat7M — Rat Motion Capture (3D)

| Field | Value |
|-------|-------|
| Species | Rat (Rattus norvegicus) |
| Joints | 20 |
| Dimensions | 3D (Z-up) |
| Classes | Unlabeled (discovery target) |
| Format | MAT / HDF5 / NPY |
| FPS | 120 Hz |
| Skeleton Name | `rat7m` |
| Reference | Dunn et al. (2021), Nature Methods |
| Source | Figshare collection (manual download, several GB) |

#### Joint Names

| # | Name | Body Part |
|---|------|-----------|
| 0-3 | nose_tip, head_top, left_ear, right_ear | head |
| 4 | neck | torso |
| 5-6 | left/right_shoulder | torso |
| 7-10 | left/right_elbow/wrist | arms |
| 11 | spine_mid | torso |
| 12-17 | left/right_hip/knee/ankle | legs |
| 18-19 | tail_base, tail_mid | tail |

---

### Shank3KO — Mouse 3D Behavior (3D)

| Field | Value |
|-------|-------|
| Species | Mouse (Shank3 knockout) |
| Joints | 16 |
| Dimensions | 3D (Z-up) |
| Classes | 11 movement types |
| Format | MAT / NPY |
| FPS | 30 Hz |
| Skeleton Name | `shank3ko`, `shank3ko_mouse` |
| Reference | Huang et al. (2021), Nature Communications |
| Source | [Zenodo](https://doi.org/10.5281/zenodo.4629544) |

#### Class Names (11 movement types)

| ID | Name |
|----|------|
| 0 | running |
| 1 | trotting |
| 2 | stepping |
| 3 | diving |
| 4 | sniffing |
| 5 | rising |
| 6 | right_turning |
| 7 | up_stretching |
| 8 | falling |
| 9 | left_turning |
| 10 | walking |

#### Joint Names (from raw .mat `Body_name` field)

| # | Name | Body Part |
|---|------|-----------|
| 0 | nose | head |
| 1 | left_ear | head |
| 2 | right_ear | head |
| 3 | neck | head |
| 4-5 | left/right_front_limb | front_leg |
| 6-7 | left/right_hind_limb | back_leg |
| 8-9 | left/right_front_claw | front_leg |
| 10-11 | left/right_hind_claw | back_leg |
| 12 | back | body |
| 13 | root_tail | tail |
| 14 | mid_tail | tail |
| 15 | tip_tail | tail |

**Note**: Raw Zenodo data is MATLAB (.mat). `Fs=30` Hz confirmed from raw data field.

---

## Coordinate Systems & Preprocessing

### Coordinate Conventions

| Dataset | Up Axis | System | Unit |
|---------|---------|--------|------|
| NTU RGB+D | Y-up | Kinect v2 (depth sensor) | meters |
| NW-UCLA | Y-up | Kinect v1 (depth sensor) | meters |
| CalMS21 | N/A (2D) | DLC top-view | pixels |
| MABe22 | N/A (2D) | DLC top-view | pixels |
| SUBTLE | Z-up | 3D motion capture | mm |
| Shank3KO | Z-up | 3D SLEAP tracking | mm |
| Rat7M | Z-up | Marker-based mocap | mm |

Visualization converts all 3D data to matplotlib's Z-up convention
(`_to_viz_coords` in `visualization/skeleton.py`).

### Outlier Clipping Strategy

Raw keypoint tracking data often contains outliers from:
- Occlusion-related jumps (e.g., Shank3KO `tip_tail` spikes to Z=-2476mm)
- Marker swap events
- Detection failures

**Two modes** (`clip_outlier_joints` in `visualization/skeleton.py`):

| Mode | Method | When to Use |
|------|--------|-------------|
| **Global percentile** | Clip all coords to [p, 100-p] range | 2D data, uniform distributions |
| **Per-joint IQR** | Tukey's fences per joint (Q1-k×IQR, Q3+k×IQR) | 3D data with per-joint tracking errors |

**Literature basis**:
- **Tukey (1977)**: k=1.5 for "outlier", k=3.0 for "far out" fences
- **DeepLabCut** (Mathis et al., 2018): Median filtering + likelihood thresholding
- **SLEAP** (Pereira et al., 2022): Confidence-based NaN replacement
- **Our default**: k=3.0 (conservative, preserves genuine behavioral extremes)

**Configuration** (in `scripts/test_e2e.py` `VIZ_CONFIG`):
```python
VIZ_CONFIG = {
    "clip_per_joint": True,     # Per-joint IQR mode
    "clip_iqr_factor": 3.0,    # Tukey's "far out" fence
    "clip_percentile": 1.0,    # Global mode fallback
    "gif_n_frames": 240,       # 2x previous default (120)
    "per_class_n_frames": 240,  # Per-class/cluster GIF length
}
```

---

## 3축 동시 충족 매트릭스 (260817 신설)

> **질문**: "멀티뷰 영상 + 3D 키포인트 + 행동 레이블" 셋을 다 가진 오픈 데이터셋이 있는가?
> **답**: 있다. 다만 우리 로컬 자산 상태는 별개 문제다 — 아래 두 열을 구분해서 볼 것.
> 스펙 정본(카메라·해상도·레이블 열 포함) = `_html/260619_datasets_comparison.html`.

| 데이터셋 | 멀티뷰 | 3D kp | 행동 레이블 | 3축 | 로컬 자산 상태 |
|---|---|---|---|---|---|
| **s-DANNCE lone** (SCN2A_WK1 M1) | 6 cam | ✅ 23 kp | ✅ **9클래스**(원전 확인) | 🟢 **충족** | 🔴 **레이블 미취득** · 마스크 부재(`sam3_multimask_masks/wk1_*`) |
| **s-DANNCE social** (SCN2A_WK1) | 6 cam | ✅ 23 kp | ✅ 9클래스 + social 상호작용 | 🟢 **충족** | 🔴 미다운로드 · 마스크 null |
| **PAIR-R24M** (18 pairs) | 6 cam | ✅ 12 kp/rat | ✅ **11 behavioral + 3 interaction** | 🟢 **충족** | 🔴 학습 대기 |
| **NTU RGB+D 60** (인간) | **3 cam** | ✅ Kinect skeleton | ✅ 60 actions | 🟡 **뷰 수 부족** | — · 3뷰는 BS 재구성(통상 6뷰)에 희소 |
| markerless_mouse_1 (M5) | 6 cam | ✅ 22 kp | ❌ none | ❌ | 🟢 main |
| Rat7M | 6 cam | ✅ MoCap | ❌ *"Unlabeled (discovery target)"* | ❌ | 🟢 보유 |
| PoseSplatter mouse | 6 cam | ✅ 2D/3D bundled | ❌ none | ❌ | 🟢 보유 |
| **SBeA** individual·social | 4 cam | ❌ *needs MAMMAL fit* | ❌ **none (unsupervised)** | ❌ | 🟢 원본 보유 |
| CalMS21 | ❌ 2D top-view | ❌ 2D only | ✅ 4 classes | ❌ | 🟢 E2E 검증 |
| Shank3KO | 🟡 멀티뷰 영상 미확인(MAT/NPY 배포) | ✅ 16 kp | ✅ 11 movement | 🟡 | 🟡 미취득 |

**여기서 나오는 세 가지**

1. **3축을 다 만족하는 동물 데이터셋은 이미 3개 있다** — s-DANNCE(lone·social), PAIR-R24M.
   "그런 데이터가 없다" 는 전제는 틀렸다.
2. **병목은 데이터 부재가 아니라 취득이다.** s-DANNCE lone 은 🟢 main 이지만
   **레이블을 받은 적이 없다** — `DATASET_INDEX.yaml: sdannce_wk1` 에 레이블 경로가 없고,
   `SDANNCE_SOC1_v1.yaml` 의 `exclude_hlac_classes` 는 *"(있을 시)"* 주석이 달린 예정 옵션이다.
   마스크(`sam3_multimask_masks/wk1_*`)도 부재다. ⇒ **레이블 + 마스크 취득이 최우선.**
3. **SBeA 는 3축 중 2축이 비어 있다** — 3D 키포인트는 우리가 만든 것이고(배포본엔 없음),
   행동 레이블은 애초에 없다(unsupervised 프레임워크). 레이블 기반 작업의 후보가 아니다.

## Investigated (Not Yet Integrated)

### s-DANNCE (Klibaite et al., **Cell 2025** — 구 표기 "2024" 정정 260817)

| Field | Value |
|-------|-------|
| Species | **랫 + 마우스 둘 다** (WT rats and mice · ASD 랫 계통 포함) |
| Cameras | **6** (lone·social 양쪽) |
| Joints | 23 |
| **행동 레이블** | ✅ **전문가 주석 9클래스** — idle · sniff · groom · scrunch · active crouch · rearing · explore · locomotion · fast locomotion |
| Scale | 140M+ 3D poses · 80 animals |
| Source | Harvard Dataverse · 코드 `github.com/tqxli/sdannce` |
| Status | 🔴 **레이블 미취득** — 우리 로컬 WK1 항목(`DATASET_INDEX.yaml: sdannce_wk1`)에는 `raw_data`·`mocap_path`·`preprocessed` 만 있고 **레이블 파일 경로가 없다** |

> 🔴 **260817 정정** — 앞선 "s-DANNCE lone 은 레이블이 있는데 우리가 안 쓴다" 는 부정확했다.
> 공개 데이터셋에는 있으나 **우리가 받은 적이 없다**. "미사용" 이 아니라 "미취득" 이다.
> 원전: Klibaite, Li, Aldarondo, Akoad, Ölveczky, Dunn — *Mapping the landscape of social
> behavior*, Cell (2025). 6-camera lone and social recordings across 80 animals.

### PAIR-R24M (Marshall 2021)

| Field | Value |
|-------|-------|
| Species | Rat (pairs) |
| Markers | 12 per rat (24 total) |
| Categories | **11 behavioral + 3 interaction** (🔴 260817 정정: 구 '84 fine-grained' 는 오기) |
| Viewpoints | 24.3M 프레임 · 18 pairs · 원전은 "24 different viewpoints" 표기 (동시 카메라 6대와 별개 개념일 수 있음, 🟡) |
| Source | Figshare |
| Status | Multi-animal 3D, future integration |

---

## Backlinks

- [Architecture](architecture.md) — Module map, skeleton registry
- [Model Taxonomy](model_taxonomy.md) — Which models work with which datasets
- [E2E Verification](e2e_verification.md) — Pipeline test results
- [Quick Start](guides/quickstart.md) — How to load and use datasets
- [Benchmark Guide](guides/benchmark_guide.md) — Multi-model comparison

---

*behavior-lab v0.1 | Updated: 2026-02-09*
