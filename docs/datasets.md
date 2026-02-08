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
| Dimensions | 3D (depth sensor) |
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
| 21-22 | left_thumb/tip | left_arm |
| 23-24 | right_thumb/tip | right_arm |

#### Analysis Model
- **Linear Probe** (Supervised Baseline): Mean-pooled keypoints → LogisticRegression

---

### NW-UCLA — Human Action (3D)

| Field | Value |
|-------|-------|
| Species | Human |
| Subjects | 1 person |
| Joints | 20 (Kinect v1) |
| Dimensions | 3D |
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
| Dimensions | 3D |
| Classes | 4: walking, grooming, rearing, standing |
| Format | CSV / NPY |
| Skeleton Name | `subtle`, `subtle_mouse` |
| Reference | Kwon et al. (2022), bioRxiv |
| Source | [GitHub](https://github.com/jeakwon/subtle) |

#### Joint Names

| # | Name | Body Part |
|---|------|-----------|
| 0 | nose | head |
| 1 | head | head |
| 2 | neck | head |
| 3 | body_center | body |
| 4 | hip_left | body |
| 5 | hip_right | body |
| 6 | tail_base | tail |
| 7 | tail_mid | tail |
| 8 | tail_tip | tail |

---

### Rat7M — Rat Motion Capture (3D)

| Field | Value |
|-------|-------|
| Species | Rat (Rattus norvegicus) |
| Joints | 20 |
| Dimensions | 3D |
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
| Dimensions | 3D |
| Classes | 11 movement types |
| Format | MAT / NPY |
| FPS | 60 Hz |
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

**Note**: Raw Zenodo data is video + MATLAB. 3D keypoints may need reconstruction preprocessing.

---

## Investigated (Not Yet Integrated)

### s-DANNCE (Klibaite 2024)

| Field | Value |
|-------|-------|
| Species | Mouse |
| Joints | 23 |
| Features | 9 HLACs (Hierarchical Linear Activity Categories) |
| Scale | 140M+ poses |
| Source | Harvard Dataverse |
| Status | Large-scale, future integration |

### PAIR-R24M (Marshall 2021)

| Field | Value |
|-------|-------|
| Species | Rat (pairs) |
| Markers | 12 per rat (24 total) |
| Categories | 84 fine-grained |
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

*behavior-lab v0.1 | Updated: 2026-02-08*
