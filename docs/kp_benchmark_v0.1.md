# kp_benchmark v0.1 — DLC Pretraining Comparison

> Smallest defensible 3D mouse keypoint benchmark integrated into behavior-lab.
> Date: 2026-06-02. Status: scaffold ready, training+inference TBD.

## Hypothesis

SuperAnimal pretraining improves DeepLabCut (DLC) keypoint accuracy on mouse data,
measured on (a) in-distribution MAMMAL M1 held-out test split and
(b) external Li 2023 M1 manual GT (81 total tp, 50 fully-valid).

Training signal = MAMMAL mesh-fit pseudo-GT (dense, low coverage of video timeline).
True OOD evaluation = Li manual labels.

## Design

Two DLC backbones trained on **identical** MAMMAL M1 80/20 split (seed=42):

| Predictor       | Backbone     | Initialization              |
|-----------------|--------------|-----------------------------|
| dlc_resnet50    | ResNet-50    | ImageNet                    |
| dlc_superanimal | HRNet-w32    | SuperAnimal-TopViewMouse    |

Single primary metric: **root-relative MPJPE** (mm) with 10k-bootstrap 95% CI.
Procrustes (`pmpjpe`) available as secondary diagnostic.

## Why this scope (audit-driven)

3-auditor consensus (Haiku/Gemini/o3) flagged the original 4-paradigm plan:
- **C1**: P0 (Li GT) was mis-classified as a "predictor" — it is the evaluation reference.
- **C2**: 4 predictors trained on different data → MPJPE measures data, not method.
- **C3**: DANNCE-pytorch (P2) blocked by private weights + Blackwell incompatibility
  + CPU 1.3 s/frame (Marshall et al. 2022). Dropped to v0.3.
- **C4**: jitter/fps metrics need video sequence — cannot use sparse-only validation.
- **C5**: 14-module skeleton premature before any baseline works.

v0.1 fixes all five by reducing to a **single-variable controlled experiment** that
fits inside existing behavior-lab structure (no new top-level dirs, follows the
`data/loaders/`, `evaluation/`, `configs/dataset/`, `scripts/` pattern).

## Files added (this scaffold)

```
src/behavior_lab/
├── data/loaders/li2023.py            # Li 2023 label3d_dannce.mat sparse 3D loader
├── data/loaders/mammal_mouse.py      # MAMMAL dense (T, 22, 3) npz loader
└── evaluation/mpjpe.py               # root-relative MPJPE + bootstrap CI + Procrustes
configs/
├── dataset/li2023_m1.yaml            # Hydra dataset config
├── dataset/mammal_m1.yaml
└── experiment/kp_dlc_pretraining.yaml
scripts/
├── prepare_kp_splits.py              # deterministic 80/20 + Li OOD CSV
└── benchmark_kp_dlc.py               # orchestrator: predictions → MPJPE + CI → CSV
docs/
└── kp_benchmark_v0.1.md              # (this file)
```

## Data SSOT (gpu03)

| Asset              | Path                                                                     | Size  | Schema |
|--------------------|--------------------------------------------------------------------------|-------|--------|
| MAMMAL M1 dense    | `/home/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz` | 958 KB | `keypoints` (3600,22,3), `frame_indices` (3600,) ∈ {0,5,…,17995} (5-step downsample), `keypoint_names` (22,) |
| Li M1 sparse GT    | `/node_data_2/joon/data/external/markerless_mouse_1/labels/label3d_dannce.mat` | 1.2 MB | `labelData[0].data_3d` (81,66), `data_frame` (81,) ∈ [27,17858] — arbitrary video frames |
| M1 raw video       | `~/data/external/MAMMAL_Mesh_markerless_mouse_1/.../videos_undist/{0..5}.mp4`   | 127 MB | 6 cam, 18000 fr |

### 22-kp ordering (MAMMAL `keypoint_names`)

```
0:L_ear  1:R_ear  2:nose  3:neck  4:body_middle  5:tail_root  6:tail_middle  7:tail_end
8:L_paw  9:L_paw_end  10:L_elbow  11:L_shoulder  12:R_paw  13:R_paw_end  14:R_elbow  15:R_shoulder
16:L_foot  17:L_knee  18:L_hip  19:R_foot  20:R_knee  21:R_hip
```

Root joint candidates for `root_relative_mpjpe`: `body_middle` (idx 4) or `tail_root` (idx 5).

### Frame alignment (critical)

- MAMMAL covers only 3600 of 18000 video frames (every 5th).
- Li GT spans arbitrary frames; only 17/81 happen to fall on MAMMAL grid.
- **Implication**: DLC must train on **video frames directly** (not MAMMAL grid indices),
  using MAMMAL kp as supervision signal. Li GT evaluation uses 81 Li frames as a
  separate inference query — no MAMMAL grid alignment required.
- Generated `data/markerless_mouse_1/labels/li_m1_gt.npz` contains:
  - `keypoints_3d` (81, 22, 3) float32
  - `frame_ids` (81,) int64 — video frame index
  - `valid_mask` (81, 22) bool — `True` where kp manually labeled (50 frames fully valid, 31 partial)

## Reproduce

```bash
# 1) Fetch data (~2.2 MB)
scp gpu03:/home/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
    data/mammal_mouse/v012345_kp22_20260126/
scp gpu03:/node_data_2/joon/data/external/markerless_mouse_1/labels/label3d_dannce.mat \
    data/markerless_mouse_1/labels/

# 2) Generate splits (deterministic)
python scripts/prepare_kp_splits.py \
    --mammal-npz data/mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
    --li-label3d data/markerless_mouse_1/labels/label3d_dannce.mat

# 3) Train DLC ResNet50 + SuperAnimal (TODO — separate scripts, gpu required)
#    Outputs:
#      outputs/kp_benchmark/predictions/dlc_resnet50.npz
#      outputs/kp_benchmark/predictions/dlc_superanimal.npz

# 4) Evaluate
python scripts/benchmark_kp_dlc.py \
    --gt-npz data/markerless_mouse_1/labels/li_m1_gt.npz \
    --pred-resnet50 outputs/kp_benchmark/predictions/dlc_resnet50.npz \
    --pred-superanimal outputs/kp_benchmark/predictions/dlc_superanimal.npz
```

## Explicit exclusions (v0.2+)

| Item                              | Defer to | Reason |
|-----------------------------------|----------|--------|
| DANNCE-pytorch (P2)               | v0.3     | private weights, Blackwell incompat, CPU 1.3 s/fr |
| MAMMAL M2 / M3 sessions           | v0.2     | M1 pipeline validation first |
| 22-kp ↔ 27-kp schema mapping      | v0.5     | YAGNI before second predictor needs it |
| jitter / fps / conf-weighted      | v0.2     | needs video sequence + per-cam inference |
| SLEAP comparison                  | v0.2     | add after pretraining benchmark complete |
| Per-KP MPJPE breakdown            | v0.2     | start with mean only |

## Known caveats (must disclose in any report)

- Li external: 81 total / 50 fully-valid → CI will be wide; null result is ambiguous (underpowered).
- For partial-valid Li frames (31), use `valid_mask` to mask NaN keypoints before metric averaging.
- Training signal is **MAMMAL pseudo-GT**, not human labels — DLC inherits MAMMAL's mesh-fit bias.
- DLC default HPs may favor one backbone — comparison is "out-of-box", not tuned.
- Root-relative metric eliminates global translation only; coord-frame rotation
  between MAMMAL/DLC outputs may need Procrustes if residual logs flag it.

## References

- Li et al. 2023, MAMMAL mesh modeling (PMC10810175).
- Lauer et al. 2022, DLC SuperAnimal (Nature Methods).
- Marshall et al. 2022, DANNCE volumetric (CPU benchmarks).

---

*Created 2026-06-02. Owner: kp_benchmark. SSOT for repo-side design.*
