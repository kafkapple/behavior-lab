# Keypoint models on the AVATAR multiview data

Status as of 2026-10-03. Scope: 2D keypoint predictors that were run, planned or only mentioned on AVATAR
(Justin lab multiview mouse data, 5 cameras). Numbers are copied from the source line; none were recomputed.
Source of the comparison: vault `30_Projects/2609_IBS_Postdoc/_Agent/260929_AVATAR_SUBTLE_overlay_compare/260930_AVATAR_keypoint_model_comparison.html` section 4,
plus `kp_model_selection` (status: plan). UNVERIFIED = read in a note, not confirmed on the host.

## Protocol of every row below

- 9 clips (2023-12 to 2025-05), all 5 cameras, 30 s clips, no ground-truth labels
- Reprojection = DLT over at least 3 cameras, conf >= 0.5, in cell pixels
- The ranking is label-free. It shows stability, not accuracy. Accuracy needs the hand labels (see below)

## What was run on AVATAR

| Method | What was done | Key numbers (clip 2023-12-05) |
|---|---|---|
| SLEAP single-instance n=1423, 11 keypoints | SUBTLE weights, inference only | detected 87.8 %, conf >= 0.5 65.1 %, reprojection median 9.0 px (n=4937), shuffled control 151.3 px |
| SLEAP tailless n=1501, 9 keypoints | inference only, used as SAM3 prompts | 85.5 %, 66.8 %, 8.3 px (n=4208) |
| SLEAP top-down n=1501 | not run on AVATAR | own val only |
| YOLO11m (AVATAR3D, balbc, KHU-527) and RT-DETRv2 | inference only, box-to-keypoint rule | reprojection 7.3 to 9.2 px |
| Legacy YOLOv4 (Darknet) | existing sidecar files only | too few points to use (n=4) |
| DLC SuperAnimal-Quadruped (HRNet-w32) | zero-shot, 150 images, WSL | nose 5.9 px from SLEAP; neck, ears, paws 12 to 26 px |
| Anipose | 3D smoothing on SLEAP output, not a 2D detector | large jumps (> 20 mm) 2.7 -> 0.3 % |

## Not run on AVATAR

| Method | State |
|---|---|
| SLEAP retrain / fine-tune | planned after hand labels. Labels and weights are readable on justingpu1 `~/spkim/SUBTLE_gpu/model/SLEAP/260504.single_instance.n=1423-1/` (1280 train + 142 val labelled frames, 289 of them AVATAR). The embedded images (`20260504_reorganize.pkg.slp`) were not found there; source videos are `D:/spkim/data/AVATAR_split/*` (checked 2026-10-03) |
| Lightning Pose (+EKS) | planned; installed on justingpu1 (conda env `ltn-pose`, repo `~/spkim/coding/lightning-pose`, seen 2026-10-03, import not tested); multi-view LP has no context frames or unsupervised losses yet |
| DANNCE / s-DANNCE | no AVATAR mention found; appear only as datasets and as "dropped to v0.3" in `kp_benchmark_v0.1.md` |
| DeepLabCut own training | none on AVATAR; the DLC scripts here target MAMMAL / Li 2023 |
| MAMMAL, SBeA, PoseSplatter, RTMPose | no AVATAR keypoint use found; RTMPose skipped (AP-10K domain, same as ViTPose++) |

"No mention found" covers the searched paths only (vault, this repo, BehaviorSplatter notes), not the hosts.

## Hand labels for training and for accuracy

- Ground-truth template: 30 frames x 5 cameras x 11 keypoints = 1650 rows
  (`gt_label_261002/labels_template.csv`: `frame,camera,image,image_w,image_h,keypoint,x_px,y_px,visible,note`). Labelling not started.
- Trainers read different files, so `behavior_lab.pose.labels` turns that one table into them:

| Trainer | Function | File |
|---|---|---|
| DeepLabCut, Lightning Pose | `to_dlc_csv` | `CollectedData_<scorer>_<camera>.csv`, one per camera |
| SuperAnimal fine-tune, mmpose, YOLO-pose | `to_coco` / `write_coco` | COCO keypoints json (visibility 2 / 1 / 0) |
| SLEAP | `to_sleap` (needs `sleap` extra) | `.slp`, one image-sequence video per camera |
| DANNCE | not produced | needs 3D points and calibration (`label3d`) |

- Keep ground-truth labels out of any training package, otherwise the accuracy comparison is invalid.
- Install the exporter dependencies with `pip install behavior-lab[labels]` (+ `[sleap]` for `.slp`).

## Already in this repo

- SLEAP import only: `src/behavior_lab/pose/sleap.py`; loaders `sleap`, `sleap_h5`, `sleap_slp`
- DLC: scaffolds for MAMMAL / Li 2023 (`scripts/01_train_dlc_resnet50.sh` to `04_zeroshot_superanimal.sh`, `benchmark_kp_dlc.py`)
- `scripts/sbea_export_dlc_csv.py` writes predictions (with likelihood) for the SBeA triangulator; it is not the label exporter
- AVATAR rig (calibration, DLT, residuals): `src/behavior_lab/rig/avatar.py`, `docs/avatar_rig.md`
- `src/behavior_lab/pose/predictors/`: long prediction table + SUBTLE/DLC adapters, body-part map, agreement and reprojection,
  model registry (15 candidates with status), runners `run_dlc.py` (SuperAnimal) and `run_vitpose.py` (ViTPose++ AP-10K)
- `scripts/kp_compare.py` (assemble, boxes, score) and `scripts/kp_label_page.py` (human labelling page, no predictions shown)
- Not wrapped as modules: SLEAP, YOLO and RT-DETR inference (their outputs are read from SUBTLE json), Lightning Pose, DANNCE

## Zero-shot run on the 150 GT images (2026-10-03, label-free)

- 9 predictors on the same images: 6 existing + SuperAnimal-Quadruped + SuperAnimal-TopViewMouse + ViTPose++ AP-10K
- Median distance to SLEAP 1423 (px): SuperAnimal-Quadruped nose 5, tail base 9; TopViewMouse nose 6, tail base 8; ViTPose++ nose 51, tail base 182
- Reprojection median (px): SLEAP 1423 8.6, SuperAnimal-Quadruped 9.3 (90 of 790 candidates triangulated), TopViewMouse 10.2, ViTPose++ 60.3
- Pitfalls found: a DLC video must be built from the selection images only (an earlier 30-frame page set is disjoint from the GT frames and gave 132 px);
  the ViTPose++ checkpoint's id2label is COCO-17 order, the AP-10K expert needs the mmpose AP-10K order
- Page: vault `260929_AVATAR_SUBTLE_overlay_compare/261003_AVATAR_keypoint_predictor_comparison.html`

## Priority of what remains

1. Hand-label the 150 images (human). Unlocks accuracy for every row above
2. Score SLEAP 1423, SLEAP 1501 and SuperAnimal-Quadruped against those labels (same images, same metric)
3. Install Lightning Pose on WSL and pilot on the labelled frames
4. Retrain SLEAP only if access to the training package is obtained
