# Keypoint models on the AVATAR multiview data

Status as of 2026-10-03, except the training package (obtained 2026-10-13, see "Base training labels"). Scope: 2D keypoint predictors that were run, planned or only mentioned on AVATAR
(Justin lab multiview mouse data, 5 cameras). Numbers are copied from the source line; none were recomputed.
Source of the comparison: vault `30_Projects/Behavior-Lab/_Agent/AVATAR/260929_AVATAR_SUBTLE_overlay_compare/260930_AVATAR_keypoint_model_comparison.html` (moved from `2609_IBS_Postdoc/_Agent/`) section 4,
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

## Base training labels (SLEAP 1423 and 1500)

- Source: `20260504_reorganize.pkg.slp`, justingpu2 `~/spkim/coding/sleap/training_job/20260504/`; Mac copy `~/data/avatar_subtle_sleap1423_261003/pkg/`
- Content: 1500 labelled frames, 1 mouse, 11 keypoints, 34 source videos = AVATAR 450 + KHU 772 + JHL 278 (other-lab mice)
- AVATAR 450 = `Baseline Ptd005 Fit` 383 (bot 73, side1 74, side2 73, side3 81, side4 82) + `260505 Sci 1` 67 (bot 55, side1 12)
- The 5 views were labelled at different time points (0-1 shared frame numbers between views)
- n=1423 model: train 1280 + val 142 = 1422 frames, identical to the pkg (frame number and 11 coordinates, tolerance 0.001 px)
- AVATAR share of the n=1423 split = 438 (train 399, val 39); the other 78 frames (AVATAR 12) are only in the n=1500 model (train 1350 + val 150)
- Open (no record found): how frames were selected (the pkg holds 932 SLEAP suggestions; 345 of the 383 `Baseline Ptd005 Fit` frames match them), why KHU and JHL are pooled, the recording dates
- Known defect: about 200 frames have a visible ear point outside the body (14 % of ear points)
- Overlay: 12 samples in `pkg_images/overlay/`; all 34 videos in `pkg_images/overlay_sheets/` (`scripts/pkg_overlay_sheets.py`)
- Details: vault note `261013_AVATAR_sleap1423_pkg_recovery`

## Not run on AVATAR

| Method | State |
|---|---|
| SLEAP retrain / fine-tune | planned after hand labels. Labels and weights are readable on justingpu1 `~/spkim/SUBTLE_gpu/model/SLEAP/260504.single_instance.n=1423-1/` (1280 train + 142 val labelled frames, 438 of them AVATAR; see "Base training labels"). The embedded images (`20260504_reorganize.pkg.slp`) were found on justingpu2 on 2026-10-13 |
| Lightning Pose (+EKS) | planned; installed on justingpu1 (conda env `ltn-pose`, repo `~/spkim/coding/lightning-pose`, seen 2026-10-03, import not tested). Also installs and imports on WSL with `uv run --python 3.10 --with lightning-pose` (lightning_pose 2.4.2, torch 2.14.1+cu130, cuda True; training not run); multi-view LP has no context frames or unsupervised losses yet |
| DANNCE / s-DANNCE | no AVATAR mention found; appear only as datasets and as "dropped to v0.3" in `kp_benchmark_v0.1.md` |
| DeepLabCut own training | none on AVATAR; the DLC scripts here target MAMMAL / Li 2023 |
| MAMMAL, SBeA, PoseSplatter, RTMPose | no AVATAR keypoint use found; RTMPose skipped (AP-10K domain, same as ViTPose++) |

"No mention found" covers the searched paths only (vault, this repo, BehaviorSplatter notes), not the hosts.

## Hand labels for training and for accuracy

- Ground-truth template: 30 frames x 5 cameras x 11 keypoints = 1650 rows
  (`gt_label_261002/labels_template.csv`: `frame,camera,image,image_w,image_h,keypoint,x_px,y_px,visible,note`). Labelling not started as of 2026-10-03; current state = vault handoff `261011_Handoff_AVATAR_keypoint_mask_filter`.
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
- `scripts/kp_compare.py` (assemble, boxes, score) and `scripts/kp_label_page.py` (human labelling page, no predictions shown), `scripts/slp_label_report.py` + `pose/slp_labels.py` (SLEAP label files without images: counts, outside-image points, train/val leakage, L/R check, dashboard), `kp_compare.py report` (comparison page)
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
4. Retrain SLEAP only if access to the training package is obtained (condition met 2026-10-13, see "Base training labels"; the retrain decision is not recorded in this file)
