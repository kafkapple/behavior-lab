# 3D keypoint quality: metrics, gaps and priority plan (261024)

Status: P1 built, P2 partly built, P3 to P8 planned. Code: `tools/subtle_viewer/quality.py`.

## 1. Problem
- The 3D skeleton comes from `recon3D` (top 2 cameras by confidence, triangulation, fill and smoothing; see `subtle_pipeline_io.md`).
- There is no ground truth, no accuracy metric and no agreed visual check for it. Group comparisons rest on this input.

## 2. Measured now (8 sessions, all frames; label-free consistency checks, not accuracy)
- Frames with at least one node identical to the previous frame (a copy): cort 1.0 to 3.6 percent, veh 1.3 to 7.2 percent (veh4 7.2, veh5 6.1).
- In the first 3,000 frames the same share is higher for veh4 (10.8) and veh5 (14.5); the problem sits early in those recordings.
- Frames where any node moves more than half a body length in one frame (50 ms): 6 to 19 percent. This is a heuristic; the half-body-length line is not sourced, so use it to rank sessions, not to judge them.
- Triangulation support (node-frames with 2 or more cameras above confidence 0.25): 98.9 to 99.9 percent. Only 9 to 63 percent of the node-frames below that line are exact copies, so the copy test undercounts weak points.
- Weak-perspective residual of 3D against 2D per camera: about 16 to 35 px of a 1,200 px cell. It is not a calibrated reprojection error.
- Reading: the data is jittery and has group-unequal copy rates. Whether it is accurate is unknown.

## 3. Standard metrics (checked by web search and a grounded model call)
- MPJPE: mean 3D joint error in mm against ground truth from motion capture, multi-view manual annotation or a calibrated rig (Rat7M for DANNCE).
- PCK: share of keypoints within a distance threshold.
- Reprojection error: triangulated 3D points projected back into each calibrated camera (Anipose).
- Bone-length consistency and temporal jitter, as tracking-quality checks.
- No single protocol exists for rodents; papers combine the items above. Thresholds are descriptive until our own ground truth exists.

## 4. Priority plan (each step has a check)
- P1 per-session label-free metrics in the page. Check: table for all sessions matches `quality.py` output; copy test agrees with the 2D support count where both apply.
- P2 group balance gate: warn when the groups differ in copy rate or jump rate; sensitivity runs that exclude copied frames or named animals. Check: run `Sweep n` with the switches; result differences are shown next to the baseline.
- P3 worst-frame review: clips of the worst 30 frames (copy, jump) with 2D overlay on video, rated good or bad by a person in about 30 minutes. Check: precision of each flag.
- P4 small ground truth: annotate about 50 frames in 5 cameras for 3 animals, triangulate with the rig calibration. Needs the calibration files (open question with the previous maintainer). Check: MPJPE and PCK of the current pipeline in mm.
- P5 calibrated reprojection error per camera and node. Needs P4 calibration. Check: error map by node and camera.
- P6 better reconstruction: confidence-weighted use of all cameras, outlier rejection, bone-length and smoothness constraints. Check: P4 metrics before and after on the same frames.
- P7 left-right swap check of limb nodes. Check: precision on P3 review frames.
- P8 rerun clusters and the group tests on the improved 3D with a new `fit_id`. Check: seed stability and sensitivity table.

## 5. Order and reason
- P1 to P3 cost hours and show whether the problem is large enough to matter.
- P4 is the expensive step and the only one that gives accuracy; P5 to P7 need it.
- Until P4 exists, every group result carries the line: quality unverified.
