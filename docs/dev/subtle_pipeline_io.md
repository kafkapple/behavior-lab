# SUBTLE / AVATAR pipeline: stage input and output (261023)

Verified against read-only copies in `~/dev/external/subtle_ref/` (Justin lab repos, access granted 261007).
Line numbers are for the clone of 261023; re-check after a `git pull` there.

## 1. Stages

| # | Stage | Input | Output | Code |
|---|---|---|---|---|
| 0 | Recording | 5 cameras tiled in one mp4 (3600x2000 or 3840x2160, 20 fps) | composite video | `behavior_lab/rig/avatar.py` (layout, split) |
| 1 | Detection (RT-DETR) | video frame | `detection[frame] = [[class, cx, cy, w, h, conf], ...]`, values relative 0-1 | `SUBTLE_gpu/infer_video.py:save_to_json` |
| 2 | 2D keypoint (SLEAP 1423) | crop per camera | `keypoint[frame][camera][node] = [x, y, conf]`, x and y relative 0-1 or absolute | `SUBTLE_gpu/infer_video.py` |
| 3 | 3D reconstruction (`recon3D`) | stage-2 keypoints + camera calibration | `skeleton3d[frame] = 11 x 3`, flag `skeleton3d_extraction: true` | `SUBTLE_gpu/utils/recon3d.py` |
| 4 | SUBTLE train | list of `skeleton3d` arrays (all chosen files) | Mapper model (`.pkl`) | `SUBTLE_storage/celery_tasks.py` (train task) |
| 5 | SUBTLE analysis | Mapper + same kind of arrays | per file `embedding` (Z), `subcluster` (y), `supercluster` (Y); flag `subtle_extraction: true` | `SUBTLE_storage/celery_tasks.py:803-813` |

## 2. What `skeleton3d_extraction: true` means
- It is written by `save_to_json` (`infer_video.py:298`) after `recon3D` ran, so the file holds a 3D skeleton.
- It is a "stage 3 finished" marker, not a quality mark.
- The storage server also uses it to classify an uploaded `.json.gz` as a result file (`storage_server.py:3634`).
- Related keys: `skeleton3d_postprocessing` (a second, post-processed 3D series, written by `SUBTLE_tools`) and `subtle_extraction` (stage 5 done).

## 3. Stage 3 details (why 3D has known defects)
- Per keypoint and frame: undistort the 2D point of each camera, keep cameras with conf > 0.25.
- Fewer than 2 cameras: the point is `[0, 0, 0]`.
- With 2 or more: only the two highest-conf cameras are triangulated (`cv2.triangulatePoints`); the other cameras are not used.
- Result is multiplied by `square_size` and rotated into the world frame (camera 3 as z axis).
- Then three repairs: fill missing first frames, copy missing points from the previous frame, correct sudden jumps.
- Consequence: a point can be frozen (copied) and the file shows no NaN. Do not read "no NaN" as "well tracked".

## 4. Stage 4-5 details
- Input loader: `utils/subtle_utils.py:load_skeleton_data_from_json_gz` reads only `skeleton3d`, flattens 11 x 3 to 33 features, then runs `avatar_preprocess`.
- The official app fits one Mapper on all selected files, then runs it on each file; one model, one cluster id space.
- `avatar_preprocess` (`engine/subtle/utils.py:12`, same code in public `subtle/utils.py:12`) reshapes to (T, joints, 3) and subtracts the mean over frames and joints for each coordinate.
- Our wrapper (`behavior_lab.models.discovery.subtle_wrapper`, `_preprocess`) does the same centering (with `nanmean`). 261023 correction: an earlier version of this note said the wrapper skipped it; that was wrong.
- What can still differ: Mapper parameters (`fs`, `embedding_method`, `n_train_frames`). The app takes them from its form; the viewer uses the wrapper defaults (20 fps, 120,000 frames, umap). The app's default values were not found yet.

## 4b. Randomness and supercluster levels (verified in `engine/subtle/mapper.py` and `module.py`)
- The Justin app sets no seed anywhere (no `seed` or `random_state` in the app, celery tasks or utils).
- Random parts of one fit:
  - `UMAP(n_neighbors=50, n_components=2)` has no `random_state` (`mapper.py:69`).
  - `Phenograph()` clusters the 2D embedding with Leiden (k=30, min cluster size 10); `phenograph.cluster` accepts `seed`, the engine does not pass it.
  - Training frames are sampled with `np.random` when there are more than `n_train_frames` (`mapper.py:147`, `176`).
  - `run_DIB` draws a random beta and a random cluster count k at each of N=1000 iterations (`module.py:269-270`), so the supercluster solutions are random.
- Subclusters: Phenograph decides the number; it was 39-55 in our 8-animal fits.
- Superclusters: `Y` is (frames, levels). Each column is one DIB solution with its own number of clusters (min 2, max 30 in `run_DIB`).
- The app lets the user pick a level (`subtleSuperclusterSize`); the number of superclusters is therefore chosen after the fit, not set before it.
- Our wrapper's default picks the level with the most clusters; the viewer now lists all levels (`L0: n clusters`).
- Why two fits gave 4 and 5 clusters (261023): same 8 animals, first 3,000 frames each, default settings, no seed. The number of levels DIB finds changed from run to run.

## 5. File names
- The app finds the video of a result through database file ids (`resize_file_id`); the fallback strips `_det.json.gz` from the name.
- Our viewer pairs by stem (name minus `_det`, `_sleap_*`), in any subfolder.
- The app inflates `.json.gz` in the browser (`pako`); no manual unzip.

## 6. Open items
- Mapper parameter values used by the app form (`SUBTLE_homepage`); then set the viewer defaults equal.
- Quality of stage-3 output per animal: use the overlay check (PRD P5) before reading cluster differences.
