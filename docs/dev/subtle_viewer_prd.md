# SUBTLE Lite Viewer PRD (261023)

## 1. Goal
- Local web viewer for SUBTLE results: load `*.json.gz`, play 3D skeleton, run SUBTLE, compare groups.
- Does not copy Justin's Meteor app. Reuses `behavior_lab.models.discovery.SUBTLE` and the file format only.
- Start: `subtle-view [dir]` (alias) opens `http://127.0.0.1:8777`.

## 2. Reference (read-only copies)
- Path: `~/dev/external/subtle_ref/` (`SUBTLE_homepage`, `SUBTLE_storage`, `SUBTLE_gpu`, `SUBTLE_tools`, `subtle`).
- Format facts (from `infer_api.js`, `celery_tasks.py:803-813`, sample `cort1_full_sleap_1423.json.gz`):
  - gzip JSON. Keys: `node.id` (11 names), `node.line` (10 edges), `skeleton3d` {frame: 11x3}.
  - SUBTLE adds `subtle_extraction: true` and `subtle` {`embedding`, `subcluster`, `supercluster`}.
  - The app inflates in the browser (`pako`). No manual unzip.
  - The app pairs video and JSON by DB file id; fallback = file stem minus `_det.json.gz`.

## 3. Requirements
- R1 Browse: recursive scan of a root dir; subfolders do not matter. Pair video by stem.
- R2 View: 3D skeleton playback (frame slider, speed), cluster color per frame.
- R3 Analyze: run SUBTLE on `skeleton3d`, write a sidecar `<stem>.subtle.json` (original untouched).
- R4 Cluster color: one fixed color per cluster id, identical in every view and group.
- R5 Groups: group = named list of sessions inside a project. Save and load as `<name>.project.json`.
- R6 Project metadata: name, description, per-group note, per-session fields (animal, condition, date).
- R7 Frequency: per group, per cluster pct; the sum must be 100 (tolerance 1e-6) or the UI shows an error.
- R8 Progress: every job shows stage, percent, elapsed. Debug log panel shows the server log tail.
- R9 Resource safety: LRU cache max 2 sessions, job list max 20, subprocess timeout, files closed with `with`.

## 4. Non-goals
- No login, no multi-user, no upload to Justin's server, no DB.
- Phase 3 (video to keypoints with SLEAP on eren) is out of the first release. Gate: user approval for GPU use.

## 5. Phases and checks
- P1 Server and viewer: `GET /api/sessions` lists the sample; skeleton renders. Check: `selftest` passes.
- P2 Groups, projects, frequency: save then load returns the same project; frequency sums to 100. Check: assert in selftest.
- P3 SUBTLE job: sidecar written; progress reaches 100. Check: run on a 3,000-frame slice of `cort1`.
- P3b Joint fit (done 261023): one fit over all project sessions, shared `fit_id`; frequency refuses to compare labels from different fits. Preprocessing already matches the app (same centering, verified 261023). Open: Mapper parameter values of the app form (see `subtle_pipeline_io.md` section 4).
- P4 Alias (done 261023): shell function `subtle [dir] [--port N] [--no-open]` in `~/.zshrc`; the name `subtle` was free.
- P5 Overlay check (planned, reuse first): 2D keypoints over the 5-camera video, to judge stage-3 quality before reading clusters.
  - P5.1 Split composite into camera cells: reuse `behavior_lab.rig.avatar` (layout, `split`). Check: cell sizes equal the rig yaml.
  - P5.2 Draw keypoints and skeleton: reuse `visualization.video_overlay.overlay_keypoints_on_frame_array`. Check: one known frame matches the vault overlay dashboard.
  - P5.3 Color: bone color from `visualization.colors` (body part); border or timeline color from the cluster id function of the viewer. One id, one color.
  - P5.4 Playback block: evaluate `visualization.player.player_block` (3D skeleton, cluster map, ethograms in sync) to replace the viewer's own canvas.
  - P5.5 Quality flags per frame: mark frames where stage 3 had fewer than 2 cameras (point copied from the previous frame). Needs `keypoint` conf from the json.gz.
  - Each step is its own commit and its own check; the next starts after the check passes.
- P6 Group design (planned): the unit of comparison is the animal, not the frame.
  - Exact permutation test, minimum two-sided p by group size: 3 vs 3 = 0.100, 3 vs 5 = 0.0357, 4 vs 4 = 0.0286, 5 vs 5 = 0.0079.
  - The 8 `CORT28d` animals (cort1-3, veh1-5) have `skeleton3d` on eren.
  - Group names come from folder names, not a dosing record.

## 6. Risks
- 🟡 SUBTLE needs `umap` and `phenograph`; macOS can SIGSEGV, so the job runs isolated (`isolate=True`).
- 🟡 Full `cort1` (12,000 frames) runtime is unmeasured.
- 🟡 `skeleton3d` quality in AVATAR has known hindleg and tail defects (see vault AVATAR 3D defects note).

## 7. Status and difference from the Justin app (261023)
Legend: ✅ done and checked, 🟡 done, not checked in a browser, ⬜ not started, ➖ not planned.

| Function | Justin app (`SUBTLE_homepage` + storage + gpu) | SUBTLE Lite Viewer |
|---|---|---|
| Login, upload, file database | oauth, MinIO, file ids | ➖ local folder scan |
| Video to keypoints (stages 1-3) | GPU server (RT-DETR, SLEAP, `recon3D`) | ➖ out of scope; input is a finished `.json.gz` |
| SUBTLE train and analysis | two tasks; model saved as `.pkl` | ✅ one joint fit (train + analysis), sidecar per session |
| Cluster levels | subcluster and supercluster mode | 🟡 both, switch in the page |
| 3D skeleton playback | modal with frame slider, speed, grid | 🟡 canvas, frame slider, drag rotate |
| Video next to skeleton | by file id | ✅ by file stem; cam_2 proxy checked with a real video |
| Group comparison | DEK tab: group A vs B, per cluster and kinematic feature, Welch t-test, heatmap (`calculateDekStatistics`) | 🟡 saved groups, cluster-share permutation test with stars (P11); kinematic DEK not yet (P10) |
| Frequency table with sum check | not found in the app | ✅ sum = 100 and one-fit check, asserted in selftest |
| Cluster id to color | per page | ✅ one function for all views |
| Overlay on video (2D points on 5 cameras) | not found | ✅ 2D keypoints on the camera video, dim for low confidence; all-camera tiles built, only checked for cam_2 |
| Progress and debug log | task status polling | 🟡 job stage and percent, log tail panel |

## 8. Plan for generalization (261023, design only; nothing below is built)
Goal: the viewer must not stay tied to SUBTLE or to the AVATAR 5-camera rig.

### P7 Triangulation as a module
- Reuse `behavior_lab.rig.multiview` (`load_calib`, `triangulate` = DLT over n cameras by SVD, `reproj_px`).
- Interface: `triangulate_sequence(kp2d, calib, method)`, `kp2d` = (T, cameras, nodes, 3) with x, y, conf; returns (T, nodes, 3) plus the number of cameras used per point.
- Methods: `dlt_all` (all cameras above a conf threshold, general), `top2` (Justin's `recon3D`, for comparison only).
- Check: on one AVATAR clip, `dlt_all` vs the file's own `skeleton3d`; report reprojection residual per method.
- Rig info comes from a yaml (like `configs/rig/avatar.yaml`), so another rig is a new yaml, not new code.

### P8 Clustering as a plug-in
- Existing parts: `behavior_lab.models.get_model(name)`, wrappers for `SUBTLE`, `KeypointMoSeq`, `BehaveMAE`, `BSOiD`, `VAME`, and `scripts/isolated_run.py` (one method in its own env, writes `labels.npy` and `features.npy`).
- Facts that shape the design:
  - Methods conflict in one environment, so each runs in its own process or env.
  - SUBTLE takes a list of sequences; the repo protocol `BehaviorClusterer` takes one array. The viewer needs the list form (joint fit).
  - Output rate differs (B-SOiD 10 fps vs 20 fps): labels must be aligned to the frame count before frequency is computed.
- Interface (viewer side): `run(method, sessions, params) -> {labels per session, embedding per session or None, fit_id}`.
- Sidecar stays one file per session; `method` is recorded next to `fit_id`; frequency compares only labels with the same method and `fit_id`.
- Order: SUBTLE (done) -> B-SOiD (CPU, small) -> keypoint-MoSeq (JAX, own env) -> hBehaveMAE (GPU; needs a go on GPU use).
- Check per method: label count equals frame count for every session; pct sum = 100; same `fit_id` in one comparison.

### Status of the viewer features requested 261023
| Feature | State |
|---|---|
| Zoom and pan of the skeleton | 🟡 wheel zoom, shift or right-drag pan, double-click reset |
| Cluster map (embedding scatter) | 🟡 zoom, pan, click to jump; needs the embedding stored by the job (stored from 261023) |
| 2D keypoints on the raw video | 🟡 camera cell mapping checked on one frame (points on the mouse); bones colored by cluster |
| Note: the Justin app has no keypoint-on-video overlay | video is shown next to the 3D skeleton only (`infer_api.js` has no drawing on the video) |
| Reproducibility | ✅ with `seed=0` (numpy, random, UMAP, Phenograph) two joint fits of the 8 animals gave identical sub and super labels (55 subclusters, 5 levels, 261023). Earlier unseeded fits differed (4 vs 5 levels). Note: the first 'seeded' tests were not seeded because the seed did not reach the isolated subprocess; fixed in `SUBTLE.__init__`. |
| Open: video in a real browser | headless capture showed the overlay but a black video area; Range serving and codec (H.264 High, yuv420p) checked by request, playback itself not confirmed |

### Further differences found 261023
| Function | Justin app | SUBTLE Lite Viewer |
|---|---|---|
| Train a model, save it, load it, map new files | train task stores the Mapper as `.pkl`; analysis task loads it (`celery_tasks.py`) | ⬜ P9 (the wrapper already has save, load, predict) |
| Supercluster size choice | select box over the DIB levels (`subtleSuperclusterSize`) | 🟡 level select (`L0: n clusters`) |
| Cluster select all or none, replay | `selectAllClustersBtn`, `selectNoneClustersBtn`, `clusterSelectionInput` | 🟡 click chips, double-click solo, replay skips other clusters, next bout |
| Kinematics (angle, edge, vertex; DEK) | `Kinematics` class, per-cluster means, group heatmap | ⬜ P10 |
| Significance stars | not found (Welch t-test p-value is computed with a self-made t CDF approximation; no multiple-comparison correction) | ⬜ P11 |
| Seed | none anywhere in the app | 🟡 seed field (default 0) |
| Video next to skeleton | `View switch` button shows video or skeleton, an `_resize.mp4` copy is used | 🟡 video with 2D keypoint overlay (the desktop `SUBTLE_tools/label_editor.py` has a Skeleton overlay on video) |
| Manuals | none in the app | 🟡 Manuals list in the page |

## 9. Stages requested 261023
Details, checks and residual risks: `subtle_stats_plan.md`.

| Stage | What | State |
|---|---|---|
| P9 | train, save, load, map new files | ✅ checked on 2 sessions (51 s train, all 11,998 frames mapped, 85 and 88 percent of trained frames keep their label) |
| P10 | kinematics by node name, DEK group test | ⬜ |
| P11 | cluster frequency statistics and plots with stars | ✅ table, dots, stars, CSV; PNG export not built |
| P12 | seed stability across seeds, all levels shown | ⬜ |
| P13 | manuals readable in the page | ✅ |

## 10. UX basis and module structure (261023)
Sources read: BS scene-flow viewer (`src/behaviorsplatter/web/`, `docs/code/viewer_controls_audit.md`) and the HTML report rule (`html-report-pages.md`).

### What the BS viewer does that the first Lite page did not
- One viewport, no page scroll: top toolbar, side panels, bottom dock; panels collapse on narrow screens.
- Toolbar in labelled groups; one home per control; no value copied into a second place.
- Modules by question: data (what is open), view (what is drawn), measurement (how numbers arose), output (what leaves).
- A small token set (colors, type scale) in one stylesheet; JS split by role.
- Progressive disclosure: rare settings behind one extra step.

### Adopted in `tools/subtle_viewer/web/`
| File | Role |
|---|---|
| `index.html`, `style.css` | shell and tokens (BS palette and type scale) |
| `js/state.js` | shared state, event bus, label level and filter, canvas fit |
| `js/colors.js` | one id to one color, for every view |
| `js/views.js` | 3D skeleton, cluster map, timeline |
| `js/video.js` | per-camera video and 2D keypoint overlay (letterbox aware) |
| `js/legend.js` | cluster chips, filter, replay, next bout |
| `js/project.js` | groups, save and load, fit job, frequency table |
| `js/main.js` | session list, playback, tabs, manuals |

- Layout: video (tall) left, skeleton and cluster map right, sessions left rail, project panel right, dock below with slider, timeline and legend.
- Cluster ids: Tableau 10 then Tableau 20 colors, golden-angle hue beyond 20; the legend shows the `fit_id` because an id means something only inside one fit.
- Not adopted yet: keyboard shortcuts, long-press sub-menus, collapsible panels on wide screens.
