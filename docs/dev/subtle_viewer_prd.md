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
- P3b Joint fit (done 261023): one fit over all project sessions, shared `fit_id`; frequency refuses to compare labels from different fits. Open: optional `avatar_preprocess` to match the Justin app (see `subtle_pipeline_io.md` section 4).
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
