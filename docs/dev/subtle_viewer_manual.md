# SUBTLE Lite Viewer manual (261023)

## Start
- Terminal: `subtle <folder> [--port N] [--no-open]` (default port 8777; the folder is scanned recursively).
- Deep link: `http://127.0.0.1:8777/#open=<path>&cam=<0-4>&frame=<n>` opens a session at a frame and camera.

## Files the viewer reads
- Result file: `*.json.gz` with `skeleton3d` (3D, 11 joints) and `keypoint` (2D per camera). No unzip needed.
- Video: per-camera `*.mp4` whose name has the animal token and `cam_N` (for example `..._cort1_cam_2.mp4`). Subfolders do not matter.
- Large originals: make a small copy for display with `tools/subtle_viewer/make_proxy.sh raw.mp4 out.mp4`. The script checks that the frame count is equal.
- Sidecar `<name>.subtle.json` next to the result file: cluster labels, embedding, `fit_id`, `seed`. The original file is never changed.

## Views
- 3D skeleton: wheel = zoom, drag = rotate, shift or right-drag = pan, double-click = reset. The view follows the animal.
- Cluster map: wheel = zoom (around the cursor), drag = pan, click = jump to the nearest frame.
- Video: choose the camera; green points are 2D keypoints with conf above 0.25, missing points are hidden, bones take the cluster color.
- Timeline: one color per frame; click to jump.

## Clusters
- One cluster id has one color in every view.
- `cluster` select: `sub` or `super`. For `super`, the second select lists the levels (`L0: n clusters`); the default is the level with the most clusters.
- Legend chips: click = filter, double-click = solo, `all`, `none`; with a filter, playback skips other clusters; `next bout` jumps to the next run of the selected clusters.

## Running SUBTLE
- Add sessions to groups, then `Run SUBTLE jointly on all project sessions`. One fit covers all sessions, so ids are comparable.
- Set a seed (default 0) for a repeatable fit; empty = random. Max frames limits the time (3,000 frames for 8 animals took about 100 s).
- Frequency table: the sum must be 100 and all sessions must come from one fit; otherwise the badge turns red.

## Limits
- The unit of a group comparison is the animal, not the frame. With 3 vs 5 animals the smallest possible exact permutation p is 0.0357.
- Group names come from folder names; there is no dosing record for the AVATAR `CORT28d` animals.
- 3D points can be copied from the previous frame when fewer than 2 cameras saw them, and no NaN is shown (see `subtle_pipeline_io.md`).
