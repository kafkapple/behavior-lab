# SUBTLE Lite Viewer manual (261023)

## Start
- Terminal: `subtle <folder> [--port N] [--no-open]` (default port 8777; the folder is scanned recursively).
- Deep link: `http://127.0.0.1:8777/#open=<path>&cam=<0-4>&frame=<n>` opens a session at a frame and camera.

## Files the viewer reads
- Result file: `*.json.gz` with `skeleton3d` (3D, 11 joints) and `keypoint` (2D per camera). No unzip needed.
- Video: per-camera `*.mp4` whose name has the animal token and `cam_N` (for example `..._cort1_cam_2.mp4`). Subfolders do not matter.
- Large originals: make a small copy for display with `tools/subtle_viewer/make_proxy.sh raw.mp4 out.mp4`. The script checks that the frame count is equal.
- Sidecar `<name>.subtle.json` next to the result file: cluster labels, embedding, `fit_id`, `seed`. The original file is never changed.

## Screen
- One screen, no page scroll. Left: sessions. Center: video with 2D keypoints (tall), 3D skeleton, cluster map. Right: tabs Clusters, Project, Frequency, Stats, Log. Bottom: play, slider, timeline, key list.
- Top toolbar: Clusters (level), Fit (frames, seed, Run), Help (manuals).

## Views
- 3D skeleton (follows the animal): wheel = zoom, drag = rotate, shift or right-drag = pan, double-click or Enter = back to the selected camera's view.
- The default 3D pose is the selected camera's view: the server fits a weak-perspective camera from the 2D keypoints to the 3D points (`core.cam_view`) and rotates the skeleton to it. The footer prints the camera; "free view" means no fit was possible.
- Cluster map: wheel = zoom (around the cursor), drag = pan, click = jump to the nearest frame, double-click = reset.
- Video: pick the camera in the pane header or press 1-5; press 0 for all cameras with video at once (click a tile to select it). Only cameras whose video file sits next to the data appear.
- Keypoints: green filled = conf above 0.25. Hollow green and dashed bones = low confidence, shown dim. Nothing is drawn only when the point is missing (x = y = 0, conf 0). Bones take the cluster color. Key `k` toggles the overlay.
- Why a bone can disappear: it is drawn only when both end points exist in that camera's 2D track. The 3D view always draws every bone because the 3D track fills missing points from earlier frames.
- Timeline: one color per frame; click or drag to scrub, also while playing. A hatched region means no label exists there: labels cover only the frames that were fitted (frames/session). Map the session with a saved model to label all frames.

## Clusters
- One id has one color in every view (Tableau palette). Ids mean something only inside one fit; the legend shows the `fit_id`.
- `cluster` select: `sub` or `super`. For `super` the second select lists the levels (`L0: n clusters`); the default is the level with the most clusters. Why levels: `subtle_theory.md`.
- Clusters tab (right): one row per id with color, name field and share. Click = filter, double-click = solo, `all`, `none`. With a filter, playback skips other clusters and `next bout` jumps to the next run.
- Names: type in the name field; it is saved per fit and level in `<root>/.subtle_projects/cluster_names.json` and shown in the Stats tab.

## Project metadata
- Project tab: a description box, a note per group, and animal, cond.(ition), date per session. `Save` writes them into `<name>.project.json` under `meta`; the server keeps only these fields, 200 characters each.
- Deep links for checks: `#open=<path>&cam=1&frame=600&all=1&project=<name>&tab=proj` (all = every camera tile; tab = clu, proj, freq, stat, log).

## Running SUBTLE
- Add sessions to groups, then `Run SUBTLE jointly on all project sessions`. One fit covers all sessions, so ids are comparable.
- Set a seed (default 0) for a repeatable fit; empty = random. What the seed controls: `subtle_theory.md` section 4. Max frames limits the time (3,000 frames for 8 animals took about 100 s).
- Model: `Train` fits the project sessions once and saves `<root>/.subtle_models/<name>/model.pkl` and `model.json`. `Map` loads a saved model and labels every frame of the project sessions with it; the log prints how many trained frames keep their label (about 85 to 88 percent in a 2-session test, because mapping classifies new points).
- Frequency table: the sum must be 100 and all sessions must come from one fit; otherwise the badge turns red.

## Statistics
- Stats tab, `Compare groups`: for each pair of groups, the share of each cluster per animal, exact permutation p of the mean difference, Hedges g, Benjamini-Hochberg adjusted p, and stars (* below .05, ** below .01, *** below .001, on the adjusted p). One dot = one animal.
- The page prints the smallest possible p for the group sizes; with 3 vs 5 animals it is 0.0357, so ** and *** cannot appear.
- Demo (8 animals, cort 3 vs veh 5, level L4, first 3,000 frames): adjusted p 0.77 to 1.0, no star. Groups are provisional (folder names).
- DEK tab, `Compute DEK`: same steps as the Justin app, with our 11-node layout by node name.
  - Frames are centered like `avatar_preprocess`, then the public `subtle.Kinematics` gives 84 features: speed, edge and angle terms, each as value (n=0), first (n=1) and second (n=2) time derivative.
  - Per animal and cluster the features are averaged; an animal needs at least 30 frames in a cluster. Welch t per cell (scipy `ttest_ind`, same call as the app), Benjamini-Hochberg over all cells of a pair, Hedges g.
  - Heatmap: rows = clusters, cell = signed t (orange = first group higher, blue = second), `*` on adjusted p; hover shows p, adjusted p, g. Pick type (vertex, edge, angle), derivative and pair above the table.
  - Demo (cort 3 vs veh 5, level L4, 420 cells): smallest adjusted p 0.38, no star. Shuffling the 8 animals over all 56 splits gave 4.6 percent raw p below 0.05, as expected for no effect.
  - Not excluded: frames whose 3D points were copied from earlier frames (the viewer's loader reads no per-frame flag). Speed terms of such frames are zero, so read speed differences with care.

## Level sweep and saved results
- Question it answers: which number of superclusters (level) shows a group difference, without fooling ourselves by trying many levels.
- "Most clusters" is not a valid rule: it ignores the data. The sweep compares every level on one criterion chosen before looking (select next to `Sweep n`).
  - `omnibus` (default): PERMANOVA pseudo-F on Hellinger-transformed shares of all clusters together. One number per level; handles the sum-to-100 constraint; no cluster multiplicity.
  - `maxt`: the largest |Welch t| of any cluster.
  - `nsig`: number of clusters whose raw permutation p is below .05.
  - `meanp`: mean raw permutation p over the clusters (lower is better).
- Null and selection correction: every criterion is a statistic with its own null over all splits of the animals into groups of the observed sizes (exact up to 20,000 splits). Levels are compared by the z-score against that null; the best level's z is referred to the null of the best level over all splits, which gives p_fw (Westfall-Young max-T logic applied across levels). Cluster flags use max-T over all levels and clusters.
- Why this is valid here: the clustering never sees the group labels, so permuting labels does not require refitting. Conditions: units exchangeable under the null, criterion chosen beforehand, levels from one fit. `/fact --deep` (Gemini, GPT, web) agreed on these conditions; PERMANOVA also reacts to unequal group spread, and zero shares need Hellinger rather than Aitchison.
- Reading it: if p_fw is not below .05, the page says in red that no difference should be reported. With 3 vs 5 animals the smallest possible p is 0.0179 (1 of 56 splits).
- `Use n=k` switches the level; the legend marks significant clusters with ★ and exploratory ones (raw p below .05, not significant) with ☆. `Show flagged only` filters the map, timeline and replay to them.
- `Export results` writes `outputs/subtle/<project>/<time>_n<k>_<criterion>/` (server option `--results-dir`; `summary.json` records the code commit): `sweep_levels.csv` (all criteria, plus the recommended n per criterion), `clusters.csv`, `summary.json`, `sweep.png`, and `cluster_<id>_<tag>.mp4` (up to 3 bouts per group from the proxy video, 2D keypoints and a caption). If nothing is significant, the 2 clusters with the largest |t| are clipped and tagged `exploratory_not_significant`. A group without per-camera video is skipped and the log says so.
- Checks: planted effect found, hand-known p (2 of 70), and all 56 assignments of the real 8 animals gave 2 false positives (3.6 percent) for each of the four criteria.
- Demo (cort 3 vs veh 5, first 3,000 frames): omnibus n = 4 (p_fw 0.39), maxt n = 4 (0.50), nsig n = 2 (1.0), meanp n = 2 (0.57). No criterion is significant.

## Proxy videos for all sessions
- `python tools/subtle_viewer/fetch_cams.py <root> --project <name>` makes five camera proxies per session next to the session file. The 5-camera composite (about 10 GB) is copied from the server one session at a time, cut into the five cells in one decoding pass (cells from `core.CELLS`), written half size (H.264 CRF 28, keyframe every 20 frames), checked against the session's frame count, and then deleted.
- Safe to re-run: sessions with all five proxies are skipped. Needs 14 GB free. Overlay viewing only; inference must use the raw video.
- Defaults point at the CORT 28d composites on the lab server; change `--remote-dir` and `PREFIX` in the module for another dataset. Time: about 25 to 30 minutes per session.

## Result retention
- `python tools/subtle_viewer/clean.py` lists result folders beyond the latest 3 per project; a file named `KEEP` inside a folder protects it. `--apply` deletes the listed folders.

## Seed stability
- Stats tab, `Seed stability`: refits the project sessions with seeds 0, 1, 2 (isolated, sidecars untouched; about 80 s per seed for 8 animals at 3,000 frames) and saves `<name>.stability.json`.
- Shows the adjusted Rand index (ARI) between seeds for subclusters and for the default supercluster level, and the group test at every level (min adjusted p, number of clusters below .05), so no level is chosen after seeing the stars.
- `stable` means every seed agrees on whether any cluster passes .05 at its default level; it can also mean a stable null. `seed-dependent` means they disagree.
- Demo (8 animals, first 3,000 frames): ARI 0.41 to 0.45 for subclusters (61 to 66 clusters) and 0.40 to 0.60 for the default supercluster level. No seed gave a cluster below .05. The partition itself is only partly reproducible across seeds, so keep the seed fixed and say which seed a figure used.

## Keys
- Same keys as the BS viewer: Space play, Left/Right 1 frame, Up/Down 10 frames, `,` `.` 1 frame, `+` `-` zoom skeleton, Enter reset view, `h` hide side panels, `?` manuals, `Esc` clear filter.
- SUBTLE viewer additions: `1`-`5` camera, `0` all cameras, `k` keypoints, `l` Clusters tab.

## Limits
- The unit of a group comparison is the animal, not the frame. With 3 vs 5 animals the smallest possible exact permutation p is 0.0357.
- Group names come from folder names; there is no dosing record for the AVATAR `CORT28d` animals.
- 3D points can be copied from the previous frame when fewer than 2 cameras saw them, and no NaN is shown (see `subtle_pipeline_io.md`).

## More
- Theory (DIB, levels, randomness): `subtle_theory.md`. Stage input and output: `subtle_pipeline_io.md`. Planned model, DEK and statistics: `subtle_stats_plan.md`. Plan and design basis: `subtle_viewer_prd.md`.
